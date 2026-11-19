"""Visualize KAE and state-conditioned Grassmann KAE models."""

from __future__ import annotations

import argparse
import json
import sys
from xml.etree import ElementTree
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib import animation
from matplotlib.collections import LineCollection
from matplotlib.ticker import FixedLocator
from matplotlib.colors import hsv_to_rgb
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
REACHING_PLOT_HOLDOUTS = 64
REACHING_TARGET_PALETTE = (
    "#486A8C",  # slate blue
    "#4F8178",  # muted teal
    "#738A5B",  # sage
    "#A08A55",  # ochre
    "#A7685F",  # terracotta
    "#7B668C",  # muted plum
)
ROLLOUT_ANIMATION_MAX_FRAMES = 240
ROLLOUT_ANIMATION_MAX_POINTS = 160
HEATMAP_ANIMATION_MAX_FRAMES = 120
DYSTS_PLOT_HOLDOUTS = 12
DYSTS_GRASSMANN_RGB_POINT_BUDGET = 120000
GRASSMANN_05C_IMAGE_PIXELS = (1400, 1100)
DYSTS_GENERIC_ROLLOUTS = 3
DYSTS_DEFAULT_VISUALIZATION_STEPS = 2000
DYSTS_3D_SYSTEMS = {
    "chua", "dadras", "lu_chen_cheng", "lu-chen-cheng", "luchencheng",
    "shimizu_morioka", "shimizu-morioka", "shimizumorioka",
}

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    # Use the repository-qualified module name. A generic ``from utils`` can
    # resolve to an unrelated module already present in sys.modules/sys.path.
    from src.utils import make_data
    _MAKE_DATA_IMPORT_ERROR = None
except Exception as exc:
    make_data = None
    _MAKE_DATA_IMPORT_ERROR = exc

from src.model.koopman_baseline import KoopmanAutoencoder

from src.utils import (
    MULTISTABLE_3D_SYSTEMS as MULTISTABLE_3D,
    canonical_system,
    GOODWIN_NAMES,
    PAM_NAMES,
    REACHING_NAMES,
    get_device,
    denorm,
)

from src.utils import (
    pamy_checkpoint_config, pamy_visualization_arrays, pamy_visualization_indices,
    pamy_workspace_windows,
)

CURRENT_MODEL_CLASSES = {
    "KoopmanAutoencoder": KoopmanAutoencoder,
}


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def strip_state_prefixes(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for key, value in state.items():
        k = key
        for prefix in ("_orig_mod.", "module."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        out[k] = value
    return out


def _fit_projector_color_embedding(projector_features, rank):
    """Fit mean-centered projector PCs and select a central reference subspace."""
    center = projector_features.mean(axis=0, keepdims=True)
    centered = projector_features - center
    _, _, directions = np.linalg.svd(centered, full_matrices=False)
    directions = directions[:3].copy()
    for channel, direction in enumerate(directions):
        pivot = int(np.argmax(np.abs(direction)))
        if direction[pivot] < 0:
            directions[channel] *= -1.0
    reference_index = int(np.argmin(np.linalg.norm(centered, axis=1)))
    return center, directions, projector_features[reference_index], int(rank)


def _apply_projector_color_embedding(projector_features, embedding):
    """Map the first projector PC to hue with fixed saturation and brightness."""
    center, directions, reference, rank = embedding
    coordinates = (projector_features - center) @ directions.T
    coordinates = np.pad(
        coordinates, ((0, 0), (0, max(0, 3-coordinates.shape[1]))),
    )[:, :3]
    # Projector Frobenius distance is bounded by sqrt(2 rank). Unlike
    # percentile scaling, this does not amplify tiny subspace differences.
    rgb_contrast = 50.0
    # A smooth mapping avoids clipping both extremes to the same red hue.
    hue_coordinate = 0.5 + np.arctan(
        rgb_contrast * coordinates[:, 0] / np.sqrt(2.0 * max(rank, 1))
    ) / np.pi
    # Use the full hue wheel, with no gray reference or distance shading.
    hsv = np.column_stack((
        hue_coordinate,
        np.full(len(coordinates), 0.75),
        np.full(len(coordinates), 0.80),
    ))
    rgb = hsv_to_rgb(hsv)
    chordal_distance = np.linalg.norm(
        projector_features-reference[None], axis=1,
    ) / np.sqrt(2.0)
    return rgb, chordal_distance


def _three_nice_ticks(limits):
    """Choose three in-range ticks with a round 1/2/5-times-power-of-ten step."""
    lower, upper = sorted(limits)
    half_span = (upper - lower) / 2
    exponent = int(np.floor(np.log10(half_span)))
    for power in range(exponent, exponent - 3, -1):
        for multiplier in (5, 2, 1):
            step = multiplier * 10.0 ** power
            first = int(np.ceil(lower / step - 1e-10))
            last = int(np.floor(upper / step + 1e-10))
            if last - first >= 2:
                start = int(np.clip(round((lower + upper) / (2 * step)) - 1, first, last - 2))
                return (start + np.arange(3)) * step
    raise ValueError(f"Cannot choose three ticks for axis limits {limits}")


def _style_05c_axes(axis, point_scale=1.0):
    """Readable 05c labels and heavier axes, shared by PNG and MP4 output."""
    axis.tick_params(axis="both", which="major", labelsize=14 * point_scale,
                     width=1.6 * point_scale, length=6 * point_scale, pad=3 * point_scale)
    coordinates = [axis.xaxis, axis.yaxis]
    if axis.name == "3d":
        coordinates.append(axis.zaxis)
        axis.tick_params(axis="z", which="major", labelsize=14 * point_scale,
                         width=1.6 * point_scale, length=6 * point_scale, pad=3 * point_scale)
    for coordinate in coordinates:
        coordinate.label.set_fontsize(18 * point_scale)
        coordinate.labelpad = (14 if axis.name == "3d" else 7) * point_scale
        if axis.name == "3d":
            coordinate.line.set_linewidth(1.6 * point_scale)
            # Axes3D draws tick widths from this mapping rather than tick_params.
            coordinate._axinfo["tick"]["linewidth"] = {
                True: 1.6 * point_scale, False: 1.6 * point_scale,
            }
    for spine in axis.spines.values():
        spine.set_linewidth(1.6 * point_scale)


def _layout_05c_figure(figure, point_scale=1.0):
    """Increase label padding if rendered tick labels overlap an axis label."""
    for _ in range(10):
        figure.tight_layout(pad=0)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        changed = False
        for axis in figure.axes:
            if axis.get_label() == "<colorbar>" or not axis.get_visible():
                continue
            coordinates = [axis.xaxis, axis.yaxis]
            if axis.name == "3d":
                coordinates.append(axis.zaxis)
            for coordinate in coordinates:
                label_box = coordinate.label.get_window_extent(renderer).expanded(1.05, 1.05)
                # get_ticklabels() resets 3D label positions to unprojected
                # coordinates; inspect the labels already positioned by draw.
                tick_labels = [tick.label1 for tick in coordinate.majorTicks] + [coordinate.get_offset_text()]
                if any(label.get_visible() and label.get_text()
                       and label_box.overlaps(label.get_window_extent(renderer))
                       for label in tick_labels):
                    coordinate.labelpad += 2 * point_scale
                    changed = True
        if not changed:
            break
    if len(figure.axes) == 1 and figure.axes[0].name == "3d":
        axis = figure.axes[0]
        # Tight layout reserves the entire rectangular Axes3D panel, including
        # empty space above the projected cube. Fit the visible cube and text
        # instead, without cropping or changing the output pixel dimensions.
        for _ in range(8):
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            corners = np.array(np.meshgrid(axis.get_xlim(), axis.get_ylim(),
                                           axis.get_zlim())).reshape(3, -1)
            x, y, _ = proj3d.proj_transform(*corners, axis.get_proj())
            projected = axis.transData.transform(np.column_stack((x, y)))
            boxes = [Bbox.from_extents(*projected.min(axis=0), *projected.max(axis=0))]
            for coordinate in (axis.xaxis, axis.yaxis, axis.zaxis):
                for label in [coordinate.label, coordinate.get_offset_text(),
                              *(tick.label1 for tick in coordinate.majorTicks)]:
                    if label.get_visible() and label.get_text():
                        boxes.append(label.get_window_extent(renderer))
            bounds = Bbox.union(boxes)
            width, height = figure.bbox.size
            margin = 3 * point_scale
            scale = min((width - 2 * margin) / bounds.width,
                        (height - 2 * margin) / bounds.height)
            position = axis.get_position()
            center = np.array([(position.x0 + position.x1) * width / 2,
                               (position.y0 + position.y1) * height / 2])
            content_center = np.array([(bounds.x0 + bounds.x1) / 2,
                                       (bounds.y0 + bounds.y1) / 2])
            target_center = np.array([width, height]) / 2
            new_center = target_center + scale * (center - content_center)
            new_size = np.array([position.width * width, position.height * height]) * scale
            axis.set_position([*( (new_center - new_size / 2) / [width, height]),
                               *(new_size / [width, height])])
            if abs(scale - 1) < 0.001 and np.max(abs(content_center - target_center)) < 0.5:
                break
        figure.canvas.draw()


def savefig(fig, path: Path) -> str:
    axes_3d = [axis for axis in fig.axes if axis.name == "3d"]
    for axis in axes_3d:
        for location in ("left", "center", "right"):
            axis.set_title("", loc=location)
    if axes_3d and fig._suptitle is not None:
        fig._suptitle.set_text("")
    if Path(path).name.startswith("05c_"):
        for axis in fig.axes:
            if axis.get_label() == "<colorbar>" or not axis.get_visible():
                continue
            axis.set(xlabel="X", ylabel="Y")
            axis.xaxis.set_major_locator(FixedLocator(_three_nice_ticks(axis.get_xlim())))
            axis.yaxis.set_major_locator(FixedLocator(_three_nice_ticks(axis.get_ylim())))
            if axis.name == "3d":
                axis.set_zlabel("Z")
                axis.zaxis.set_major_locator(FixedLocator(_three_nice_ticks(axis.get_zlim())))
            _style_05c_axes(axis)
        dpi = 170
        width, height = GRASSMANN_05C_IMAGE_PIXELS
        fig.set_size_inches(width / dpi, height / dpi, forward=True)
        fig.set_dpi(dpi)
        # Include manually positioned axes (e.g. Goodwin) in tight layout.
        if len(fig.axes) == 1 and fig.axes[0].get_subplotspec() is None:
            fig.axes[0].set_subplotspec(fig.add_gridspec(1, 1)[0])
        _layout_05c_figure(fig)
        # A tight bounding-box crop changes pixel dimensions between models.
        fig.savefig(path, dpi=dpi, bbox_inches=None, pad_inches=0)
    else:
        fig.tight_layout()
        fig.savefig(path, dpi=170)
    plt.close(fig)
    return str(path)


def plot_training_loss_history(checkpoint: Path, outdir: Path) -> Optional[str]:
    """Plot train and held-out test losses saved by train.py."""
    history_path = checkpoint.parent / "history.json"
    if not history_path.is_file():
        print(f"Training history not found: {history_path}")
        return None
    with open(history_path, "r", encoding="utf-8") as handle:
        history = json.load(handle)
    if not isinstance(history, list) or not history:
        print(f"Training history is empty: {history_path}")
        return None

    progress_key = "optimizer_step" if any("optimizer_step" in row for row in history) else "epoch"
    progress_label = "Optimizer step" if progress_key == "optimizer_step" else "Epoch"
    train_positions = np.asarray([row.get(progress_key, index + 1) for index, row in enumerate(history)])
    train_losses = np.asarray([row.get("train_loss", np.nan) for row in history], dtype=float)
    test_positions = np.asarray([
        row.get(progress_key, index + 1) for index, row in enumerate(history)
        if np.isfinite(float(row.get("test_loss", np.nan)))
    ])
    test_losses = np.asarray([
        float(row["test_loss"]) for row in history
        if np.isfinite(float(row.get("test_loss", np.nan)))
    ])

    figure, axis = plt.subplots(figsize=(9.0, 5.0))
    finite_train = np.isfinite(train_losses)
    axis.plot(
        train_positions[finite_train], train_losses[finite_train],
        color="#2563EB", linewidth=1.8, label="train evaluation loss",
    )
    if len(test_losses):
        axis.plot(
            test_positions, test_losses, color="#EF4444", linewidth=1.8,
            marker="o", markersize=3.5, label="held-out evaluation loss",
        )

    # Mark the exact training position represented by the selected checkpoint so the
    # displayed curves are not mistaken for the visualized model state.
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    selected_position = checkpoint_data.get(progress_key)
    if selected_position is not None:
        selected_position = int(selected_position)
        metric_name = str(
            checkpoint_data.get("selection_metric", "validation metric")
        ).replace("_", " ")
        metric_value = checkpoint_data.get("selection_metric_value")
        marker_label = f"{checkpoint.name} · {progress_label.lower()} {selected_position}"
        if metric_value is not None and np.isfinite(float(metric_value)):
            marker_label += f" · {metric_name}={float(metric_value):.4g}"
        axis.axvline(
            selected_position, color="#16A34A", linestyle="--", linewidth=1.6,
            alpha=0.9, label=marker_label,
        )
        selected_rows = [
            row for row in history
            if int(row.get(progress_key, -1)) == selected_position
            and np.isfinite(float(row.get("test_loss", np.nan)))
        ]
        if selected_rows:
            axis.scatter(
                [selected_position], [float(selected_rows[-1]["test_loss"])],
                s=70, marker="*", color="#16A34A", edgecolor="white",
                linewidth=0.7, zorder=5,
            )
    positive = np.concatenate((
        train_losses[finite_train & (train_losses > 0)],
        test_losses[test_losses > 0],
    ))
    if len(positive):
        axis.set_yscale("log")
    axis.set(
        xlabel=progress_label, ylabel="Loss",
        title="Training and held-out evaluation loss",
    )
    axis.grid(True, which="both", alpha=0.22)
    axis.legend(frameon=False)
    return savefig(figure, outdir / "00_train_test_loss.png")


def _encoding_progress(frame, total, name):
    width = 30
    filled = width * frame // total
    print(f"\r{name} [{'#' * filled}{'-' * (width - filled)}] "
          f"{frame}/{total} ({100 * frame / total:.0f}%)",
          end="\n" if frame == total else "", flush=True)


def save_colored_rollout_mp4(
    trajectories, colors, path, *, limits, ticks, image_size=(1400, 1100),
    linewidth=2.0, alpha=0.95, max_frames=240, fps=20,
):
    """Reveal all trajectories together, retaining their static-plot colors.

    Inputs have shape [trajectory, time, 3]. Frame limits affect playback
    cadence only: no trajectory vertices are discarded. Limits and tick
    locations come from the matching static plot, so its framing is preserved.
    """
    trajectories = np.asarray(trajectories)
    colors = np.asarray(colors)
    if (trajectories.ndim != 3 or trajectories.shape[-1] != 3
            or not len(trajectories) or trajectories.shape[1] < 2):
        raise ValueError("trajectories must have shape [trajectory, time >= 2, 3]")
    if colors.shape != trajectories.shape:
        raise ValueError("colors must match trajectories with an RGB value per point")
    if max_frames < 1 or fps < 1:
        raise ValueError("max_frames and fps must be positive")
    if not animation.writers.is_available("ffmpeg"):
        raise RuntimeError("FFmpeg is required to generate --animate MP4 files")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 1400 x 1100 is exactly 7 x 5.5 inches at 200 DPI, avoiding FFmpeg's
    # even-pixel roundoff at the static plot's 170 DPI.
    dpi = 200
    point_scale = 170 / dpi
    figure = plt.figure(figsize=(image_size[0] / dpi, image_size[1] / dpi), dpi=dpi)
    axis = figure.add_subplot(111, projection="3d")
    axis.set(xlabel="X", ylabel="Y", zlabel="Z",
             xlim=limits[0], ylim=limits[1], zlim=limits[2])
    for coordinate_axis, locations in zip((axis.xaxis, axis.yaxis, axis.zaxis), ticks):
        coordinate_axis.set_major_locator(FixedLocator(locations))
        coordinate_axis.pane.fill = False
        coordinate_axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
    axis.view_init(elev=22, azim=-55)
    axis.grid(False)
    axis.set_box_aspect((1.0, 1.0, 0.82))
    _style_05c_axes(axis, point_scale)

    segments = np.stack((trajectories[:, :-1], trajectories[:, 1:]), axis=2)
    segment_colors = 0.5 * (colors[:, :-1] + colors[:, 1:])
    valid = np.isfinite(segments).all(axis=(2, 3)) & np.isfinite(segment_colors).all(axis=2)
    artists = []
    for _ in trajectories:
        artist = Line3DCollection(np.empty((0, 2, 3)), linewidths=linewidth * point_scale, alpha=alpha)
        axis.add_collection3d(artist, autolim=False)
        artists.append(artist)
    _layout_05c_figure(figure, point_scale)

    segment_count = trajectories.shape[1] - 1
    frame_count = min(max_frames, segment_count)

    def initialize():
        for artist in artists:
            artist.set_segments(np.empty((0, 2, 3)))
            artist.set_colors(np.empty((0, 3)))
        return artists

    def update(frame):
        stop = ((frame + 1) * segment_count + frame_count - 1) // frame_count
        for index, artist in enumerate(artists):
            mask = valid[index, :stop]
            artist.set_segments(segments[index, :stop][mask])
            artist.set_colors(segment_colors[index, :stop][mask])
        return artists

    try:
        movie = animation.FuncAnimation(
            figure, update, init_func=initialize, frames=frame_count,
            interval=1000 / fps, blit=False, cache_frame_data=False,
        )
        writer = animation.FFMpegWriter(
            fps=fps, codec="h264", extra_args=["-pix_fmt", "yuv420p"],
        )
        _encoding_progress(0, frame_count, path.name)
        movie.save(path, writer=writer, dpi=dpi,
                   progress_callback=lambda frame, _: _encoding_progress(frame + 1, frame_count, path.name))
    finally:
        plt.close(figure)
    return str(path)


def _animation_progress(frame: int, total: int) -> None:
    """Print a compact terminal progress bar while encoding an animation."""
    width = 30
    completed = min(width, int(width * frame / max(total, 1)))
    bar = "#" * completed + "-" * (width - completed)
    end = "\n" if frame >= total else ""
    print(f"\rEncoding animation [{bar}] {frame}/{total}", end=end, flush=True)


def load_checkpoint(checkpoint: Path, device: torch.device):
    """Load only checkpoints produced by the current canonical model classes."""
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model_class = ckpt["model_class"]
    if model_class not in CURRENT_MODEL_CLASSES:
        raise ValueError(
            f"Unsupported current model class {model_class!r}; expected one of "
            f"{sorted(CURRENT_MODEL_CLASSES)}"
        )
    cfg = dict(ckpt.get("model_config", {}))
    selected = ckpt.get("selected_reencode_every")
    if selected is not None:
        cfg.update(rollout_mode="reencode", reencode_every=int(selected))
        if cfg.get("grassmann"):
            cfg["subspace_update_every"] = -1
    elif isinstance(cfg.get("reencode_every"), (list, tuple)):
        raise ValueError("Checkpoint has candidate intervals but no selected reencoding frequency")
    state = strip_state_prefixes(ckpt["model"])
    if (model_class == "KoopmanAutoencoder" and cfg.get("grassmann")
            and "grassmann_parameterization" not in cfg):
        raise RuntimeError(
            "This baseline Grassmann checkpoint predates the Householder "
            "parameterization. Add model_config.grassmann_parameterization='qr' "
            "to load it explicitly as a legacy QR checkpoint, or retrain with "
            "grassmann_parameterization='householder'."
        )
    model = CURRENT_MODEL_CLASSES[model_class](**cfg).to(device)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        if (model_class == "KoopmanAutoencoder" and cfg.get("grassmann")
                and not any(key.startswith("subspace_net.") for key in state)):
            raise RuntimeError(
                "This is an old static-Grassmann baseline checkpoint and cannot "
                "be loaded as the current state-conditioned Grassmann KAE. "
                "Retrain it or explicitly migrate its architecture."
            ) from exc
        raise
    model.eval()
    return model, ckpt, cfg


def load_arrays(
    ckpt: Dict[str, Any], *, reaching_seed_override: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    if make_data is None:
        if (
            isinstance(_MAKE_DATA_IMPORT_ERROR, ModuleNotFoundError)
            and _MAKE_DATA_IMPORT_ERROR.name == "h5py"
        ):
            raise RuntimeError(
                "The selected dataset is stored as HDF5, but h5py is not "
                f"installed in the active Python interpreter ({sys.executable}). "
                "Activate the training environment (`conda activate torch`) "
                "or install h5py in this interpreter."
            ) from _MAKE_DATA_IMPORT_ERROR
        raise RuntimeError(
            "src.utils.make_data could not be imported; it is required to "
            "regenerate the train/test data"
        ) from _MAKE_DATA_IMPORT_ERROR
    data_cfg = dict(ckpt.get("data_config", {}))
    if not data_cfg:
        raise KeyError("Checkpoint has no data_config")
    system = str(data_cfg.get("system", "")).lower()
    is_reaching = system in {
        "monkey_reaching", "reaching", "comp_neurosci_reaching", "neural_reaching"
    }
    is_dysts_3d = system in DYSTS_3D_SYSTEMS
    if system in GOODWIN_NAMES:
        # Old checkpoints record the removed raw-file setting. Goodwin now
        # always uses the parameter-matched cache; never reopen that old path.
        data_cfg.pop("goodwin_data_path", None)
    if is_reaching and reaching_seed_override is not None:
        data_cfg["seed"] = int(reaching_seed_override)
    if system in REACHING_NAMES:
        configured_path = Path(data_cfg.get("reaching_data_path", PROJECT_ROOT / "data" / "monkey_reaching" / "reaching_data.h5"))
        if not configured_path.is_file() and configured_path.parent.name == "comp_neurosci_reaching":
            data_cfg["reaching_data_path"] = str(PROJECT_ROOT / "data" / "monkey_reaching" / configured_path.name)
    if system in PAM_NAMES:
        data_cfg = pamy_checkpoint_config(data_cfg)
        # Older checkpoints can embed the previous dataset directory or filename.
        configured_path = Path(data_cfg.get("pam_data_path", PROJECT_ROOT / "data" / "PAMy2" / "pamy_data.h5"))
        if (not configured_path.is_file()
                and (configured_path.parent.name in {"pneumatic_robot_arm_PAM", "pneumatic_robot_arm"}
                     or (configured_path.parent.name == "PAMy2" and configured_path.name == "pam_data.h5"))):
            filename = "pamy_data.h5" if configured_path.name == "pam_data.h5" else configured_path.name
            data_cfg["pam_data_path"] = str(PROJECT_ROOT / "data" / "PAMy2" / filename)
    arrays = make_data(**data_cfg)
    if (
        system in {"goodwin", "goodwin_oscillator", "goodwin-oscillator"}
        or is_dysts_3d
        or (is_reaching and reaching_seed_override is not None)
    ):
        checkpoint_mean = np.asarray(ckpt.get("mean", arrays["mean"]))
        checkpoint_std = np.asarray(ckpt.get("std", arrays["std"]))
        source_mean = np.asarray(arrays["mean"])
        source_std = np.asarray(arrays["std"])
        for split in ("train", "val", "test"):
            key = f"{split}_sequences"
            if key not in arrays:
                continue
            raw = arrays[key] * source_std + source_mean
            arrays[key] = ((raw - checkpoint_mean) / checkpoint_std).astype(
                np.float32
            )
        arrays["mean"] = checkpoint_mean
        arrays["std"] = checkpoint_std
    return arrays


# -----------------------------------------------------------------------------
# Strict latent-autonomous rollouts
# -----------------------------------------------------------------------------


@torch.no_grad()
def model_rollout(
    model, x0: torch.Tensor, steps: int, mode: Optional[str],
):
    selected = mode or getattr(model, "rollout_mode", "latent")
    return model.rollout(x0, steps, include_initial=True, rollout_mode=selected)


# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------


@torch.no_grad()
def encoded_truth(model, seq_norm: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(seq_norm).float().to(device)
    enc = model.encode(x)
    if isinstance(enc, dict):
        return enc.get("z_ambient", enc.get("z"))
    return enc


def plot_state_space(
    true_x, pred_x, out: Path, max_points: int, data_cfg: Dict[str, Any]
) -> None:
    """Plot a 2-D or 3-D rollout in the original state coordinates."""
    n = min(len(true_x), len(pred_x), int(max_points))
    state_dim = true_x.shape[1]
    if state_dim == 2:
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.plot(true_x[:n, 0], true_x[:n, 1], label="true", lw=2.2, alpha=0.30)
        ax.plot(pred_x[:n, 0], pred_x[:n, 1], "--", label="latent rollout", lw=1.8)
        ax.scatter(pred_x[0, 0], pred_x[0, 1], s=35, label="initial")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    elif state_dim >= 3:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(true_x[:n, 0], true_x[:n, 1], true_x[:n, 2],
                label="true", lw=1.3, alpha=0.45)
        ax.plot(pred_x[:n, 0], pred_x[:n, 1], pred_x[:n, 2],
                "--", label="latent rollout", lw=1.3)
        ax.scatter(*pred_x[0, :3], s=30, label="initial")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    else:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(true_x[:n, 0], label="true", lw=2.2, alpha=0.30)
        ax.plot(pred_x[:n, 0], "--", label="latent rollout", lw=1.8)
        ax.set_xlabel("step")
        ax.set_ylabel("x")

    ax.set_title("Autonomous latent rollout; decoded only for visualization")
    ax.grid(alpha=0.25)
    savefig(fig, out)


def _reaching_test_obstructed(
    arrays: Dict[str, np.ndarray], test_count: int,
) -> np.ndarray:
    """Return one obstructed/unobstructed label per held-out trajectory."""
    offsets = np.asarray(arrays["reaching_split_offsets"], dtype=np.int64)
    if offsets.size != 3:
        raise ValueError("Reaching split metadata must contain train/test offsets")
    if "reaching_split_obstructed" in arrays:
        labels = np.asarray(arrays["reaching_split_obstructed"], dtype=bool)
        test_labels = labels[offsets[1]:offsets[2]]
    elif "reaching_split_path_classes" in arrays:
        labels = np.asarray(arrays["reaching_split_path_classes"]).astype(str)
        test_labels = labels[offsets[1]:offsets[2]] == "obstructed"
    else:
        path_class = str(np.asarray(arrays["reaching_path_class"]).item())
        if path_class == "both":
            raise ValueError(
                "Mixed reaching data requires per-trajectory path-class metadata"
            )
        test_labels = np.full(test_count, path_class == "obstructed", dtype=bool)
    if len(test_labels) != test_count:
        raise ValueError(
            "Reaching test path-class metadata does not match test trajectories"
        )
    return test_labels


def plot_timeseries(true_x, pred_x, dt: float, out: Path) -> None:
    T = min(len(true_x), len(pred_x))
    t = np.arange(T) * float(dt)
    d = min(true_x.shape[1], 6)
    fig, axes = plt.subplots(d, 1, figsize=(11, 2.3 * d), sharex=True, squeeze=False)
    for j in range(d):
        ax = axes[j, 0]
        ax.plot(t, true_x[:T, j], label="true")
        ax.plot(t, pred_x[:T, j], "--", label="latent rollout")
        ax.set_ylabel(f"x[{j}]")
        ax.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("time")
    savefig(fig, out)


def plot_errors(true_x, pred_x, z_true, z_pred, dt: float, out: Path) -> None:
    T = min(len(true_x), len(pred_x), len(z_true), len(z_pred))
    state_rmse = np.sqrt(np.mean((pred_x[:T] - true_x[:T]) ** 2, axis=1))
    latent_rmse = np.sqrt(np.mean((z_pred[:T] - z_true[:T]) ** 2, axis=1))
    t = np.arange(T) * float(dt)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(t, state_rmse, label="state RMSE")
    ax.plot(t, latent_rmse, label="latent linearity RMSE")
    ax.set_xlabel("time")
    ax.set_ylabel("RMSE")
    ax.set_title("Error versus autonomous latent rollout horizon")
    ax.grid(alpha=0.25)
    savefig(fig, out)


def plot_latent_norm(z_true, z_pred, dt: float, out: Path) -> None:
    T = min(len(z_true), len(z_pred))
    t = np.arange(T) * float(dt)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(t, np.linalg.norm(z_true[:T], axis=1), label="||E(x_t)||")
    ax.plot(t, np.linalg.norm(z_pred[:T], axis=1), label="||z_t rollout||")
    ax.set_xlabel("time")
    ax.set_ylabel("latent norm")
    ax.set_title("Latent-state norm")
    ax.grid(alpha=0.25)
    savefig(fig, out)


def plot_spectrum(model, out: Path) -> Dict[str, Any]:
    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.45, label="unit circle")
    summary: Dict[str, Any] = {}

    K = model.K.detach().cpu().float().numpy()
    vals = np.linalg.eigvals(K)
    rho = float(np.max(np.abs(vals)))
    summary["spectral_radius"] = rho
    ax.scatter(vals.real, vals.imag, s=35, label=f"K, rho={rho:.4f}")
    ax.set_title("Global Koopman spectrum")

    ax.axhline(0, lw=0.7)
    ax.axvline(0, lw=0.7)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Re(lambda)")
    ax.set_ylabel("Im(lambda)")
    ax.grid(alpha=0.2)
    savefig(fig, out)
    return summary


def _plot_reaching_state_conditioned_grassmann(
    model, arrays, mean, std, device, outdir, *, samples, plane_scale,
    _path_class=None,
) -> Dict[str, Any]:
    """Average held-out states per direction before predicting subspaces."""
    sequences = np.asarray(arrays["test_sequences"], dtype=np.float32)
    test_obstructed = _reaching_test_obstructed(arrays, len(sequences))
    if _path_class is None and len(np.unique(test_obstructed)) > 1:
        reports = {}
        offsets = np.asarray(arrays["reaching_split_offsets"], dtype=np.int64)
        directions = np.asarray(
            arrays["reaching_split_directions"], dtype=np.int64
        )[offsets[1]:offsets[2]]
        for path_name, is_obstructed in (
            ("unobstructed", False), ("obstructed", True),
        ):
            indices = np.flatnonzero(test_obstructed == is_obstructed)
            subset = dict(arrays)
            subset["test_sequences"] = sequences[indices]
            subset["reaching_path_class"] = np.asarray(path_name)
            subset["reaching_split_offsets"] = np.asarray(
                [0, 0, len(indices)], dtype=np.int64
            )
            subset["reaching_split_directions"] = directions[indices]
            subset["reaching_split_obstructed"] = np.full(
                len(indices), is_obstructed, dtype=bool
            )
            reports[path_name] = _plot_reaching_state_conditioned_grassmann(
                model, subset, mean, std, device, outdir,
                samples=samples, plane_scale=plane_scale,
                _path_class=path_name,
            )
        return {"by_path_class": reports}
    selected_directions = np.asarray(
        arrays["reaching_selected_directions"], dtype=np.int64
    )
    offsets = np.asarray(arrays["reaching_split_offsets"], dtype=np.int64)
    all_directions = np.asarray(arrays["reaching_split_directions"], dtype=np.int64)
    test_directions = all_directions[offsets[1]:offsets[2]]
    represented_directions = [int(direction) for direction in selected_directions
                              if np.any(test_directions == direction)]
    trial_indices = [np.flatnonzero(test_directions == direction)
                     for direction in represented_directions]
    # Average the full state (neural, cursor position, and velocity) first;
    # evaluating U(mean(x)) differs from averaging projectors or RGB values.
    sequences = np.stack([sequences[indices].mean(axis=0) for indices in trial_indices])
    test_directions = np.asarray(represented_directions, dtype=np.int64)
    trajectory_indices = np.arange(len(sequences), dtype=np.int64)
    neural_count = int(np.asarray(arrays["reaching_neural_unit_count"]).item())
    sample_indices = [
        np.unique(np.linspace(0, sequences.shape[1]-1,
                              min(samples, sequences.shape[1]), dtype=int))
        for _ in trajectory_indices
    ]
    sample_states = np.concatenate([
        sequences[trajectory, indices]
        for trajectory, indices in zip(trajectory_indices, sample_indices)
    ])
    normalized = torch.from_numpy(sample_states).to(device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        psi = model.ambient_dictionary(normalized)
        basis = model.predict_subspace(normalized)
        projected = model.project_with_basis(psi, basis)
    scale_tensor = torch.as_tensor(std, device=device, dtype=projected.dtype)
    offset_tensor = torch.as_tensor(mean, device=device, dtype=projected.dtype)
    def display_from_y(y):
        state = model.decoder(y)*scale_tensor+offset_tensor
        velocity = state[neural_count+2:neural_count+4]
        return torch.stack((state[neural_count], state[neural_count+1],
                            torch.linalg.vector_norm(velocity)))
    tangent_maps = []
    for y, frame in zip(projected, basis):
        with torch.enable_grad():
            jacobian = torch.autograd.functional.jacobian(
                display_from_y, y.detach().requires_grad_(True)
            )
        tangent_maps.append((jacobian @ frame).detach().cpu().numpy())

    physical = denorm(sequences, mean, std)
    display_paths = []
    centers_per_path = []
    for trajectory, indices in zip(trajectory_indices, sample_indices):
        cursor = physical[trajectory, :, neural_count:neural_count+4]
        path = np.column_stack((cursor[:,0], cursor[:,1],
                                np.linalg.norm(cursor[:,2:4], axis=1)))
        display_paths.append(path)
        centers_per_path.append(path[indices])
    grid = np.asarray(((-1,-1),(-1,1),(1,1),(1,-1)))
    shades = plt.cm.Greys(np.linspace(0.85, 0.25, len(sample_indices[0])))
    panel_count = len(trajectory_indices)
    column_count = int(np.ceil(np.sqrt(panel_count)))
    row_count = int(np.ceil(panel_count / column_count))
    figure = plt.figure(figsize=(5.2*column_count, 4.7*row_count))
    axes = [
        figure.add_subplot(row_count, column_count, index+1, projection="3d")
        for index in range(panel_count)
    ]
    tangent_offset = 0
    skipped = []
    for axis, trajectory, path, centers, indices in zip(
        axes,
        trajectory_indices, display_paths, centers_per_path, sample_indices
    ):
        direction = int(test_directions[trajectory])
        span = max(float(np.ptp(path, axis=0).max()), 1.0)
        half_width = float(plane_scale)*span
        axis.plot(path[:,0], path[:,1], path[:,2], color="0.58", lw=1.0)
        for local_index, (center, color) in enumerate(zip(centers, shades)):
            tangent = tangent_maps[tangent_offset + local_index]
            left, singular, _ = np.linalg.svd(tangent, full_matrices=False)
            axis.scatter(*center, color=color, s=28)
            if len(singular) < 2 or singular[1] <= 1e-8*max(singular[0], 1.0):
                skipped.append([int(trajectory), int(indices[local_index])])
                continue
            corners = center + half_width*(
                grid[:,0,None]*left[:,0] + grid[:,1,None]*left[:,1]
            )
            edge = np.asarray(color)*np.asarray((.65,.65,.65,1))
            axis.add_collection3d(Poly3DCollection(
                [corners], facecolor=color, edgecolor=edge, alpha=.28
            ))
        tangent_offset += len(indices)
        axis.set(
            xlabel="cursor X", ylabel="cursor Y", zlabel="cursor speed",
            title=f"Reaching direction {direction + 1}",
        )
    figure.suptitle(
        "Direction-averaged held-out reaching states with tangent planes"
        + (f" · {_path_class}" if _path_class else ""),
        y=0.995,
    )
    suffix = f"_{_path_class}" if _path_class else ""
    path = outdir / f"05_state_conditioned_grassmann_physical{suffix}.png"
    savefig(figure, path)
    rgb_path = _plot_reaching_state_conditioned_grassmann_rgb(
        model, sequences, trajectory_indices, mean, std, device, outdir,
        neural_count, path_class=_path_class,
    )
    if was_training:
        model.train()
    return {
        "plot": str(path),
        "rgb_subspace_plot": rgb_path,
        "subplot_count": panel_count,
        "trajectory_indices": trajectory_indices.tolist(),
        "trajectory_source": "direction_averaged_ground_truth",
        "averaged_trial_indices": [indices.tolist() for indices in trial_indices],
        "trial_counts": [len(indices) for indices in trial_indices],
        "directions": test_directions[trajectory_indices].tolist(),
        "sample_indices": [indices.tolist() for indices in sample_indices],
        "skipped_rank_deficient": skipped,
    }


@torch.no_grad()
def _grassmann_plot_rollouts(model, sequences, device):
    """Roll out held-out initial states before any display-only time subsampling."""
    sequences = np.asarray(sequences, dtype=np.float32)
    mode = str(getattr(model, "rollout_mode", "latent")).lower()
    parts = []
    for start in range(0, len(sequences), 16):
        initial = torch.from_numpy(sequences[start:start + 16, 0]).to(device)
        parts.append(model_rollout(model, initial, sequences.shape[1] - 1, mode)
                     ["x_rollout"].detach().cpu().float().numpy())
    return np.concatenate(parts)


def _plot_reaching_state_conditioned_grassmann_rgb(
    model, sequences, trajectory_indices, mean, std, device, outdir,
    neural_count, *, path_class=None,
) -> str:
    """Show ground-truth subspace colors with white direction-mean paths."""
    selected = np.asarray(sequences[trajectory_indices], dtype=np.float32)
    with torch.no_grad():
        state_samples = selected
        flat_states = torch.from_numpy(
            state_samples.reshape(-1, state_samples.shape[-1])
        ).to(device)
        projector_parts = []
        for start in range(0, len(flat_states), 2048):
            bases = model.predict_subspace(flat_states[start:start+2048])
            projectors = bases @ bases.transpose(-2, -1)
            projector_parts.append(
                projectors.flatten(start_dim=-2).cpu().numpy()
            )
    projector_features = np.concatenate(projector_parts)
    color_embedding = _fit_projector_color_embedding(
        projector_features, model.grassmann_rank,
    )
    rgb, _ = _apply_projector_color_embedding(
        projector_features, color_embedding,
    )

    physical_samples = denorm(state_samples, mean, std)
    cursor_samples = physical_samples[..., neural_count:neural_count+2].reshape(-1, 2)
    rounded = np.round(cursor_samples, decimals=8)
    _, unique_indices = np.unique(rounded, axis=0, return_index=True)
    cursor_samples = cursor_samples[unique_indices]
    rgb = rgb[unique_indices]
    padding = 0.06*np.maximum(np.ptp(cursor_samples, axis=0), 1.0e-6)
    lower = cursor_samples.min(axis=0)-padding
    upper = cursor_samples.max(axis=0)+padding
    grid_x, grid_y = np.meshgrid(
        np.linspace(lower[0], upper[0], 160),
        np.linspace(lower[1], upper[1], 160),
    )
    triangulation = mtri.Triangulation(cursor_samples[:, 0], cursor_samples[:, 1])
    rgb_grid = np.stack([
        np.asarray(mtri.LinearTriInterpolator(
            triangulation, rgb[:, channel]
        )(grid_x, grid_y).filled(np.nan))
        for channel in range(3)
    ], axis=-1)
    alpha = np.isfinite(rgb_grid).all(axis=-1).astype(float)
    rgba_grid = np.dstack((np.nan_to_num(rgb_grid, nan=1.0), alpha))

    truth_physical = denorm(selected, mean, std)
    figure, axis = plt.subplots(figsize=(8.2, 6.6))
    axis.imshow(
        rgba_grid, origin="lower",
        extent=(lower[0], upper[0], lower[1], upper[1]),
        interpolation="bilinear", aspect="equal",
    )
    for index, truth in enumerate(truth_physical):
        axis.plot(
            truth[:, neural_count], truth[:, neural_count+1],
            color="white", linewidth=1.5,
            label="direction-averaged ground truth" if index == 0 else None,
            zorder=3,
        )
    axis.set(
        xlabel="cursor X", ylabel="cursor Y",
        title=("State-conditioned Grassmann subspaces: absolute RGB"
               + (f" · {path_class}" if path_class else "")),
        xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]),
    )
    axis.legend(frameon=False)
    suffix = f"_{path_class}" if path_class else ""
    path = outdir / f"05c_reaching_state_conditioned_grassmann_rgb{suffix}.png"
    savefig(figure, path)
    return str(path)


def _heldout_plot_sequences(arrays):
    """All test and validation trajectories, preserving split order and time points."""
    return np.concatenate([
        np.asarray(arrays[f"{split}_sequences"], dtype=np.float32)
        for split in ("test", "val")
        if f"{split}_sequences" in arrays and len(arrays[f"{split}_sequences"])
    ])


def _system_plot_sequences(arrays, system):
    """Goodwin displays its validation split; keep new test trajectories held out."""
    key = canonical_system(system)
    if key in GOODWIN_NAMES:
        return np.asarray(arrays["val_sequences"], dtype=np.float32)
    if key in MULTISTABLE_3D or key == "lotka_volterra":
        return _heldout_plot_sequences(arrays)
    return np.asarray(arrays["test_sequences"], dtype=np.float32)


def plot_state_conditioned_grassmann(
    model, arrays, mean, std, device, outdir, system, *, samples=12,
    plane_scale=0.08, seed=0, animate=False, _pam_behavior_name=None,
) -> Dict[str, Any]:
    """Physical push-forward illustrations for baseline state-conditioned U(x)."""
    if "pam_window_size" in arrays and not arrays.get("pam_plot_selected", False):
        report = plot_state_conditioned_grassmann(
            model, pamy_visualization_arrays(arrays), mean, std, device, outdir, system,
            samples=samples, plane_scale=plane_scale, seed=seed, animate=animate)
        # Only 05e uses the larger selection from the complete held-out pool.
        report["fast_slow_rgb_comparison"] = _plot_robot_arm_grassmann_behavior_comparison(
            model, arrays, mean, std, device, outdir,
        )
        return report
    key = canonical_system(system)
    if key == "lotka_volterra":
        return _plot_lotka_volterra_grassmann(model, arrays, mean, std, device, outdir)
    if key in {"duffing", "duffing_oscillator", "duffing-oscillator"}:
        return _plot_duffing_state_conditioned_grassmann(
            model, arrays, mean, std, device, outdir,
            samples=samples, plane_scale=plane_scale,
        )
    if key in {"monkey_reaching", "reaching", "comp_neurosci_reaching", "neural_reaching"}:
        return _plot_reaching_state_conditioned_grassmann(
            model, arrays, mean, std, device, outdir,
            samples=samples, plane_scale=plane_scale,
        )
    if (key in {"pam", "pamy2", "pneumatic_robot_arm_pam"}
            and _pam_behavior_name is None):
        sequences = np.asarray(arrays["test_sequences"])
        labels = _pam_test_behavior_labels(arrays, len(sequences))
        reports = {}
        for behavior_name in dict.fromkeys(labels.tolist()):
            indices = np.flatnonzero(labels == behavior_name)
            subset = dict(arrays)
            subset["test_sequences"] = sequences[indices]
            subset["pam_test_behaviors"] = labels[indices]
            reports[behavior_name] = plot_state_conditioned_grassmann(
                model, subset, mean, std, device, outdir, system,
                samples=samples, plane_scale=plane_scale, seed=seed,
                animate=animate,
                _pam_behavior_name=behavior_name,
            )
        return {
            "by_behavior": reports,
            "fast_slow_rgb_comparison": (
                _plot_robot_arm_grassmann_behavior_comparison(
                    model, arrays, mean, std, device, outdir,
                ) if not arrays.get("pam_plot_selected", False) else {}
            ),
        }
    display_holdouts = _system_plot_sequences(arrays, key) if key in (MULTISTABLE_3D | GOODWIN_NAMES) else None
    sequence = (display_holdouts.reshape(-1, display_holdouts.shape[-1]) if display_holdouts is not None
                else np.asarray(arrays["test_sequences"], dtype=np.float32)[0])
    if arrays.get("pam_plot_selected", False):
        holdouts = np.asarray(arrays["test_sequences"], dtype=np.float32)
        sequence = holdouts.reshape(-1, holdouts.shape[-1])
    sample_count = min(max(int(samples), 1), len(sequence))
    if sample_count == len(sequence):
        indices = np.arange(len(sequence), dtype=int)
    else:
        # Greedy farthest-point sampling covers the trajectory's state-space
        # geometry more evenly than uniformly spaced time indices.
        center = sequence.mean(axis=0, keepdims=True)
        selected = [int(np.linalg.norm(sequence-center, axis=1).argmax())]
        minimum_squared_distance = np.sum(
            (sequence-sequence[selected[0]]) ** 2, axis=1
        )
        minimum_squared_distance[selected[0]] = -np.inf
        while len(selected) < sample_count:
            next_index = int(minimum_squared_distance.argmax())
            selected.append(next_index)
            candidate_distance = np.sum(
                (sequence-sequence[next_index]) ** 2, axis=1
            )
            minimum_squared_distance = np.minimum(
                minimum_squared_distance, candidate_distance
            )
            minimum_squared_distance[next_index] = -np.inf
        indices = np.asarray(sorted(selected), dtype=int)
    normalized = torch.from_numpy(sequence[indices]).to(device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        psi = model.ambient_dictionary(normalized)
        basis = model.predict_subspace(normalized)
        projected = torch.einsum("ndr,nr->nd", basis,
            torch.einsum("ndr,nd->nr", basis, psi))
    shades = plt.cm.Greys(np.linspace(0.85, 0.25, len(indices)))
    physical = denorm(sequence[indices], mean, std)
    report = {"sample_indices": indices.tolist(), "skipped_rank_deficient": []}

    trajectory_display = None
    if key in ({"goodwin", "goodwin_oscillator", "goodwin-oscillator",
                "lorenz", "lorenz63", "lorenz_63", "lorenz-63"}
               | DYSTS_3D_SYSTEMS | MULTISTABLE_3D):
        centers = physical[:, :3]
        trajectory_display = denorm(
            display_holdouts if display_holdouts is not None else sequence, mean, std,
        )[..., :3]
        scale_tensor = torch.as_tensor(std, device=device, dtype=projected.dtype)
        offset_tensor = torch.as_tensor(mean, device=device, dtype=projected.dtype)
        def display_from_y(y): return (model.decoder(y)*scale_tensor+offset_tensor)[:3]
        tangent_maps = []
        for y, frame in zip(projected, basis):
            with torch.enable_grad():
                jacobian = torch.autograd.functional.jacobian(display_from_y, y.detach().requires_grad_(True))
            tangent_maps.append((jacobian @ frame).detach().cpu().numpy())
        axis_labels = (("x", "y", "z") if key not in {
                           "goodwin", "goodwin_oscillator", "goodwin-oscillator"
                       }
                       else ("X1", "X2", "X3"))
        plot_title = "Decoder-pushforward local Grassmann tangent planes"
    elif key in {"pam", "pamy2", "pneumatic_robot_arm_pam"}:
        centers = _pam_kinematic_trajectory(physical[:, :4])[:, -1]
        projectors = basis @ basis.transpose(-2, -1)
        projector_features = projectors.flatten(start_dim=-2).cpu().numpy()
        color_embedding = _fit_projector_color_embedding(
            projector_features, model.grassmann_rank,
        )
        rgb, _ = _apply_projector_color_embedding(
            projector_features, color_embedding,
        )
        shades = np.column_stack((rgb, np.ones(len(rgb))))
        report["plane_color_definition"] = (
            "first three principal components of flattened local projectors"
        )
        tangent_maps = []
        delta = 1.0e-3
        for y, frame in zip(projected, basis):
            columns = []
            for direction in frame.transpose(0,1):
                states = []
                for sign in (-1.0, 1.0):
                    decoded = model.decoder(y + sign*delta*direction).detach().cpu().numpy()
                    states.append(decoded*np.asarray(std)+np.asarray(mean))
                endpoints = _pam_kinematic_trajectory(np.asarray(states)[:, :4])[:, -1]
                columns.append((endpoints[1]-endpoints[0])/(2*delta))
            tangent_maps.append(np.stack(columns, axis=1))
        axis_labels = ("end-effector X", "end-effector Y", "end-effector Z")
        plot_title = "Decoder-pushforward local Grassmann tangent planes"
    else:
        if was_training: model.train()
        return {"omitted": "No physical display mapping is defined for this dataset."}

    figure = plt.figure(figsize=(7.2, 6.0)); axis = figure.add_subplot(111, projection="3d")
    span = max(float(np.ptp(centers, axis=0).max()), 1.0); half_width = float(plane_scale)*span
    grid = np.asarray(((-1,-1),(-1,1),(1,1),(1,-1)))
    for number, (center, tangent, color) in enumerate(zip(centers, tangent_maps, shades)):
        left, singular, _ = np.linalg.svd(tangent, full_matrices=False)
        if len(singular)<2 or singular[1] <= 1e-8*max(singular[0],1.0):
            report["skipped_rank_deficient"].append(int(indices[number])); continue
        corners = center + half_width*(grid[:,0,None]*left[:,0] + grid[:,1,None]*left[:,1])
        polygon = Poly3DCollection([corners], facecolor=color, edgecolor=np.asarray(color)*np.asarray((.65,.65,.65,1)), alpha=.28)
        axis.add_collection3d(polygon); axis.scatter(*center, color=color, s=34)
    path_points = centers if trajectory_display is None else trajectory_display
    if path_points.ndim == 3:
        _draw_3d_trajectories(axis, path_points, linewidth=1.0)
    else:
        axis.plot(path_points[:,0], path_points[:,1], path_points[:,2], color="0.72", lw=1.0)
    axis.set(xlabel=axis_labels[0], ylabel=axis_labels[1], zlabel=axis_labels[2], title=plot_title)
    pam_suffix = ""
    if _pam_behavior_name is not None:
        short_name = str(_pam_behavior_name).replace(
            "fixed_movement_", ""
        ).replace("_", "-")
        pam_suffix = f"_{short_name}"
        plot_title += " · " + str(_pam_behavior_name).replace("_", " ").title()
        axis.set_title(plot_title)
    path = outdir / f"05_state_conditioned_grassmann_physical{pam_suffix}.png"
    savefig(figure, path)
    report["plot"] = str(path)
    if key in ({"lorenz", "lorenz63", "lorenz_63", "lorenz-63",
                "goodwin", "goodwin_oscillator", "goodwin-oscillator"}
               | DYSTS_3D_SYSTEMS | MULTISTABLE_3D):
        is_goodwin = key in {"goodwin", "goodwin_oscillator", "goodwin-oscillator"}
        display_sequences = _system_plot_sequences(arrays, key)
        if not is_goodwin and key not in MULTISTABLE_3D:
            display_sequences = display_sequences[:DYSTS_PLOT_HOLDOUTS]
        display_sequences = _grassmann_plot_rollouts(model, display_sequences, device)
        points_per_trajectory = max(
            1, DYSTS_GRASSMANN_RGB_POINT_BUDGET // len(display_sequences)
        )
        # Subsample only for rendering, after integrating every rollout step.
        # Goodwin retains full resolution.
        display_stride = 1 if is_goodwin else max(
            1, display_sequences.shape[1] // points_per_trajectory
        )
        display_sequences = display_sequences[:, ::display_stride]
        trajectory_length = display_sequences.shape[1]
        flattened_states = display_sequences.reshape(
            -1, display_sequences.shape[-1]
        )
        with torch.no_grad():
            feature_parts = []
            for start in range(0, len(flattened_states), 4096):
                all_basis = model.predict_subspace(torch.from_numpy(
                    flattened_states[start:start + 4096]
                ).to(device))
                feature_parts.append((
                    all_basis @ all_basis.transpose(-2, -1)
                ).flatten(start_dim=-2).cpu().numpy())
        features = np.concatenate(feature_parts)
        color_embedding = _fit_projector_color_embedding(
            features, model.grassmann_rank,
        )
        colors, _ = _apply_projector_color_embedding(features, color_embedding)
        rgb_figure = plt.figure(figsize=(7.0, 6.0) if is_goodwin else (8.0, 6.5))
        rgb_axis = (rgb_figure.add_axes((0.20, 0.08, 0.75, 0.86), projection="3d")
                    if is_goodwin else rgb_figure.add_subplot(111, projection="3d"))
        all_physical = denorm(display_sequences, mean, std)
        visible = [all_physical.reshape(-1, 3)]
        if is_goodwin or key in DYSTS_3D_SYSTEMS or key in {
            "lorenz", "lorenz63", "lorenz_63", "lorenz-63"
        }:
            training_sequences = np.asarray(arrays["train_sequences"])
            if not is_goodwin:
                training_sequences = training_sequences[:len(display_sequences)]
            source = denorm(training_sequences, mean, std)
            # Preserve the existing framing without drawing training paths.
            visible.append(source.reshape(-1, 3))
            report["rgb_context_trajectory_count"] = 0
        trajectory_colors = colors.reshape(len(display_sequences), trajectory_length, 3)
        _draw_3d_trajectories(
            rgb_axis, all_physical, colors=trajectory_colors,
            linewidth=1.5 if is_goodwin else 2.0, alpha=1.0 if is_goodwin else 0.95,
        )
        flattened_physical = np.concatenate(visible)
        rgb_axis.auto_scale_xyz(
            flattened_physical[:, 0], flattened_physical[:, 1],
            flattened_physical[:, 2],
        )
        if is_goodwin:
            rgb_axis.set_xlim(right=11)
        _style_3d_axis(rgb_axis)
        display_name = str(np.asarray(arrays.get(
            "system_display_name", arrays.get("dysts_system", "Goodwin" if is_goodwin else "Lorenz-63")
        )).item())
        filename_name = "goodwin" if is_goodwin else key.replace("-", "_")
        labels = ("X1", "X2", "X3") if is_goodwin else ("x", "y", "z")
        rgb_axis.set(
            xlabel=labels[0], ylabel=labels[1], zlabel=labels[2],
            title=f"{display_name} trajectories colored by local Grassmann subspace",
        )
        rgb_path = outdir / f"05c_{filename_name}_state_conditioned_grassmann_rgb.png"
        savefig(rgb_figure, rgb_path)
        report["rgb_subspace_plot"] = str(rgb_path)
        if animate:
            report["rgb_subspace_animation"] = save_colored_rollout_mp4(
                all_physical, trajectory_colors, rgb_path.with_suffix(".mp4"),
                limits=(rgb_axis.get_xlim(), rgb_axis.get_ylim(), rgb_axis.get_zlim()),
                ticks=(rgb_axis.get_xticks(), rgb_axis.get_yticks(), rgb_axis.get_zticks()),
                image_size=GRASSMANN_05C_IMAGE_PIXELS,
                linewidth=1.5 if is_goodwin else 2.0,
                alpha=1.0 if is_goodwin else 0.95,
                max_frames=ROLLOUT_ANIMATION_MAX_FRAMES,
            )
        report["rgb_color_definition"] = (
            "first projector PC mapped to full-wheel hue with fixed saturation and brightness"
        )
        report["rgb_trajectory_count"] = int(len(display_sequences))
        report["rgb_trajectory_source"] = "model_rollout"
        if key in DYSTS_3D_SYSTEMS | MULTISTABLE_3D:
            report["contourf_plot"] = _plot_dysts_grassmann_contourf(
                model, arrays, mean, std, device, outdir, filename_name,
            )
    if key in {"pam", "pamy2", "pneumatic_robot_arm_pam"}:
        report["rgb_subspace_plot"] = (
            _plot_robot_arm_state_conditioned_grassmann_rgb(
                model, arrays, mean, std, device, outdir,
                filename_suffix=pam_suffix,
                behavior_name=_pam_behavior_name,
            )
        )
    if was_training: model.train()
    return report


def _plot_duffing_state_conditioned_grassmann(
    model, arrays, mean, std, device, outdir, *, samples=12, plane_scale=0.08,
) -> Dict[str, Any]:
    """Show distance-aware RGB subspace fields for the Duffing system."""
    was_training = model.training
    model.eval()
    well = np.sqrt(
        float(np.asarray(arrays["duffing_linear_stiffness"]))
        / float(np.asarray(arrays["duffing_cubic_stiffness"]))
    )

    # Embed gauge-invariant local projectors into a shared RGB space and use
    # those colors along model rollouts from held-out initial states.
    test_sequences = np.asarray(arrays["test_sequences"], dtype=np.float32)
    rgb_sequences = test_sequences[:min(12, len(test_sequences))]
    rgb_sequences = _grassmann_plot_rollouts(model, rgb_sequences, device)
    rgb_length = rgb_sequences.shape[1]
    rgb_states = torch.from_numpy(
        rgb_sequences.reshape(-1, rgb_sequences.shape[-1])
    ).to(device)
    with torch.no_grad():
        rgb_basis = model.predict_subspace(rgb_states)
    projectors = rgb_basis @ rgb_basis.transpose(-2, -1)
    projector_features = projectors.flatten(start_dim=-2).cpu().numpy()
    color_embedding = _fit_projector_color_embedding(
        projector_features, model.grassmann_rank,
    )
    rgb_colors, _ = _apply_projector_color_embedding(
        projector_features, color_embedding,
    )
    rgb_colors = rgb_colors.reshape(len(rgb_sequences), rgb_length, 3)
    rgb_physical = denorm(rgb_sequences, mean, std)
    rgb_figure, rgb_axis = plt.subplots(figsize=(8.0, 6.2))
    for trajectory, point_colors in zip(rgb_physical, rgb_colors):
        segments = np.stack((trajectory[:-1], trajectory[1:]), axis=1)
        segment_colors = 0.5 * (point_colors[:-1] + point_colors[1:])
        rgb_axis.add_collection(LineCollection(
            segments, colors=segment_colors, linewidths=1.6, alpha=0.9,
        ))
    flattened = rgb_physical.reshape(-1, rgb_physical.shape[-1])
    padding = 0.04 * np.maximum(np.ptp(flattened, axis=0), 1.0)
    rgb_axis.set_xlim(flattened[:, 0].min()-padding[0], flattened[:, 0].max()+padding[0])
    rgb_axis.set_ylim(flattened[:, 1].min()-padding[1], flattened[:, 1].max()+padding[1])
    rgb_axis.scatter([-well, well], [0.0, 0.0], marker="*", s=90,
                     color="#111827", zorder=4)
    rgb_axis.set(
        xlabel="position", ylabel="velocity",
        title="Duffing trajectories colored by local Grassmann subspace",
    )
    rgb_axis.grid(alpha=0.2)
    rgb_path = outdir / "05c_duffing_state_conditioned_grassmann_rgb.png"
    savefig(rgb_figure, rgb_path)

    # Evaluate the same projector embedding on a dense phase-space grid to
    # expose how the selected subspace varies away from sampled trajectories.
    lower = flattened.min(axis=0) - padding
    upper = flattened.max(axis=0) + padding
    grid_x, grid_v = np.meshgrid(
        np.linspace(lower[0], upper[0], 180),
        np.linspace(lower[1], upper[1], 140),
    )
    grid_physical = np.column_stack((grid_x.ravel(), grid_v.ravel()))
    grid_normalized = (
        (grid_physical - np.asarray(mean)) / np.asarray(std)
    ).astype(np.float32)
    grid_projector_parts = []
    with torch.no_grad():
        for start in range(0, len(grid_normalized), 4096):
            grid_batch = torch.from_numpy(
                grid_normalized[start:start+4096]
            ).to(device)
            grid_basis = model.predict_subspace(grid_batch)
            grid_projector_parts.append(
                (grid_basis @ grid_basis.transpose(-2, -1))
                .flatten(start_dim=-2).cpu().numpy()
            )
    grid_features = np.concatenate(grid_projector_parts, axis=0)
    grid_rgb, _ = _apply_projector_color_embedding(
        grid_features, color_embedding,
    )
    background_figure, background_axis = plt.subplots(figsize=(8.0, 6.2))
    background_axis.imshow(
        grid_rgb.reshape(*grid_x.shape, 3), origin="lower",
        extent=(lower[0], upper[0], lower[1], upper[1]),
        interpolation="bilinear", aspect="auto",
    )
    for trajectory in rgb_physical:
        background_axis.plot(
            trajectory[:, 0], trajectory[:, 1], color="white",
            linewidth=0.9, alpha=0.72,
        )
    background_axis.scatter(
        [-well, well], [0.0, 0.0], marker="*", s=95,
        facecolor="white", edgecolor="#111827", linewidth=0.8, zorder=4,
    )
    background_axis.set(
        xlabel="position", ylabel="velocity",
        title="Duffing subspaces: absolute RGB",
        xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]),
    )
    background_path = (
        outdir / "05d_duffing_state_conditioned_grassmann_rgb_background.png"
    )
    savefig(background_figure, background_path)
    if was_training:
        model.train()
    return {
        "rgb_subspace_plot": str(rgb_path),
        "rgb_subspace_background_plot": str(background_path),
        "rgb_trajectory_count": int(len(rgb_sequences)),
        "rgb_color_definition": (
            "projector PCA hue with fixed saturation and brightness"
        ),
    }


def _plot_robot_arm_state_conditioned_grassmann_rgb(
    model, arrays, mean, std, device, outdir, *, filename_suffix="",
    behavior_name=None,
) -> str:
    """Interpolate an RGB subspace embedding in end-effector workspace."""
    sequences = np.asarray(arrays["test_sequences"], dtype=np.float32)
    shown = min(12, len(sequences))
    selected = sequences[:shown]
    mode = str(getattr(model, "rollout_mode", "latent")).lower()
    with torch.no_grad():
        initial = torch.from_numpy(selected[:, 0]).to(device)
        predicted = model_rollout(
            model, initial, selected.shape[1]-1, mode,
        )["x_rollout"].detach().cpu().float().numpy()
        state_samples = np.concatenate((selected, predicted), axis=0)
        flat_states = torch.from_numpy(
            state_samples.reshape(-1, state_samples.shape[-1])
        ).to(device)
        projector_parts = []
        for start in range(0, len(flat_states), 2048):
            bases = model.predict_subspace(flat_states[start:start+2048])
            projectors = bases @ bases.transpose(-2, -1)
            projector_parts.append(
                projectors.flatten(start_dim=-2).cpu().numpy()
            )
    projector_features = np.concatenate(projector_parts)
    color_embedding = _fit_projector_color_embedding(
        projector_features, model.grassmann_rank,
    )
    rgb, _ = _apply_projector_color_embedding(
        projector_features, color_embedding,
    )
    rollout_colors = rgb.reshape(*state_samples.shape[:2], 3)[len(selected):]

    physical_samples = denorm(state_samples, mean, std)
    end_effectors = _pam_kinematic_trajectory(
        physical_samples[..., :4].reshape(-1, 4)
    )[:, -1]
    workspace_center = end_effectors.mean(axis=0)
    centered_workspace = end_effectors-workspace_center
    _, _, workspace_directions = np.linalg.svd(
        centered_workspace, full_matrices=False
    )
    workspace_basis = workspace_directions[:2].T
    workspace_2d = centered_workspace @ workspace_basis
    rounded = np.round(workspace_2d, decimals=8)
    _, unique_indices = np.unique(rounded, axis=0, return_index=True)
    workspace_2d = workspace_2d[unique_indices]
    rgb = rgb[unique_indices]
    padding = 0.06*np.maximum(np.ptp(workspace_2d, axis=0), 1.0e-6)
    lower = workspace_2d.min(axis=0)-padding
    upper = workspace_2d.max(axis=0)+padding
    grid_x, grid_y = np.meshgrid(
        np.linspace(lower[0], upper[0], 160),
        np.linspace(lower[1], upper[1], 160),
    )
    triangulation = mtri.Triangulation(workspace_2d[:, 0], workspace_2d[:, 1])
    rgb_grid = np.stack([
        np.asarray(mtri.LinearTriInterpolator(
            triangulation, rgb[:, channel]
        )(grid_x, grid_y).filled(np.nan))
        for channel in range(3)
    ], axis=-1)
    alpha = np.isfinite(rgb_grid).all(axis=-1).astype(float)
    rgba_grid = np.dstack((np.nan_to_num(rgb_grid, nan=1.0), alpha))

    truth_physical = denorm(selected, mean, std)
    prediction_physical = denorm(predicted, mean, std)
    figure, axis = plt.subplots(figsize=(8.2, 6.6))
    axis.imshow(
        rgba_grid, origin="lower",
        extent=(lower[0], upper[0], lower[1], upper[1]),
        interpolation="bilinear", aspect="equal",
    )
    for index, (truth, estimate) in enumerate(zip(
        truth_physical, prediction_physical
    )):
        truth_end = _pam_kinematic_trajectory(truth[:, :4])[:, -1]
        estimate_end = _pam_kinematic_trajectory(estimate[:, :4])[:, -1]
        truth_2d = (truth_end-workspace_center) @ workspace_basis
        estimate_2d = (estimate_end-workspace_center) @ workspace_basis
        axis.plot(
            *truth_2d.T, color="black", linewidth=1.15,
            label="ground truth" if index == 0 else None,
        )
        colors = rollout_colors[index]
        axis.add_collection(LineCollection(
            np.stack((estimate_2d[:-1], estimate_2d[1:]), axis=1),
            colors=0.5 * (colors[:-1] + colors[1:]), linewidths=2,
            label="predicted" if index == 0 else None, zorder=3,
        ))
    behavior_title = (
        "" if behavior_name is None
        else " · " + str(behavior_name).replace("_", " ").title()
    )
    axis.set(
        xlabel="end-effector workspace PC1",
        ylabel="end-effector workspace PC2",
        title="Robot-arm Grassmann subspaces: absolute RGB" + behavior_title,
        xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]),
    )
    axis.legend(frameon=False)
    path = outdir / f"05c_robot_arm_state_conditioned_grassmann_rgb{filename_suffix}.png"
    savefig(figure, path)
    return str(path)


@torch.no_grad()
def _plot_robot_arm_grassmann_behavior_comparison(model, arrays, mean, std, device, outdir):
    """Compare measured fast/slow subspaces with shared RGB and workspace axes."""
    sequences = np.asarray(arrays["test_sequences"], dtype=np.float32)
    labels = _pam_test_behavior_labels(arrays, len(sequences))
    groups = [np.flatnonzero(np.char.endswith(labels.astype(str), speed))
              for speed in ("fast", "slow")]
    if any(len(indices) == 0 for indices in groups):
        return {"omitted": "Fast and slow held-out behavior labels are both required."}
    # Balance the two behaviors so one does not dominate the fitted color map.
    shown = min(16, *(len(indices) for indices in groups))
    # Cover spatial cells rather than selecting evenly spaced dataset indices.
    # Sample through each window for inexpensive coverage estimates; selected
    # windows retain the existing rendering and subspace-color calculations.
    coverage_times = np.unique(np.linspace(
        0, sequences.shape[1] - 1, min(32, sequences.shape[1]), dtype=int,
    ))
    positions = (sequences[:, coverage_times, :4] * np.asarray(std)[:4]
                 + np.asarray(mean)[:4])
    workspace_samples = _pam_kinematic_trajectory(positions.reshape(-1, 4))[:, -1]
    workspace_samples = workspace_samples.reshape(len(sequences), len(coverage_times), 3)
    selected_indices = np.concatenate([
        pamy_workspace_windows(workspace_samples, indices, shown) for indices in groups
    ])
    selected = sequences[selected_indices]
    stride = max(1, int(np.ceil(selected.shape[0] * selected.shape[1] / 12000)))
    time_indices = np.unique(np.append(np.arange(0, selected.shape[1], stride), selected.shape[1] - 1))
    selected = selected[:, time_indices]
    flat = selected.reshape(-1, selected.shape[-1])
    was_training = model.training
    model.eval()
    try:
        parts = []
        for start in range(0, len(flat), 2048):
            basis = model.predict_subspace(torch.from_numpy(flat[start:start + 2048]).to(device))
            parts.append((basis @ basis.transpose(-2, -1)).flatten(start_dim=-2).cpu().numpy())
        features = np.concatenate(parts)
        embedding = _fit_projector_color_embedding(features, model.grassmann_rank)
        colors, distances = _apply_projector_color_embedding(features, embedding)
    finally:
        model.train(was_training)

    physical = denorm(selected, mean, std)
    endpoints = _pam_kinematic_trajectory(physical[..., :4].reshape(-1, 4))[:, -1]
    centered = endpoints - endpoints.mean(axis=0)
    _, _, directions = np.linalg.svd(centered, full_matrices=False)
    workspace = (centered @ directions[:2].T).reshape(*selected.shape[:2], 2)
    colors = colors.reshape(*selected.shape[:2], 3)
    lower, upper = workspace.min(axis=(0, 1)), workspace.max(axis=(0, 1))
    padding = 0.06 * np.maximum(upper - lower, 1e-6)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)
    for panel, (axis, speed) in enumerate(zip(axes, ("Fast", "Slow"))):
        for trajectory, rgb in zip(workspace[panel * shown:(panel + 1) * shown],
                                   colors[panel * shown:(panel + 1) * shown]):
            axis.add_collection(LineCollection(
                np.stack((trajectory[:-1], trajectory[1:]), axis=1),
                colors=0.5 * (rgb[:-1] + rgb[1:]), linewidths=2,
            ))
        axis.set(title=f"{speed} · measured held-out trajectories",
                 xlabel="shared end-effector workspace PC1", ylabel="shared workspace PC2",
                 xlim=(lower[0] - padding[0], upper[0] + padding[0]),
                 ylim=(lower[1] - padding[1], upper[1] + padding[1]))
        axis.set_aspect("equal", adjustable="box")
    figure.suptitle("Robot-arm Grassmann subspaces · shared RGB mapping for fast and slow")
    return {
        "plot": savefig(figure, outdir / "05e_robot_arm_fast_slow_grassmann_rgb.png"),
        "trajectories_per_behavior": shown,
        "trajectory_indices": selected_indices.tolist(),
        "selection_method": "greedy 3-D end-effector workspace cell coverage per behavior",
        "coverage_sample_time_indices": coverage_times.tolist(),
        "sample_time_indices": time_indices.tolist(),
        "rgb_color_definition": "one projector PCA embedding and absolute distance scale fitted jointly to fast and slow",
        "workspace_definition": "one end-effector PCA plane fitted jointly to fast and slow",
        "chordal_distance_range": [float(distances.min()), float(distances.max())],
    }


def _plot_dysts_grassmann_contourf(
    model, arrays, mean, std, device, outdir, system_name,
) -> str:
    """Plot three physical coordinate slices of a 3-D subspace field."""
    sequences = _system_plot_sequences(arrays, system_name)
    physical = denorm(sequences, mean, std)
    flattened = physical.reshape(-1, 3)
    lower, upper = np.percentile(flattened, (1.0, 99.0), axis=0)
    center = np.median(flattened, axis=0)
    axis_pairs = ((0, 1), (0, 2), (1, 2))
    labels = (("x", "y"), ("x", "z"), ("y", "z"))
    grid_size = 72
    grids = []
    normalized_grids = []
    for first, second in axis_pairs:
        grid_first, grid_second = np.meshgrid(
            np.linspace(lower[first], upper[first], grid_size),
            np.linspace(lower[second], upper[second], grid_size),
        )
        states = np.repeat(center[None], grid_size * grid_size, axis=0)
        states[:, first] = grid_first.ravel()
        states[:, second] = grid_second.ravel()
        grids.append((grid_first, grid_second))
        normalized_grids.append(((states - mean) / std).astype(np.float32))

    reference = torch.from_numpy(
        ((center - mean) / std).astype(np.float32)[None]
    ).to(device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        basis = model.predict_subspace(reference)
        reference_projector = basis @ basis.transpose(-2, -1)
        distance_fields = []
        for normalized_grid in normalized_grids:
            parts = []
            for start in range(0, len(normalized_grid), 2048):
                states = torch.from_numpy(
                    normalized_grid[start:start + 2048]
                ).to(device)
                bases = model.predict_subspace(states)
                projectors = bases @ bases.transpose(-2, -1)
                parts.append((
                    torch.linalg.matrix_norm(
                        projectors - reference_projector, ord="fro"
                    ) / np.sqrt(2.0)
                ).cpu().numpy())
            distance_fields.append(
                np.concatenate(parts).reshape(grid_size, grid_size)
            )
    if was_training:
        model.train()

    maximum = max(float(field.max()) for field in distance_fields)
    levels = np.linspace(0.0, max(maximum, 1.0e-8), 33)
    figure, axes = plt.subplots(1, 3, figsize=(16.0, 4.8))
    contour = None
    for axis, pair, label, grid, field in zip(
        axes, axis_pairs, labels, grids, distance_fields
    ):
        contour = axis.contourf(*grid, field, levels=levels, cmap="viridis")
        first, second = pair
        trajectories = physical if canonical_system(system_name) in MULTISTABLE_3D else physical[:DYSTS_PLOT_HOLDOUTS]
        for trajectory in trajectories:
            stride = max(1, len(trajectory) // 2000)
            axis.plot(
                trajectory[::stride, first], trajectory[::stride, second],
                color="white", linewidth=0.55, alpha=0.45,
            )
        fixed_axis = ({0, 1, 2} - set(pair)).pop()
        axis.set(
            xlabel=label[0], ylabel=label[1],
            title=f"{label[0]}–{label[1]} slice at "
                  f"{'xyz'[fixed_axis]}={center[fixed_axis]:.3g}",
        )
    figure.colorbar(contour, ax=axes, pad=0.02).set_label(
        r"Chordal distance $\|P(x)-P(x_{ref})\|_F/\sqrt{2}$"
    )
    display_name = str(np.asarray(arrays.get("system_display_name", arrays.get("dysts_system", "3-D system"))).item())
    figure.suptitle(
        f"{display_name}: state-conditioned Grassmann subspace field",
        weight="bold",
    )
    path = outdir / f"05_{system_name}_state_conditioned_grassmann_contourf.png"
    savefig(figure, path)
    return str(path)


@torch.no_grad()
def plot_linearization_defect(
    model, arrays: Dict[str, np.ndarray], mean: np.ndarray, std: np.ndarray,
    device: torch.device, outdir: Path, system: str, *,
    max_robot_trajectories: int = 6, eps: float = 1.0e-8,
) -> Dict[str, Any]:
    """Plot ground-truth one-step latent linearization defects."""
    if "pam_window_size" in arrays and not arrays.get("pam_plot_selected", False):
        return plot_linearization_defect(
            model, pamy_visualization_arrays(arrays), mean, std, device, outdir, system,
            max_robot_trajectories=8, eps=eps)
    sequences = (_system_plot_sequences(arrays, system) if canonical_system(system) in GOODWIN_NAMES
                 else np.asarray(arrays["test_sequences"], dtype=np.float32))
    if sequences.ndim != 3 or sequences.shape[1] < 2:
        return {}
    leading, following = sequences[:, :-1], sequences[:, 1:]
    flat_leading = leading.reshape(-1, leading.shape[-1])
    flat_following = following.reshape(-1, following.shape[-1])
    encoded_leading, encoded_following, predicted_following = [], [], []
    for start in range(0, len(flat_leading), 4096):
        stop = start + 4096
        x_t = torch.from_numpy(flat_leading[start:stop]).to(device)
        x_next = torch.from_numpy(flat_following[start:stop]).to(device)
        if bool(getattr(model, "grassmann", False)) and hasattr(model, "predict_subspace"):
            z_t_batch = model.encode(x_t)
            encoded_leading.append(z_t_batch)
            encoded_following.append(model.encode(x_next))
            predicted_following.append(model.apply_operator(z_t_batch))
        else:
            encoded_leading.append(model.encode(x_t))
            encoded_following.append(model.encode(x_next))
    z_t = torch.cat(encoded_leading)
    z_next = torch.cat(encoded_following)
    predicted = torch.cat(predicted_following) if predicted_following else model.apply_operator(z_t)
    defect = torch.linalg.vector_norm(z_next - predicted, dim=-1)
    defect_relative = defect / torch.linalg.vector_norm(
        z_next, dim=-1
    ).clamp_min(float(eps))
    relative_values = defect_relative.reshape(leading.shape[:2]).cpu().numpy()

    physical = denorm(sequences, mean, std)
    destination = physical[:, 1:]
    colors = relative_values
    finite = colors[np.isfinite(colors)]
    if not len(finite):
        return {}
    color_min = 0.0
    color_max = float(np.quantile(finite, 0.98))
    if color_max <= color_min:
        color_max = max(float(finite.max()), 1.0e-12)
    norm = matplotlib.colors.Normalize(color_min, color_max, clip=True)
    cmap = plt.get_cmap("viridis")
    key = str(system).lower()

    if key in {"goodwin", "goodwin_oscillator", "goodwin-oscillator"}:
        figure = plt.figure(figsize=(9, 7.2))
        axis = figure.add_subplot(111, projection="3d")
        for trajectory, values in zip(destination, colors):
            axis.plot(*trajectory[:, :3].T, color="#9CA3AF", alpha=0.28, lw=0.8)
            axis.scatter(*trajectory[:, :3].T, c=values, cmap=cmap, norm=norm, s=8)
        axis.set(xlabel="x₁", ylabel="x₂", zlabel="x₃",
                 title="Ground-truth Koopman linearization defect")
    elif key in {"monkey_reaching", "reaching", "comp_neurosci_reaching", "neural_reaching"}:
        neural_count = int(np.asarray(arrays["reaching_neural_unit_count"]).item())
        figure, axis = plt.subplots(figsize=(8.2, 7))
        for trajectory, values in zip(destination, colors):
            cursor = trajectory[:, neural_count:neural_count + 2]
            axis.plot(cursor[:, 0], cursor[:, 1], color="#9CA3AF", alpha=0.22, lw=0.7)
            axis.scatter(cursor[:, 0], cursor[:, 1], c=values, cmap=cmap, norm=norm, s=7)
        axis.set(xlabel="Cursor x", ylabel="Cursor y",
                 title="Ground-truth latent linearization defect in cursor space")
        axis.set_aspect("equal", adjustable="datalim")
    elif key in {"pam", "pamy2", "pneumatic_robot_arm_pam"}:
        state_dim = destination.shape[-1]
        columns, rows = min(4, state_dim), int(np.ceil(state_dim / min(4, state_dim)))
        figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 2.5 * rows),
                                    squeeze=False, sharex=True)
        shown = min(max(int(max_robot_trajectories), 1), len(destination))
        selected = np.linspace(0, len(destination) - 1, shown, dtype=int)
        times = np.asarray(arrays.get("pam_test_time"))
        if times.shape != sequences.shape[:2]:
            dt = float(np.asarray(arrays.get("pam_dt", 1.0)))
            times = np.broadcast_to(np.arange(sequences.shape[1]) * dt, sequences.shape[:2])
        for dimension, axis in enumerate(axes.flat):
            if dimension >= state_dim:
                axis.set_visible(False)
                continue
            for index in selected:
                time, values = times[index, 1:], destination[index, :, dimension]
                axis.plot(time, values, color="#9CA3AF", alpha=0.25, lw=0.7)
                axis.scatter(time, values, c=colors[index], cmap=cmap, norm=norm, s=6)
            axis.set_title(f"State dimension {dimension}")
            axis.grid(alpha=0.18)
        figure.supxlabel("Time")
        figure.supylabel("Physical state value")
        figure.suptitle("Ground-truth Koopman linearization defect by robot state")
    else:
        return {}

    scalar = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    figure.colorbar(scalar, ax=[a for a in figure.axes if a.get_visible()], shrink=0.82,
                    label=r"Relative defect $\|z_{t+1}-Kz_t\|/(\|z_{t+1}\|+\epsilon)$")
    path = savefig(figure, outdir / "linearization_defect.png")
    return {
        "plot": path,
        "uses_ground_truth_transitions": True,
        "color_quantity": "relative_latent_linearization_defect",
        "color_scale": [color_min, color_max],
        "transition_count": int(defect.numel()),
        "mean_defect": float(defect.mean().cpu()),
        "std_defect": float(defect.std(unbiased=False).cpu()),
        "max_defect": float(defect.max().cpu()),
        "mean_relative_defect": float(defect_relative.mean().cpu()),
        "std_relative_defect": float(defect_relative.std(unbiased=False).cpu()),
        "max_relative_defect": float(defect_relative.max().cpu()),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Visualize KAE and GKAE rollouts and subspaces."
    )
    p.add_argument("--checkpoint", default=None,
                   help="Checkpoint path. If omitted, available trained models are listed for selection.")
    p.add_argument(
        "--checkpoint_type",
        choices=("best", "last"),
        default="best",
        help=(
            "Checkpoint to auto-discover when --checkpoint is omitted: "
            "best.pt or latest.pt in outputs/<system>/<model>/ "
            "(default: best)."
        ),
    )
    p.add_argument("--outdir", default=None)
    from src.utils import DEFAULT_SYSTEM_CONFIG
    p.add_argument(
        "--system-config", type=Path, default=DEFAULT_SYSTEM_CONFIG,
        help="Shared runtime configuration (default: src/configs/system_config.yaml).",
    )
    p.add_argument("--device", default=None,
                   help="Override compute.device from the system config.")
    p.add_argument("--rollout_steps", type=int, default=None,
                   help="Long rollout horizon. Default: checkpoint training prediction/linearity steps.")
    p.add_argument(
        "--rollout_mode", default=None,
        help="Rollout override for supported models: latent or reencode.",
    )
    p.add_argument("--reencode_every", type=int, default=None,
                   help="Override how often reencode mode projects through decoder-encoder.")
    p.add_argument(
        "--animate", action="store_true",
        help="Generate sequential MP4 animations (disabled by default).",
    )
    p.add_argument(
        "--reaching_condition_average", action="store_true",
        help=(
            "Additionally evaluate reaching rollouts after averaging repeated "
            "test trials within every target direction."
        ),
    )
    return p


def auto_find_checkpoint(
    user_path: Optional[str] = None,
    checkpoint_type: str = "best",
) -> Path:
    if user_path:
        p = Path(user_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p

    root = PROJECT_ROOT
    outputs = root / "outputs"
    candidates = []
    # Discover only outputs/<system>/<model>/<checkpoint>, excluding nested archives.
    if outputs.exists():
        if checkpoint_type == "best":
            candidates.extend(outputs.glob("*/*/best.pt"))
        elif checkpoint_type == "last":
            candidates.extend(outputs.glob("*/*/latest.pt"))
            latest_run_dirs = {path.parent for path in candidates}
            # Prefer step checkpoints; retain support for historical epoch checkpoints.
            run_dirs = {path.parent for pattern in ("*/*/step_*.pt", "*/*/epoch_*.pt")
                        for path in outputs.glob(pattern)}
            for run_dir in run_dirs - latest_run_dirs:
                snapshots = list(run_dir.glob("step_*.pt")) or list(run_dir.glob("epoch_*.pt"))
                if snapshots:
                    candidates.append(max(snapshots, key=lambda path: int(path.stem.split("_", 1)[1])))
        else:
            raise ValueError(
                f"Unsupported checkpoint_type={checkpoint_type!r}; expected 'best' or 'last'."
            )

    if not candidates:
        expected = "best.pt" if checkpoint_type == "best" else "latest.pt or step_*.pt"
        raise FileNotFoundError(
            f"No {expected} found below {outputs}. Train a model first or pass --checkpoint PATH."
        )

    candidates = sorted(candidates, key=lambda p: str(p.relative_to(root)))

    print(f"Available trained models ({checkpoint_type} checkpoints):")
    for index, checkpoint in enumerate(candidates, start=1):
        print(f"  [{index}] {checkpoint.relative_to(root)}")

    while True:
        try:
            selection = input(f"Select a model [1-{len(candidates)}]: ").strip()
        except EOFError as exc:
            raise RuntimeError(
                "Interactive checkpoint selection requires terminal input; "
                "pass --checkpoint PATH for non-interactive use."
            ) from exc
        try:
            selected_index = int(selection)
        except ValueError:
            print("Please enter one of the displayed numbers.")
            continue
        if 1 <= selected_index <= len(candidates):
            return candidates[selected_index - 1]
        print(f"Please enter a number between 1 and {len(candidates)}.")


def resolve_training_horizon(ckpt: Dict[str, Any]) -> int:
    loss_cfg = ckpt.get("loss_config", {}) or {}
    candidates = [
        int(loss_cfg.get("prediction_steps", 0) or 0),
        int(loss_cfg.get("linearity_steps", 0) or 0),
    ]
    return max(max(candidates), 1)


def run_one_horizon(
    model, seq_norm, mean, std, device, steps, mode, outdir, tag, dt,
    max_points, data_cfg, plot_individual_diagnostics=True,
):
    seq_norm = seq_norm[:int(np.isfinite(seq_norm).all(axis=-1).sum())]
    H = min(int(steps), len(seq_norm) - 1)
    x0 = torch.from_numpy(seq_norm[0:1]).float().to(device)
    roll = model_rollout(model, x0, H, mode)

    z_pred = roll["z_ambient_rollout"].squeeze(0).detach().cpu().float().numpy()
    x_pred_norm = roll["x_rollout"].squeeze(0).detach().cpu().float().numpy()

    # Initial state is supplied directly; predicted states must decode stored latents.
    with torch.no_grad():
        x_check = model.decode(roll["z_ambient_rollout"][:, 1:])
    compared_check = x_check
    compared_rollout = roll["x_rollout"][:, 1:]
    max_decode_diff = (
        float((compared_check - compared_rollout).abs().max().item())
        if compared_check.numel() else 0.0
    )

    true_norm = seq_norm[: H + 1]
    true_x = denorm(true_norm, mean, std)
    pred_x = denorm(x_pred_norm, mean, std)
    z_true = encoded_truth(model, true_norm, device).detach().cpu().float().numpy()

    if plot_individual_diagnostics:
        plot_state_space(
            true_x, pred_x, outdir / f"{tag}_state_space.png", max_points, data_cfg
        )
    if plot_individual_diagnostics:
        plot_timeseries(true_x, pred_x, dt, outdir / f"{tag}_timeseries.png")
        plot_errors(true_x, pred_x, z_true, z_pred, dt, outdir / f"{tag}_errors.png")
        plot_latent_norm(z_true, z_pred, dt, outdir / f"{tag}_latent_norm.png")

    state_rmse = np.sqrt(np.mean((pred_x - true_x) ** 2, axis=1))
    latent_rmse = np.sqrt(np.mean((z_pred - z_true) ** 2, axis=1))
    summary = {
        "steps": H,
        "max_decode_consistency_error": max_decode_diff,
        "final_state_rmse": float(state_rmse[-1]),
        "mean_state_rmse": float(state_rmse.mean()),
        "final_latent_rmse": float(latent_rmse[-1]),
        "mean_latent_rmse": float(latent_rmse.mean()),
    }

    return summary


def _goodwin_phase_reference(training_states: np.ndarray) -> Dict[str, np.ndarray]:
    """Fit a common geometric phase coordinate to the training trajectories."""
    flat = training_states.reshape(-1, 3).astype(np.float64)
    center = flat.mean(axis=0)
    scale = flat.std(axis=0)
    scale[scale < 1e-12] = 1.0
    normalized = (flat - center) / scale
    _, _, vectors = np.linalg.svd(normalized, full_matrices=False)
    basis = vectors[:2].T
    coordinates = ((training_states - center) / scale) @ basis
    phase = np.unwrap(np.arctan2(coordinates[..., 1], coordinates[..., 0]), axis=1)
    phase_step = float(np.nanmedian(np.diff(phase, axis=1)))
    if phase_step < 0.0:
        basis[:, 1] *= -1.0
    return {"center": center, "scale": scale, "basis": basis}


def _goodwin_phase(states: np.ndarray, reference: Dict[str, np.ndarray]) -> np.ndarray:
    coordinates = (
        (np.asarray(states, dtype=np.float64) - reference["center"])
        / reference["scale"]
    ) @ reference["basis"]
    return np.arctan2(coordinates[..., 1], coordinates[..., 0])


def _style_3d_axis(axis) -> None:
    """Apply the clean 3-D presentation shared by dynamical-system plots."""
    axis.view_init(elev=22, azim=-55)
    axis.grid(False)
    for coordinate_axis in (axis.xaxis, axis.yaxis, axis.zaxis):
        coordinate_axis.pane.fill = False
        coordinate_axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
    axis.set_box_aspect((1.0, 1.0, 0.82))


def _save_goodwin_sequential_rollout_mp4(
    source: np.ndarray, prediction: np.ndarray, prediction_phase: np.ndarray,
    phase_norm: plt.Normalize, source_count: int,
    limits: Tuple[np.ndarray, np.ndarray], path: Path,
) -> str:
    """Animate all cumulative Goodwin rollouts in a synchronized flow."""
    point_indices = np.linspace(
        0, prediction.shape[1] - 1,
        min(prediction.shape[1], ROLLOUT_ANIMATION_MAX_POINTS), dtype=int,
    )
    point_indices = np.unique(point_indices)
    prediction = prediction[:, point_indices]
    prediction_phase = prediction_phase[:, point_indices]
    source_indices = np.linspace(
        0, source_count - 1,
        min(source_count, ROLLOUT_ANIMATION_MAX_POINTS), dtype=int,
    )
    source_indices = np.unique(source_indices)
    figure = plt.figure(figsize=(7.0, 6.0))
    axis = figure.add_axes((0.20, 0.08, 0.75, 0.86), projection="3d")
    _draw_3d_trajectories(axis, source[:, source_indices], linewidth=0.65, alpha=0.48)
    _style_3d_axis(axis)
    axis.set(xlabel="X", ylabel="Y", zlabel="Z")
    lower, upper = limits
    axis.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), zlim=(lower[2], upper[2]))

    rollout_artists = []
    for _ in prediction:
        artist = Line3DCollection(
            np.empty((0, 2, 3)), cmap="twilight", norm=phase_norm,
            linewidth=1.5,
        )
        axis.add_collection3d(artist, autolim=False)
        rollout_artists.append(artist)

    samples_per_trajectory = prediction.shape[1] - 1
    total_frames = min(samples_per_trajectory, ROLLOUT_ANIMATION_MAX_FRAMES)

    def update(frame: int):
        visible_segments = min(
            samples_per_trajectory,
            ((frame + 1) * samples_per_trajectory + total_frames - 1)
            // total_frames,
        )
        sample_count = visible_segments + 1
        for trajectory, phase, artist in zip(
            prediction, prediction_phase, rollout_artists
        ):
            valid = (
                np.isfinite(trajectory[:sample_count - 1]).all(axis=1)
                & np.isfinite(trajectory[1:sample_count]).all(axis=1)
                & np.isfinite(phase[:sample_count - 1])
            )
            segments = np.stack((
                trajectory[:sample_count - 1], trajectory[1:sample_count]
            ), axis=1)[valid]
            artist.set_segments(segments)
            artist.set_array(phase[:sample_count - 1][valid])
        return tuple(rollout_artists)

    movie = animation.FuncAnimation(
        figure, update, frames=total_frames, interval=50, blit=False,
    )
    writer = animation.FFMpegWriter(
        fps=20, codec="h264", extra_args=["-pix_fmt", "yuv420p"],
    )
    movie.save(
        path, writer=writer,
        progress_callback=lambda frame, _: _animation_progress(
            frame + 1, total_frames
        ),
    )
    plt.close(figure)
    return str(path)


def _normalized_dtw(
    reference: np.ndarray, estimate: np.ndarray,
    max_samples: int = 1000, band_fraction: float = 0.1,
) -> float:
    """Path-length-normalized multivariate DTW in normalized state space."""
    stride = max(1, int(np.ceil(max(len(reference), len(estimate)) / max_samples)))
    reference = np.asarray(reference[::stride], dtype=np.float64)
    estimate = np.asarray(estimate[::stride], dtype=np.float64)
    n, m = len(reference), len(estimate)
    band = max(abs(n - m), int(np.ceil(max(n, m) * band_fraction)))
    previous = np.full(m + 1, np.inf)
    previous_length = np.zeros(m + 1, dtype=np.int32)
    previous[0] = 0.0
    for i in range(1, n + 1):
        current = np.full(m + 1, np.inf)
        current_length = np.zeros(m + 1, dtype=np.int32)
        for j in range(max(1, i - band), min(m, i + band) + 1):
            candidates = (previous[j], current[j - 1], previous[j - 1])
            selected = int(np.argmin(candidates))
            parent_lengths = (
                previous_length[j], current_length[j - 1], previous_length[j - 1]
            )
            distance = float(np.sqrt(np.mean((reference[i - 1] - estimate[j - 1]) ** 2)))
            current[j] = candidates[selected] + distance
            current_length[j] = parent_lengths[selected] + 1
        previous, previous_length = current, current_length
    return float(previous[m] / max(int(previous_length[m]), 1))


def _rollout_ndtw(reference: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    return np.asarray([
        _normalized_dtw(truth[valid], prediction[valid])
        for truth, prediction in zip(reference, estimate)
        for valid in [np.isfinite(truth).all(axis=-1)]
    ])


def _goodwin_nrmse(reference: np.ndarray, estimate: np.ndarray) -> float:
    valid = np.isfinite(reference).all(axis=-1)
    reference, estimate = reference[valid].astype(np.float64), estimate[valid].astype(np.float64)
    scale = np.sqrt(np.mean((reference - reference.mean(axis=0)) ** 2))
    error = np.sqrt(np.mean((reference - estimate) ** 2))
    return float(error / max(float(scale), 1e-12))


def _plot_goodwin_phase_diagnostics(
    model, arrays, mean, std, outdir, truth_norm, prediction_norm, truth,
    prediction, max_points, dt, animate,
) -> Dict[str, Any]:
    """Goodwin-specific phase figures and metrics, after shared rollout evaluation."""
    n_traj, count = truth.shape[:2]
    rollout_ndtw = _rollout_ndtw(truth_norm, prediction_norm)
    source = denorm(np.asarray(arrays["train_sequences"]), mean, std)

    phase_stride = max(1, source.shape[1] // max(int(max_points), 2))
    phase_reference = _goodwin_phase_reference(source[:, ::phase_stride])
    truth_phase = _goodwin_phase(truth, phase_reference)
    prediction_phase = _goodwin_phase(prediction, phase_reference)
    phase_norm = plt.Normalize(vmin=-np.pi, vmax=np.pi)

    # Both figures share the same canvas, limits, and training-data context.
    source_count = min(source.shape[1], max_points)
    visible = np.concatenate((source[:, :source_count].reshape(-1, 3),
                              truth.reshape(-1, 3), prediction.reshape(-1, 3)))
    visible = visible[np.isfinite(visible).all(axis=1)]
    lower = np.minimum(0.0, visible.min(axis=0)) if len(visible) else np.zeros(3)
    upper = np.maximum(1e-8, 1.04 * visible.max(axis=0)) if len(visible) else np.ones(3)
    truth_rollout_path = outdir / "02_goodwin_phase_rollouts_ground_truth.png"
    model_rollout_path = outdir / "02_goodwin_phase_rollouts_model.png"
    for trajectories, phases, path in (
        (truth, truth_phase, truth_rollout_path),
        (prediction, prediction_phase, model_rollout_path),
    ):
        figure = plt.figure(figsize=(7.0, 6.0))
        axis = figure.add_axes((0.20, 0.08, 0.75, 0.86), projection="3d")
        _draw_3d_trajectories(axis, source[:, :source_count], linewidth=0.65, alpha=0.48)
        _draw_3d_trajectories(axis, trajectories, phases=phases,
                              phase_norm=phase_norm, linewidth=1.5)
        _style_3d_axis(axis)
        axis.set(xlabel="X", ylabel="Y", zlabel="Z")
        if len(visible):
            axis.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]),
                     zlim=(lower[2], upper[2]))
        if path == truth_rollout_path:
            colorbar_axis = figure.add_axes((0.075, 0.19, 0.025, 0.64))
            figure.colorbar(plt.cm.ScalarMappable(norm=phase_norm, cmap="twilight"),
                            cax=colorbar_axis, label="Shared geometric phase [rad]")
            colorbar_axis.yaxis.set_ticks_position("left")
            colorbar_axis.yaxis.set_label_position("left")
        figure.savefig(path, dpi=170)
        plt.close(figure)
    animated_rollout_path = None
    if animate:
        animated_rollout_path = _save_goodwin_sequential_rollout_mp4(
            source, prediction, prediction_phase, phase_norm,
            source_count, (lower, upper),
            outdir / "02_goodwin_phase_rollouts_model_sequential.mp4",
        )

    # EDMD-style 2x2 diagnostic: ensemble coordinates, state geometry,
    # learned spectra, and phase error over the autonomous rollout.
    figure = plt.figure(figsize=(13, 8.5), facecolor="#F3F4F6")
    grid = figure.add_gridspec(2, 2, hspace=0.30, wspace=0.24)
    trajectory_colors = plt.cm.turbo(np.linspace(0.08, 0.92, n_traj))
    plot_time = np.arange(count) * float(dt)

    def style_axis(axis, equal: bool = False) -> None:
        axis.set_facecolor("#FCFCFD")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#CBD5E1")
        axis.spines["bottom"].set_color("#CBD5E1")
        axis.grid(True, color="#E2E8F0", linewidth=0.7, alpha=0.75)
        if equal:
            axis.set_aspect("equal", adjustable="box")

    ax_time = figure.add_subplot(grid[0, 0])
    coordinate_colors = ("#2563EB", "#F59E0B", "#10B981")
    for coordinate, color, label in zip(
        range(3), coordinate_colors, ("x", "y", "z")
    ):
        true_low, true_high = np.quantile(
            truth[:, :, coordinate], (0.1, 0.9), axis=0
        )
        pred_low, pred_high = np.nanquantile(
            prediction[:, :, coordinate], (0.1, 0.9), axis=0
        )
        ax_time.fill_between(
            plot_time, true_low, true_high, color=color, alpha=0.10, linewidth=0
        )
        ax_time.fill_between(
            plot_time, pred_low, pred_high, color=color, alpha=0.05, linewidth=0
        )
        ax_time.plot(
            plot_time, truth[:, :, coordinate].mean(axis=0),
            color=color, linewidth=1.8, label=f"{label} data",
        )
        ax_time.plot(
            plot_time, np.nanmean(prediction[:, :, coordinate], axis=0),
            color=color, linestyle=(0, (4, 2)), linewidth=1.5,
            label=f"{label} model",
        )
    ax_time.set(
        xlabel="Time from initial condition", ylabel="Concentration",
        title="Ensemble rollout · lines: mean · bands: 10–90%",
    )
    ax_time.legend(ncol=3, fontsize=8, frameon=False)
    style_axis(ax_time)

    ax_state = figure.add_subplot(grid[0, 1])
    for index, color in enumerate(trajectory_colors):
        ax_state.plot(
            truth[index, :, 0], truth[index, :, 2],
            color="#94A3B8", alpha=0.45, linewidth=1.0,
        )
        ax_state.plot(
            prediction[index, :, 0], prediction[index, :, 2],
            color=color, linewidth=1.35,
        )
    rollout_nrmse = _goodwin_nrmse(truth, prediction)
    ax_state.set(
        xlabel="x concentration", ylabel="z concentration",
        title=(
            f"State-space geometry · NRMSE {rollout_nrmse:.3f} "
            f"· mean nDTW {np.mean(rollout_ndtw):.3f}"
        ),
    )
    style_axis(ax_state, equal=True)

    values = np.linalg.eigvals(model.K.detach().cpu().float().numpy())
    candidates = values[np.imag(values) > 1e-7]
    selected_modes = ([candidates[np.argmin(np.abs(np.abs(candidates) - 1.0))]]
                      if len(candidates) else [])
    ax_spectrum = figure.add_subplot(grid[1, 0])
    angle = np.linspace(0.0, 2.0 * np.pi, 400)
    ax_spectrum.plot(
        np.cos(angle), np.sin(angle), color="#94A3B8", linewidth=1.1
    )
    ax_spectrum.scatter(values.real, values.imag, s=34, label="K")
    if selected_modes:
        modes = np.asarray(selected_modes)
        ax_spectrum.scatter(
            modes.real, modes.imag, color="#EF4444", marker="*", s=115,
            edgecolor="white", linewidth=0.6, zorder=4,
        )
    ax_spectrum.axhline(0.0, color="#CBD5E1", linewidth=0.8)
    ax_spectrum.axvline(0.0, color="#CBD5E1", linewidth=0.8)
    ax_spectrum.set(
        xlabel="Real part", ylabel="Imaginary part",
        title="Koopman spectra · stars: dominant oscillatory modes",
    )
    ax_spectrum.legend(fontsize=8, frameon=False)
    style_axis(ax_spectrum, equal=True)

    ax_phase = figure.add_subplot(grid[1, 1])
    phase_errors_array = np.unwrap(prediction_phase, axis=1) - np.unwrap(truth_phase, axis=1)
    phase_errors_array -= phase_errors_array[:, :1]
    for error, color in zip(phase_errors_array, trajectory_colors):
        ax_phase.plot(plot_time, error, color=color, alpha=0.45, linewidth=0.9)
    ax_phase.plot(
        plot_time, np.nanmedian(phase_errors_array, axis=0),
        color="#0F172A", linewidth=2.2, label="median error",
    )
    ax_phase.axhline(0.0, color="#94A3B8", linewidth=0.9)
    frequency_text = ""
    if selected_modes:
        frequency = float(np.median(
            [abs(np.angle(value)) / (2.0 * np.pi * float(dt))
             for value in selected_modes]
        ))
        frequency_text = f" · operator frequency {frequency:.3f} Hz"
    ax_phase.set(
        xlabel="Time from initial condition", ylabel="Phase error [rad]",
        title=f"Geometric phase fidelity{frequency_text}",
    )
    ax_phase.legend(fontsize=8, frameon=False)
    style_axis(ax_phase)

    production = float(np.asarray(arrays.get("goodwin_production", np.nan)))
    hill = float(np.asarray(arrays.get("goodwin_hill_exponent", np.nan)))
    figure.suptitle(
        f"Goodwin oscillator Koopman AE: production={production:g}, "
        f"Hill exponent={hill:g}",
        fontsize=16, fontweight="bold", color="#0F172A",
    )
    diagnostic_path = savefig(figure, outdir / "01_goodwin_diagnostics.png")

    final_errors = phase_errors_array[:, -1]
    return {
        "diagnostic_plot": diagnostic_path,
        "phase_rollout_ground_truth_plot": str(truth_rollout_path),
        "phase_rollout_model_plot": str(model_rollout_path),
        "phase_rollout_model_sequential_mp4": animated_rollout_path,
        "n_test_rollouts": int(n_traj),
        "rollout_points": int(count),
        "rollout_nrmse": rollout_nrmse,
        "rollout_ndtw": rollout_ndtw.tolist(),
        "mean_rollout_ndtw": float(np.mean(rollout_ndtw)),
        "median_rollout_ndtw": float(np.median(rollout_ndtw)),
        "median_absolute_final_phase_error": float(
            np.nanmedian(np.abs(final_errors))
        ),
        "phase_definition": "PCA geometric phase fitted on source training data",
        "phase_rollout_color_definition": (
            "atan2 of coordinates in a shared standardized PCA plane fitted "
            "once on ground-truth training trajectories"
        ),
    }


@torch.no_grad()


def _save_reaching_sequential_rollout_mp4(
    truth: np.ndarray, prediction: np.ndarray, display_indices: np.ndarray,
    neural_count: int, test_directions: np.ndarray, target_colors: Dict[int, Any],
    limits: Tuple[Tuple[float, float], Tuple[float, float]], path: Path,
) -> str:
    """Animate cumulative reaching rollouts over the complete measured paths."""
    point_indices = np.linspace(
        0, prediction.shape[1] - 1,
        min(prediction.shape[1], ROLLOUT_ANIMATION_MAX_POINTS), dtype=int,
    )
    point_indices = np.unique(point_indices)
    truth = truth[:, point_indices]
    prediction = prediction[:, point_indices]
    figure = plt.figure(figsize=(7.0, 6.0))
    axis = figure.add_axes((0.14, 0.12, 0.82, 0.82))
    rollout_lines = []
    for index in display_indices:
        actual = truth[index]
        color = target_colors[int(test_directions[index])]
        axis.plot(
            actual[:, neural_count], actual[:, neural_count + 1],
            color=color, linewidth=1.45, alpha=0.55,
        )
        line, = axis.plot(
            [], [], color=color, linewidth=1.25, linestyle="--", alpha=0.90,
        )
        rollout_lines.append(line)
        axis.scatter(
            actual[0, neural_count], actual[0, neural_count + 1],
            color=color, s=18, edgecolor="white", linewidth=0.35,
            alpha=0.90, zorder=4,
        )
    axis.set(xlabel="X", ylabel="Y")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#CBD5E1")
    axis.spines["bottom"].set_color("#CBD5E1")
    axis.set(xlim=limits[0], ylim=limits[1])
    figure.tight_layout()

    samples_per_trajectory = prediction.shape[1] - 1
    total_frames = min(samples_per_trajectory, ROLLOUT_ANIMATION_MAX_FRAMES)

    def update(frame: int):
        visible_segments = min(
            samples_per_trajectory,
            ((frame + 1) * samples_per_trajectory + total_frames - 1)
            // total_frames,
        )
        for index, line in zip(display_indices, rollout_lines):
            estimated = prediction[index, :visible_segments + 1]
            line.set_data(
                estimated[:, neural_count], estimated[:, neural_count + 1]
            )
        return tuple(rollout_lines)

    movie = animation.FuncAnimation(
        figure, update, frames=total_frames, interval=50, blit=False,
    )
    writer = animation.FFMpegWriter(
        fps=20, codec="h264", extra_args=["-pix_fmt", "yuv420p"],
    )
    movie.save(
        path, writer=writer,
        progress_callback=lambda frame, _: _animation_progress(
            frame + 1, total_frames
        ),
    )
    plt.close(figure)
    return str(path)


def _save_reaching_neural_heatmap_mp4(
    direction_truth: list[np.ndarray],
    direction_prediction: list[np.ndarray],
    heatmap_directions: list[int],
    plot_time: np.ndarray,
    neural_count: int,
    direction_bins: int,
    activity_min: float,
    activity_max: float,
    path: Path,
) -> str:
    """Efficiently reveal all direction-mean neural heatmaps together."""
    point_indices = np.linspace(
        0, len(plot_time) - 1,
        min(len(plot_time), ROLLOUT_ANIMATION_MAX_POINTS), dtype=int,
    )
    point_indices = np.unique(point_indices)
    animation_time = plot_time[point_indices]
    truth = [values[:, point_indices] for values in direction_truth]
    prediction = [values[:, point_indices] for values in direction_prediction]

    figure, axes = plt.subplots(
        len(heatmap_directions), 2,
        figsize=(12.0, 2.15 * len(heatmap_directions)),
        sharex=True, sharey=True, squeeze=False, facecolor="#F3F4F6",
    )
    extent = (
        float(plot_time[0]), float(plot_time[-1]), float(neural_count), 0.0,
    )
    covers = []
    image = None
    for row, (direction, actual, estimated) in enumerate(zip(
        heatmap_directions, truth, prediction
    )):
        for column, activity in enumerate((actual, estimated)):
            axis = axes[row, column]
            axis.set_facecolor("#FCFCFD")
            image = axis.imshow(
                activity, aspect="auto", interpolation="nearest", cmap="magma",
                vmin=float(activity_min), vmax=float(activity_max), extent=extent,
            )
            axis.axvline(
                0.0, color="white", linestyle=":", linewidth=0.8, alpha=0.7,
            )
            cover = Rectangle(
                (float(plot_time[0]), 0.0),
                float(plot_time[-1] - plot_time[0]), float(neural_count),
                facecolor="#FCFCFD", edgecolor="none", zorder=5,
            )
            axis.add_patch(cover)
            covers.append(cover)
            axis.set_xlim(float(plot_time[0]), float(plot_time[-1]))
            axis.margins(x=0)
        degrees = 360.0 * direction / direction_bins
        axes[row, 0].set_ylabel(f"{degrees:.0f}° target\nNeural unit")

    axes[0, 0].set_title(
        "Ground-truth mean firing rates", loc="left", fontweight="bold"
    )
    axes[0, 1].set_title(
        "Model-rollout mean firing rates", loc="left", fontweight="bold"
    )
    for axis in axes[-1]:
        axis.set_xlabel("Time from movement onset [s]")
    figure.subplots_adjust(
        left=0.10, right=0.90, bottom=0.055, top=0.965,
        hspace=0.18, wspace=0.10,
    )
    if image is not None:
        colorbar = figure.colorbar(
            image, ax=axes.ravel().tolist(), pad=0.015, fraction=0.025,
        )
        colorbar.set_label("Firing rate [Hz]")

    total_frames = min(len(animation_time), HEATMAP_ANIMATION_MAX_FRAMES)

    def update(frame: int):
        visible_index = min(
            len(animation_time) - 1,
            ((frame + 1) * (len(animation_time) - 1) + total_frames - 1)
            // total_frames,
        )
        reveal_time = float(animation_time[visible_index])
        remaining_width = max(0.0, float(plot_time[-1]) - reveal_time)
        for cover in covers:
            cover.set_x(reveal_time)
            cover.set_width(remaining_width)
        return tuple(covers)

    movie = animation.FuncAnimation(
        figure, update, frames=total_frames, interval=50, blit=False,
    )
    writer = animation.FFMpegWriter(
        fps=20, codec="h264", extra_args=["-pix_fmt", "yuv420p"],
    )
    movie.save(
        path, writer=writer,
        progress_callback=lambda frame, _: _animation_progress(
            frame + 1, total_frames
        ),
    )
    plt.close(figure)
    return str(path)


def fit_reaching_ground_truth_jpca(
    arrays: Dict[str, np.ndarray], pca_rank: int = 6,
) -> Dict[str, np.ndarray]:
    """Fit a deterministic jPCA plane on ground-truth training trajectories."""
    neural_count = int(np.asarray(
        arrays["reaching_neural_unit_count"]
    ).item())
    training = np.asarray(
        arrays["train_sequences"][..., :neural_count], dtype=np.float64
    )
    offsets = np.asarray(arrays["reaching_split_offsets"], dtype=np.int64)
    split_directions = np.asarray(
        arrays["reaching_split_directions"], dtype=np.int64
    )
    train_directions = split_directions[offsets[0]:offsets[1]]
    if len(train_directions) != len(training):
        raise ValueError("Training directions do not match reaching trajectories")
    directions = np.unique(train_directions)
    condition_means = np.stack([
        np.nanmean(training[train_directions == direction], axis=0)
        for direction in directions
    ])
    # Churchland preprocessing: remove the condition-independent neural
    # trajectory at every time point before fitting rotational dynamics.
    condition_independent = np.nanmean(condition_means, axis=0)
    residual = condition_means - condition_independent[None]
    samples = residual.reshape(-1, residual.shape[-1])
    center = samples.mean(axis=0)
    centered = samples - center
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    rank = min(max(int(pca_rank), 2), residual.shape[-1])
    pca_basis = eigenvectors[:, np.argsort(eigenvalues)[::-1][:rank]]
    reduced = (residual - center) @ pca_basis
    derivative = np.gradient(reduced, axis=1)
    source = reduced.reshape(-1, rank)
    target = derivative.reshape(-1, rank)
    gram = source.T @ source
    ridge = 1.0e-8 * max(float(np.trace(gram)) / rank, 1.0e-12)
    dynamics = np.linalg.solve(gram + ridge * np.eye(rank), source.T @ target)
    skew_dynamics = 0.5 * (dynamics - dynamics.T)
    values, vectors = np.linalg.eig(skew_dynamics)
    positive = np.flatnonzero(values.imag > 1.0e-10)
    if not len(positive):
        raise ValueError("Ground-truth neural dynamics have no jPCA rotation plane")
    mode = vectors[:, positive[np.argmax(values.imag[positive])]]
    plane, _ = np.linalg.qr(np.column_stack((mode.real, mode.imag)))
    # Fix sign/orientation ambiguities so all models use identical displayed axes.
    pivot = int(np.argmax(np.abs(plane[:, 0])))
    if plane[pivot, 0] < 0.0:
        plane[:, 0] *= -1.0
    projected_dynamics = plane.T @ skew_dynamics @ plane
    if projected_dynamics[0, 1] < 0.0:
        plane[:, 1] *= -1.0
    basis = pca_basis @ plane
    projected = centered @ basis
    total_variance = max(float(np.square(centered).sum()), 1.0e-12)
    variance_ratio = np.square(projected).sum(axis=0) / total_variance
    return {
        "center": center.astype(np.float32),
        "basis": basis.astype(np.float32),
        "condition_independent": condition_independent.astype(np.float32),
        "variance_ratio": variance_ratio.astype(np.float32),
        "angular_rate": np.asarray(
            abs(values[positive[np.argmax(values.imag[positive])]].imag),
            dtype=np.float32,
        ),
        "pca_rank": np.asarray(rank, dtype=np.int64),
    }


def project_reaching_jpca(
    neural: np.ndarray, jpca: Dict[str, np.ndarray],
) -> np.ndarray:
    values = np.asarray(neural)
    values = values[..., :np.asarray(jpca["basis"]).shape[0]]
    reference = np.asarray(jpca["condition_independent"])
    if values.shape[1] > reference.shape[0]:
        raise ValueError("jPCA reference is shorter than the requested rollout")
    residual = values - reference[None, :values.shape[1]]
    return ((residual - jpca["center"]) @ jpca["basis"]).astype(np.float32)


@torch.no_grad()
def plot_reaching_diagnostics(
    model,
    arrays: Dict[str, np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    steps: int,
    mode: str,
    outdir: Path,
    max_points: int,
    dt: float,
    animate: bool = False,
    condition_average: bool = False,
) -> Dict[str, Any]:
    """Create a diagnostic summary and held-out cursor-rollout overlay."""
    test_norm = np.asarray(arrays["test_sequences"])
    n_traj = len(test_norm)
    count = min(int(steps) + 1, test_norm.shape[1], max(int(max_points), 2))
    if n_traj == 0 or count < 3:
        return {}

    initial = torch.from_numpy(test_norm[:, 0]).float().to(device)
    rollout = model_rollout(model, initial, count - 1, mode)
    prediction_norm = rollout["x_rollout"].detach().cpu().float().numpy()
    neural_count = int(np.asarray(
        arrays["reaching_neural_unit_count"]
    ).item())
    # Exclude the first predicted sample. Quantitative evaluation and the error
    # panel begin at the second autonomous prediction.
    metric_start = 2
    truth = denorm(test_norm[:, :count], mean, std)
    prediction = denorm(prediction_norm, mean, std)
    cursor_truth = truth[..., neural_count:neural_count + 4]
    cursor_prediction = prediction[..., neural_count:neural_count + 4]
    cursor_velocity_prediction = cursor_prediction[..., 2:4]
    position_slice = slice(neural_count, neural_count + 2)
    velocity_slice = slice(neural_count + 2, neural_count + 4)
    metric_slices = {
        "all": slice(0, neural_count + 4),
        "neural": slice(0, neural_count),
        "cursor": slice(neural_count, neural_count + 4),
    }

    def metric_report(reference, estimate, group_slice):
        actual = reference[:, metric_start:, group_slice]
        predicted = estimate[:, metric_start:, group_slice]
        ndtw = _rollout_ndtw(actual, predicted)
        return {
            "nrmse": _goodwin_nrmse(actual, predicted),
            "ndtw": ndtw,
            "mean_ndtw": float(np.mean(ndtw)),
            "median_ndtw": float(np.median(ndtw)),
            "rmse": np.sqrt(np.nanmean((predicted - actual) ** 2, axis=-1)),
        }

    def serializable_metric_report(report):
        return {
            "nrmse": float(report["nrmse"]),
            "ndtw": report["ndtw"].tolist(),
            "mean_ndtw": float(report["mean_ndtw"]),
            "median_ndtw": float(report["median_ndtw"]),
            "median_final_rmse": float(np.nanmedian(report["rmse"][:, -1])),
        }

    single_metric_reports = {
        label: metric_report(test_norm[:, :count], prediction_norm, group_slice)
        for label, group_slice in metric_slices.items()
    }
    reaching_time = np.asarray(
        arrays.get("reaching_time", np.arange(test_norm.shape[1]) * float(dt))
    )
    plot_time = reaching_time[:count]
    direction_bins = int(np.asarray(arrays["reaching_direction_bins"]).item())
    selected_directions = np.asarray(
        arrays["reaching_selected_directions"], dtype=np.int64
    )
    split_offsets = np.asarray(arrays["reaching_split_offsets"], dtype=np.int64)
    all_split_directions = np.asarray(
        arrays["reaching_split_directions"], dtype=np.int64
    )
    # Reaching data has two splits (train, test), so its three offsets are
    # [0, end_of_train, end_of_test].
    if split_offsets.size != 3:
        raise ValueError(
            "Reaching split metadata must contain offsets for train and test; "
            f"got {split_offsets.size} offsets"
        )
    test_directions = all_split_directions[
        split_offsets[1]:split_offsets[2]
    ]
    if len(test_directions) != n_traj:
        raise ValueError(
            "Reaching test-direction metadata does not match test trajectories"
        )
    test_obstructed = _reaching_test_obstructed(arrays, n_traj)
    mixed_path_classes = len(np.unique(test_obstructed)) > 1
    path_groups = (
        (("unobstructed", False), ("obstructed", True))
        if mixed_path_classes else
        (("obstructed" if bool(test_obstructed[0]) else "unobstructed",
          bool(test_obstructed[0])),)
    )
    if np.any((selected_directions < 0) |
              (selected_directions >= len(REACHING_TARGET_PALETTE))):
        raise ValueError("Reaching target labels exceed the fixed color palette")
    target_colors = {
        int(target): REACHING_TARGET_PALETTE[int(target)]
        for target in selected_directions
    }

    condition_directions = [
        int(direction) for direction in selected_directions
        if np.any(test_directions == direction)
    ]
    condition_metric_reports = None
    if condition_average:
        condition_truth_full_norm = np.stack([
            np.nanmean(
                test_norm[test_directions == direction, :count], axis=0
            )
            for direction in condition_directions
        ])
        condition_initial = torch.from_numpy(
            condition_truth_full_norm[:, 0]
        ).float().to(device)
        condition_rollout = model_rollout(
            model, condition_initial, count - 1, mode,
        )
        condition_prediction_norm = (
            condition_rollout["x_rollout"].detach().cpu().float().numpy()
        )
        condition_metric_reports = {
            label: metric_report(
                condition_truth_full_norm, condition_prediction_norm, group_slice
            )
            for label, group_slice in metric_slices.items()
        }

    # Match EDMD's clutter control: show the same limited number from every
    # selected direction, evenly spaced through that direction's holdouts.
    represented_directions = [
        int(direction) for direction in selected_directions
        if np.any(test_directions == direction)
    ]
    shown_per_direction = max(
        1, REACHING_PLOT_HOLDOUTS // max(len(represented_directions), 1)
    )
    display_indices = []
    for direction in represented_directions:
        candidates = np.flatnonzero(test_directions == direction)
        if len(candidates) > shown_per_direction:
            candidates = candidates[
                np.linspace(
                    0, len(candidates) - 1,
                    shown_per_direction, dtype=int,
                )
            ]
        display_indices.extend(candidates.tolist())
    display_indices = np.asarray(display_indices, dtype=np.int64)

    def style_axis(axis, equal: bool = False) -> None:
        axis.set_facecolor("#FCFCFD")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#CBD5E1")
        axis.spines["bottom"].set_color("#CBD5E1")
        axis.grid(True, color="#E2E8F0", linewidth=0.7, alpha=0.75)
        if equal:
            axis.set_aspect("equal", adjustable="box")

    def add_trajectory_arrows(
        axis, trajectory: np.ndarray, color, arrow_count: int = 5,
    ) -> None:
        """Place shaft-free arrowheads tangent to interior arc-length points."""
        points = np.asarray(trajectory, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
            return
        finite = np.isfinite(points).all(axis=1)
        segment_length = np.linalg.norm(np.diff(points, axis=0), axis=1)
        segment_length[~(finite[:-1] & finite[1:])] = 0.0
        cumulative = np.concatenate(([0.0], np.cumsum(segment_length)))
        total_length = float(cumulative[-1])
        if not np.isfinite(total_length) or total_length < 1.0e-9:
            return
        locations = np.linspace(0.14, 0.86, max(int(arrow_count), 1)) * total_length
        tangent_span = max(0.02 * total_length, 1.0e-9)
        for location in locations:
            center = min(
                max(int(np.searchsorted(cumulative, location)), 1), len(points) - 1
            )
            left = max(0, int(np.searchsorted(
                cumulative, location - tangent_span, side="left"
            )))
            right = min(len(points) - 1, int(np.searchsorted(
                cumulative, location + tangent_span, side="right"
            )))
            tangent = points[right] - points[left]
            if not np.isfinite(tangent).all() or np.linalg.norm(tangent) < 1.0e-9:
                tangent = points[center] - points[center - 1]
            if not np.isfinite(tangent).all() or np.linalg.norm(tangent) < 1.0e-9:
                continue
            denominator = cumulative[center] - cumulative[center - 1]
            weight = (
                (location - cumulative[center - 1]) / denominator
                if denominator > 1.0e-12 else 1.0
            )
            position = (1.0 - weight) * points[center - 1] + weight * points[center]
            angle = float(np.degrees(np.arctan2(tangent[1], tangent[0])))
            marker = MarkerStyle(">").transformed(
                matplotlib.transforms.Affine2D().rotate_deg(angle)
            )
            arrow_color = color(center) if callable(color) else color
            axis.scatter(
                [position[0]], [position[1]], marker=marker, s=24,
                facecolor=arrow_color, edgecolor="none", alpha=0.92, zorder=6,
            )

    # Plot 1: an EDMD-style 2x2 summary of kinematics, neural population,
    # learned Koopman spectra, and rollout error.
    figure = plt.figure(figsize=(13, 8.5), facecolor="#F3F4F6")
    grid = figure.add_gridspec(2, 2, hspace=0.30, wspace=0.24)

    ax_kinematics = figure.add_subplot(grid[0, 0])
    coordinate_colors = ("#2563EB", "#F59E0B", "#10B981", "#EF4444")
    coordinate_labels = ("position x", "position y", "velocity x", "velocity y")
    kinematics = np.concatenate(
        (truth[..., position_slice], truth[..., velocity_slice]), axis=-1
    )
    predicted_kinematics = np.concatenate(
        (prediction[..., position_slice], prediction[..., velocity_slice]),
        axis=-1,
    )
    for coordinate, color, label in zip(
        range(4), coordinate_colors, coordinate_labels
    ):
        ax_kinematics.plot(
            plot_time, np.nanmean(kinematics[..., coordinate], axis=0),
            color=color, linewidth=1.8, label=f"{label} data",
        )
        ax_kinematics.plot(
            plot_time,
            np.nanmean(predicted_kinematics[..., coordinate], axis=0),
            color=color, linestyle=(0, (4, 2)), linewidth=1.4,
            label=f"{label} model",
        )
    ax_kinematics.axvline(0.0, color="#94A3B8", linestyle=":", linewidth=1.0)
    ax_kinematics.set(
        xlabel="Time from movement onset [s]",
        ylabel="Cursor coordinate",
        title="Direct joint-state cursor prediction · ensemble mean",
    )
    ax_kinematics.legend(ncol=2, fontsize=7, frameon=False)
    style_axis(ax_kinematics)

    ax_population = figure.add_subplot(grid[0, 1])
    true_population = np.nanmean(truth[..., :neural_count], axis=-1)
    predicted_population = np.nanmean(
        prediction[..., :neural_count], axis=-1
    )
    true_low, true_high = np.nanquantile(
        true_population, (0.1, 0.9), axis=0
    )
    pred_low, pred_high = np.nanquantile(
        predicted_population, (0.1, 0.9), axis=0
    )
    ax_population.fill_between(
        plot_time, true_low, true_high, color="#2563EB", alpha=0.15,
        linewidth=0,
    )
    ax_population.fill_between(
        plot_time, pred_low, pred_high, color="#F59E0B", alpha=0.12,
        linewidth=0,
    )
    ax_population.plot(
        plot_time, np.nanmean(true_population, axis=0),
        color="#2563EB", linewidth=2.0, label="data",
    )
    ax_population.plot(
        plot_time, np.nanmean(predicted_population, axis=0),
        color="#F59E0B", linestyle=(0, (4, 2)), linewidth=1.7,
        label="model",
    )
    ax_population.axvline(0.0, color="#94A3B8", linestyle=":", linewidth=1.0)
    ax_population.set(
        xlabel="Time from movement onset [s]",
        ylabel="Mean firing rate [Hz]",
        title="Neural population activity · bands: 10–90%",
    )
    ax_population.legend(frameon=False)
    style_axis(ax_population)

    ax_spectrum = figure.add_subplot(grid[1, 0])
    spectra = [np.linalg.eigvals(model.K.detach().cpu().float().numpy())]
    unit_angle = np.linspace(0.0, 2.0 * np.pi, 400)
    ax_spectrum.plot(
        np.cos(unit_angle), np.sin(unit_angle),
        color="#94A3B8", linewidth=1.1,
    )
    for index, values in enumerate(spectra):
        ax_spectrum.scatter(
            values.real, values.imag, s=34,
            label=f"K[{index}]" if len(spectra) > 1 else "K",
        )
    ax_spectrum.axhline(0.0, color="#CBD5E1", linewidth=0.8)
    ax_spectrum.axvline(0.0, color="#CBD5E1", linewidth=0.8)
    ax_spectrum.set(
        xlabel="Real part", ylabel="Imaginary part",
        title="Learned Koopman spectrum",
    )
    ax_spectrum.legend(fontsize=8, frameon=False)
    style_axis(ax_spectrum, equal=True)

    ax_error = figure.add_subplot(grid[1, 1])
    metric_colors = {
        "all": "#334155", "neural": "#7C3AED", "cursor": "#059669"
    }
    metric_labels = {
        "all": "all state", "neural": "neural", "cursor": "cursor"
    }
    for label, report in single_metric_reports.items():
        values = report["rmse"]
        median = np.nanmedian(values, axis=0)
        low, high = np.nanquantile(values, (0.1, 0.9), axis=0)
        ax_error.fill_between(
            plot_time[metric_start:], low, high,
            color=metric_colors[label], alpha=0.08, linewidth=0,
        )
        ax_error.plot(
            plot_time[metric_start:], median,
            color=metric_colors[label], linewidth=1.7,
            label=(
                f"single {metric_labels[label]} · NRMSE {report['nrmse']:.3f} "
                f"· nDTW {report['mean_ndtw']:.3f}"
            ),
        )
    if condition_metric_reports is not None:
        for label, report in condition_metric_reports.items():
            values = report["rmse"]
            median = np.nanmedian(values, axis=0)
            low, high = np.nanquantile(values, (0.1, 0.9), axis=0)
            ax_error.fill_between(
                plot_time[metric_start:], low, high,
                color=metric_colors[label], alpha=0.05, linewidth=0,
            )
            ax_error.plot(
                plot_time[metric_start:], median,
                color=metric_colors[label], linestyle="--", linewidth=1.5,
                label=(
                    f"average {metric_labels[label]} · NRMSE "
                    f"{report['nrmse']:.3f} · nDTW {report['mean_ndtw']:.3f}"
                ),
            )
    ax_error.set(
        xlabel="Time from movement onset [s]",
        ylabel="RMSE / training standard deviation",
        title="Autonomous rollout error by state group · bands: 10–90%",
    )
    ax_error.legend(frameon=False, fontsize=7)
    style_axis(ax_error)


    path_class = str(np.asarray(arrays["reaching_path_class"]).item())
    figure.suptitle(
        f"Reaching Koopman AE · {path_class} paths · "
        f"direction bins {selected_directions.tolist()}",
        fontsize=16, fontweight="bold", color="#0F172A",
    )
    diagnostic_path = savefig(
        figure, outdir / "01_reaching_diagnostics.png"
    )

    # Plot 2: measured and directly predicted cursor paths, separated by path
    # class so obstacle-avoiding reaches do not obscure straight reaches.
    cursor_paths = {}
    cursor_velocity_paths = {}
    velocity_labels = ("velocity x", "velocity y")
    for path_name, is_obstructed in path_groups:
        class_mask = test_obstructed == is_obstructed
        class_directions = [
            direction for direction in represented_directions
            if np.any(class_mask & (test_directions == direction))
        ]
        class_display_indices = []
        for direction in class_directions:
            candidates = np.flatnonzero(
                class_mask & (test_directions == direction)
            )
            if len(candidates) > shown_per_direction:
                candidates = candidates[np.linspace(
                    0, len(candidates) - 1, shown_per_direction, dtype=int
                )]
            class_display_indices.extend(candidates.tolist())
        class_display_indices = np.asarray(class_display_indices, dtype=np.int64)

        cursor_figure = plt.figure(figsize=(7.0, 6.0))
        cursor_axis = cursor_figure.add_axes((0.14, 0.12, 0.82, 0.82))
        if condition_average:
            cursor_plot_truth = np.stack([
                np.nanmean(
                    truth[class_mask & (test_directions == direction),
                          :, neural_count:neural_count + 2], axis=0
                )
                for direction in class_directions
            ])
            cursor_plot_prediction = np.stack([
                np.nanmean(
                    prediction[class_mask & (test_directions == direction),
                               :, neural_count:neural_count + 2], axis=0
                )
                for direction in class_directions
            ])
            cursor_plot_directions = np.asarray(class_directions)
            truth_alpha = 0.90
        else:
            cursor_plot_truth = truth[
                class_display_indices, :, neural_count:neural_count + 2
            ]
            cursor_plot_prediction = prediction[
                class_display_indices, :, neural_count:neural_count + 2
            ]
            cursor_plot_directions = test_directions[class_display_indices]
            truth_alpha = 0.55

        for actual, estimated, direction in zip(
            cursor_plot_truth, cursor_plot_prediction, cursor_plot_directions
        ):
            color = target_colors[int(direction)]
            cursor_axis.plot(
                actual[:, 0], actual[:, 1], color=color,
                linewidth=1.45, alpha=truth_alpha,
            )
            cursor_axis.plot(
                estimated[:, 0], estimated[:, 1], color=color,
                linewidth=1.25, linestyle="--", alpha=0.90,
            )
            add_trajectory_arrows(cursor_axis, actual, color)
            add_trajectory_arrows(cursor_axis, estimated, color)
            cursor_axis.scatter(
                actual[-1, 0], actual[-1, 1], color=color, s=10,
                marker="o", edgecolor="none", alpha=0.80, zorder=7,
            )
            cursor_axis.scatter(
                estimated[-1, 0], estimated[-1, 1], color=color, s=10,
                marker="o", edgecolor="none", alpha=0.95, zorder=7,
            )
            cursor_axis.scatter(
                actual[0, 0], actual[0, 1], color=color, s=18,
                edgecolor="white", linewidth=0.35, alpha=0.90, zorder=4,
            )
        if condition_average:
            cursor_axis.plot([], [], color="#334155", linewidth=1.45,
                             label="direction-averaged ground truth")
            cursor_axis.plot([], [], color="#334155", linewidth=1.25,
                             linestyle="--", label="direction-averaged rollout")
            cursor_axis.legend(frameon=False, fontsize=8)
        cursor_axis.set_title(
            ("Direction-averaged " if condition_average else "")
            + f"direct cursor-state rollouts · {path_name}"
        )
        cursor_axis.set(xlabel="X", ylabel="Y")
        cursor_axis.set(xlim=(-170.0, 170.0), ylim=(-120.0, 120.0))
        cursor_axis.set_aspect("equal", adjustable="box")
        cursor_axis.grid(False)
        cursor_axis.spines["top"].set_visible(False)
        cursor_axis.spines["right"].set_visible(False)
        cursor_axis.spines["left"].set_color("#CBD5E1")
        cursor_axis.spines["bottom"].set_color("#CBD5E1")
        suffix = f"_{path_name}" if mixed_path_classes else ""
        cursor_paths[path_name] = savefig(
            cursor_figure,
            outdir / f"02_reaching_cursor_rollouts{suffix}.png",
        )

        velocity_figure, velocity_axes = plt.subplots(
            2, 1, figsize=(9.0, 6.5), sharex=True, facecolor="#F3F4F6",
        )
        for component, (axis, label) in enumerate(
            zip(velocity_axes, velocity_labels)
        ):
            for direction in class_directions:
                indices = np.flatnonzero(
                    class_mask & (test_directions == direction)
                )
                color = target_colors[direction]
                measured = np.nanmean(
                    cursor_truth[indices, :, 2 + component], axis=0
                )
                estimated = np.nanmean(
                    cursor_velocity_prediction[indices, :, component], axis=0
                )
                axis.plot(
                    plot_time, measured, color=color, linewidth=1.8,
                    label=f"target {direction + 1}",
                )
                axis.plot(
                    plot_time, estimated, color=color, linewidth=1.5,
                    linestyle=(0, (4, 2)),
                )
            axis.axvline(0.0, color="#94A3B8", linestyle=":", linewidth=1.0)
            axis.set_ylabel(label)
            style_axis(axis)
        velocity_axes[0].set_title(
            f"Direct joint-state cursor velocity · {path_name} "
            "(solid: measured, dashed: model rollout)"
        )
        velocity_axes[0].legend(
            ncol=min(4, max(len(class_directions), 1)),
            fontsize=7, frameon=False,
        )
        velocity_axes[-1].set_xlabel("Time from movement onset [s]")
        velocity_figure.tight_layout()
        cursor_velocity_paths[path_name] = savefig(
            velocity_figure,
            outdir / f"02a_reaching_cursor_velocity_rollouts{suffix}.png",
        )

    # Plot all held-out neural trajectories in one fixed jPCA
    # plane learned only from ground-truth training neural trajectories.
    neural_jpca = fit_reaching_ground_truth_jpca(arrays)
    truth_pca = project_reaching_jpca(test_norm[:, :count], neural_jpca)
    rollout_pca = project_reaching_jpca(prediction_norm, neural_jpca)
    # Churchland-style display orientation: use the across-target spread of
    # ground-truth movement-onset states as the horizontal preparatory axis.
    onset_means = np.stack([
        np.nanmean(
            truth_pca[test_directions == direction, 0], axis=0
        )
        for direction in represented_directions
    ])
    onset_center = np.nanmean(onset_means, axis=0)
    onset_covariance = (onset_means - onset_center).T @ (
        onset_means - onset_center
    )
    _, onset_vectors = np.linalg.eigh(onset_covariance)
    horizontal = onset_vectors[:, -1]
    pivot = int(np.argmax(np.abs(horizontal)))
    if horizontal[pivot] < 0.0:
        horizontal *= -1.0
    display_rotation = np.column_stack((
        horizontal, np.asarray((-horizontal[1], horizontal[0]))
    ))
    truth_pca = (truth_pca - onset_center) @ display_rotation
    rollout_pca = (rollout_pca - onset_center) @ display_rotation
    pca_figure, pca_axes = plt.subplots(
        1, 2, figsize=(12.0, 5.5), sharex=True, sharey=True,
        facecolor="#F3F4F6",
    )
    pca_paths = []
    truth_paths = []
    rollout_paths = []
    trajectory_colors = []
    representative_paths = {}
    for index in range(n_traj):
        direction = int(test_directions[index])
        color = target_colors[direction]
        truth_path = truth_pca[index].copy()
        rollout_path = rollout_pca[index].copy()
        # Place each measured/model pair at the same trial-specific onset point
        # on the horizontal preparatory line, preserving subsequent displacement.
        onset_x = float(truth_path[0, 0])
        truth_path -= np.asarray((0.0, truth_path[0, 1]))
        rollout_path += np.asarray((
            onset_x - rollout_path[0, 0], -rollout_path[0, 1]
        ))
        truth_paths.append(truth_path)
        rollout_paths.append(rollout_path)
        trajectory_colors.append(color)

    # Remove display outliers as matched truth/rollout pairs. Filtering is
    # performed within each target cluster so genuine between-target geometry
    # is preserved. It does not affect jPCA fitting or reported model metrics.
    plotted_directions = np.asarray(test_directions[:n_traj], dtype=np.int64)
    scores = np.asarray([
        max(
            np.nanmax(np.linalg.norm(truth_path - truth_path[0], axis=1)),
            np.nanmax(np.linalg.norm(rollout_path - rollout_path[0], axis=1)),
        )
        for truth_path, rollout_path in zip(truth_paths, rollout_paths)
    ])
    keep = np.isfinite(scores)
    for direction in np.unique(plotted_directions):
        members = np.flatnonzero((plotted_directions == direction) & keep)
        if len(members) < 4:
            continue
        group_scores = scores[members]
        median = float(np.median(group_scores))
        mad = float(np.median(np.abs(group_scores - median)))
        if mad > np.finfo(np.float64).eps:
            keep[members] &= group_scores <= median + 3.0 * 1.4826 * mad
    kept_indices = np.flatnonzero(keep)
    removed_jpca_outliers = int(len(keep) - len(kept_indices))
    truth_paths = [truth_paths[index] for index in kept_indices]
    rollout_paths = [rollout_paths[index] for index in kept_indices]
    trajectory_colors = [trajectory_colors[index] for index in kept_indices]
    plotted_directions = plotted_directions[kept_indices]
    # Reduce clutter without collapsing a target to one condition mean. For
    # each target, sort trials by preparatory-state position and average the
    # same three or four subgroups for ground truth and model rollouts.
    grouped_truth_paths = []
    grouped_rollout_paths = []
    grouped_colors = []
    grouped_directions = []
    for direction in np.unique(plotted_directions):
        members = np.flatnonzero(plotted_directions == direction)
        order = members[np.argsort([truth_paths[index][0, 0] for index in members])]
        subgroup_count = min(4, len(order))
        for subgroup in np.array_split(order, subgroup_count):
            if not len(subgroup):
                continue
            grouped_truth_paths.append(np.nanmean(
                np.stack([truth_paths[index] for index in subgroup]), axis=0
            ))
            grouped_rollout_paths.append(np.nanmean(
                np.stack([rollout_paths[index] for index in subgroup]), axis=0
            ))
            grouped_colors.append(target_colors[int(direction)])
            grouped_directions.append(int(direction))
    truth_paths = grouped_truth_paths
    rollout_paths = grouped_rollout_paths
    trajectory_colors = grouped_colors
    plotted_directions = np.asarray(grouped_directions, dtype=np.int64)
    pca_paths = [path for pair in zip(truth_paths, rollout_paths) for path in pair]
    representative_paths = {}
    for direction, truth_path, rollout_path, color in zip(
        plotted_directions, truth_paths, rollout_paths, trajectory_colors
    ):
        representative_paths.setdefault(
            int(direction), (truth_path, rollout_path, color)
        )

    pca_axes[0].add_collection(LineCollection(
        truth_paths, colors=trajectory_colors, linewidths=1.15, alpha=0.65,
    ))
    pca_axes[1].add_collection(LineCollection(
        rollout_paths, colors=trajectory_colors, linewidths=1.15, alpha=0.72,
    ))
    for direction, (_, _, color) in representative_paths.items():
        pca_axes[0].plot([], [], color=color, linewidth=1.8,
                         label=f"target {direction + 1}")
    for truth_path, rollout_path, color in zip(
        truth_paths, rollout_paths, trajectory_colors
    ):
        add_trajectory_arrows(
            pca_axes[0], truth_path, color, arrow_count=4
        )
        add_trajectory_arrows(
            pca_axes[1], rollout_path, color, arrow_count=4
        )
    for axis, paths in zip(pca_axes, (truth_paths, rollout_paths)):
        starts = np.asarray([path[0] for path in paths])
        ends = np.asarray([path[-1] for path in paths])
        axis.scatter(
            starts[:, 0], starts[:, 1], c=trajectory_colors, s=7, marker="o",
            linewidths=0, alpha=0.45, zorder=5,
        )
        axis.scatter(
            ends[:, 0], ends[:, 1], c=trajectory_colors, s=10, marker="o",
            linewidths=0, alpha=0.72, zorder=7,
        )
    if pca_paths:
        finite_pca = np.concatenate(pca_paths, axis=0)
        finite_pca = finite_pca[np.isfinite(finite_pca).all(axis=1)]
        if len(finite_pca):
            lower = finite_pca.min(axis=0)
            upper = finite_pca.max(axis=0)
            padding = np.maximum(0.08 * (upper - lower), 0.25)
            for axis in pca_axes:
                axis.set_xlim(lower[0] - padding[0], upper[0] + padding[0])
                axis.set_ylim(lower[1] - padding[1], upper[1] + padding[1])
    pca_axes[0].set_title("Ground-truth subgroup trajectories")
    pca_axes[1].set_title("Matched model-rollout subgroups")
    for axis in pca_axes:
        axis.axhline(0.0, color="#94A3B8", linestyle=":", linewidth=1.0, zorder=0)
    for axis in pca_axes:
        axis.set_xlabel("Preparatory-state axis (rotated jPC)")
        axis.set_ylabel("Rotational jPC")
        style_axis(axis, equal=True)
    pca_axes[0].legend(
        ncol=min(2, max(len(represented_directions), 1)),
        fontsize=7, frameon=False,
    )
    plane_variance = float(neural_jpca["variance_ratio"].sum())
    pca_figure.suptitle(
        "Churchland-style reaching subgroup means in fixed ground-truth jPCA plane "
        f"({100.0 * plane_variance:.1f}% variance; circle=start, dot=end; "
        f"{removed_jpca_outliers} display outliers omitted)"
    )
    pca_figure.tight_layout()
    neural_jpca_path = savefig(
        pca_figure, outdir / "02c_reaching_neural_jpca_rollouts.png"
    )

    animated_cursor_path = None
    if animate:
        cursor_limits = ((-170.0, 170.0), (-120.0, 120.0))
        animated_cursor_path = _save_reaching_sequential_rollout_mp4(
            truth, prediction, display_indices, neural_count, test_directions,
            target_colors, cursor_limits,
            outdir / "02_reaching_cursor_rollouts_sequential.mp4",
        )

    # Plot 3: direction-averaged neural activity heatmaps. Each row compares
    # measured and autonomous-rollout firing rates for one movement target.
    direction_truth = []
    direction_prediction = []
    heatmap_directions = []
    for direction in represented_directions:
        indices = np.flatnonzero(test_directions == direction)
        if not len(indices):
            continue
        direction_truth.append(
            np.nanmean(truth[indices, :, :neural_count], axis=0).T
        )
        direction_prediction.append(
            np.nanmean(prediction[indices, :, :neural_count], axis=0).T
        )
        heatmap_directions.append(direction)

    finite_activity = np.concatenate([
        values[np.isfinite(values)]
        for values in (*direction_truth, *direction_prediction)
    ])
    if len(finite_activity):
        activity_min, activity_max = np.quantile(
            finite_activity, (0.01, 0.99)
        )
        if activity_max <= activity_min:
            activity_max = activity_min + 1.0
    else:
        activity_min, activity_max = 0.0, 1.0

    heatmap_figure, heatmap_axes = plt.subplots(
        len(heatmap_directions), 2,
        figsize=(12.0, 2.15 * len(heatmap_directions)),
        sharex=True, sharey=True, squeeze=False,
        facecolor="#F3F4F6",
    )
    extent = (
        float(plot_time[0]), float(plot_time[-1]),
        float(neural_count), 0.0,
    )
    image = None
    for row, (direction, actual, estimated) in enumerate(zip(
        heatmap_directions, direction_truth, direction_prediction
    )):
        for column, activity in enumerate((actual, estimated)):
            axis = heatmap_axes[row, column]
            axis.set_facecolor("#FCFCFD")
            image = axis.imshow(
                activity,
                aspect="auto",
                interpolation="nearest",
                cmap="magma",
                vmin=float(activity_min),
                vmax=float(activity_max),
                extent=extent,
            )
            axis.axvline(
                0.0, color="white", linestyle=":",
                linewidth=0.8, alpha=0.7,
            )
            axis.set_xlim(float(plot_time[0]), float(plot_time[-1]))
            axis.margins(x=0)
        degrees = 360.0 * direction / direction_bins
        heatmap_axes[row, 0].set_ylabel(
            f"{degrees:.0f}° target\nNeural unit"
        )

    heatmap_axes[0, 0].set_title(
        "Ground-truth mean firing rates", loc="left", fontweight="bold"
    )
    heatmap_axes[0, 1].set_title(
        "Model-rollout mean firing rates", loc="left", fontweight="bold"
    )
    for axis in heatmap_axes[-1]:
        axis.set_xlabel("Time from movement onset [s]")
    heatmap_figure.subplots_adjust(
        left=0.10, right=0.90, bottom=0.055, top=0.965,
        hspace=0.18, wspace=0.10,
    )
    if image is not None:
        colorbar = heatmap_figure.colorbar(
            image, ax=heatmap_axes.ravel().tolist(),
            pad=0.015, fraction=0.025,
        )
        colorbar.set_label("Firing rate [Hz]")

    heatmap_path = (
        outdir / "03_reaching_direction_mean_neural_heatmaps.png"
    )
    heatmap_figure.savefig(
        heatmap_path, dpi=170, facecolor=heatmap_figure.get_facecolor()
    )
    plt.close(heatmap_figure)
    neural_heatmap_video_path = None
    if animate:
        neural_heatmap_video_path = _save_reaching_neural_heatmap_mp4(
            direction_truth, direction_prediction, heatmap_directions, plot_time,
            neural_count, direction_bins, float(activity_min), float(activity_max),
            outdir / "03_reaching_direction_mean_neural_heatmaps_animated.mp4",
        )

    return {
        "diagnostic_plot": diagnostic_path,
        "cursor_rollout_plots_by_path_class": cursor_paths,
        "cursor_velocity_rollout_plots_by_path_class": cursor_velocity_paths,
        "neural_jpca_rollout_plot": neural_jpca_path,
        "neural_jpca": {
            "fit_split": "ground-truth training trials",
            "pca_rank": int(neural_jpca["pca_rank"]),
            "plane_variance_ratio": neural_jpca["variance_ratio"].tolist(),
            "angular_rate_per_bin": float(neural_jpca["angular_rate"]),
            "preprocessing": (
                "target-condition means with the ground-truth "
                "condition-independent trajectory removed at each time"
            ),
            "display_alignment": (
                "ground-truth movement-onset states define the horizontal "
                "priming proxy; truth and rollout share each target start"
            ),
        },
        "cursor_prediction": {
            "source": "direct joint-state model rollout",
            "state_coordinates": [
                "cursor_x", "cursor_y", "velocity_x", "velocity_y"
            ],
            "auxiliary_decoder": "retained in source but inactive",
        },
        "cursor_rollout_sequential_mp4": animated_cursor_path,
        "direction_mean_neural_heatmap_plot": str(heatmap_path),
        "direction_mean_neural_heatmap_video": neural_heatmap_video_path,
        "n_test_rollouts": int(n_traj),
        "n_displayed_rollouts": int(len(display_indices)),
        "rollout_points": int(count),
        "rollout_metric_space": (
            "train-standardized joint-state coordinates, reported for all "
            "coordinates, neural rates only, and cursor values only; the "
            "first predicted sample is excluded"
        ),
        "rollout_metric_start_index": metric_start,
        "single_trial_rollout_metrics": {
            label: serializable_metric_report(report)
            for label, report in single_metric_reports.items()
        },
        "path_class": path_class,
        "selected_directions": selected_directions.astype(int).tolist(),
        "condition_averaged_evaluation": (
            {
                "enabled": True,
                "directions": condition_directions,
                "trials_per_direction": {
                    str(direction): int(np.sum(test_directions == direction))
                    for direction in condition_directions
                },
                "rollout_metrics": {
                    label: serializable_metric_report(report)
                    for label, report in condition_metric_reports.items()
                },
            }
            if condition_average else {"enabled": False}
        ),
    }

PAMY2_KINEMATIC_MODEL = (
    ROOT.parent / "data" / "PAMy2" / "kinematics"
    / "pamy2_body_template_no_muscles.xml"
)


def _mjcf_vector(value: Optional[str]) -> np.ndarray:
    if value is None:
        return np.zeros(3, dtype=np.float64)
    vector = np.fromstring(value, sep=" ", dtype=np.float64)
    if vector.size != 3:
        raise ValueError(f"Expected a three-dimensional MJCF vector, got {value!r}")
    return vector


def _find_pam_end_effector_path(
    element: ElementTree.Element,
    parents: Optional[list[ElementTree.Element]] = None,
) -> Optional[tuple[list[ElementTree.Element], ElementTree.Element]]:
    parents = [] if parents is None else parents
    for body in element.findall("body"):
        path = [*parents, body]
        for geometry in body.findall("geom"):
            if geometry.get("name", "").endswith("_racket"):
                return path, geometry
        result = _find_pam_end_effector_path(body, path)
        if result is not None:
            return result
    return None


def _load_pam_kinematic_chain(
    model_path: Path = PAMY2_KINEMATIC_MODEL,
) -> tuple[list[tuple[np.ndarray, Optional[np.ndarray], np.ndarray]], np.ndarray]:
    """Load the nested PAMY2 body offsets and four hinge axes from MJCF."""
    root = ElementTree.parse(model_path).getroot()
    result = _find_pam_end_effector_path(root)
    if result is None:
        raise ValueError(f"No racket/end-effector geometry found in {model_path}")
    body_path, end_effector_geometry = result
    chain = []
    joint_count = 0
    for body in body_path:
        joint = next(
            (
                item for item in body.findall("joint")
                if item.get("type", "hinge") == "hinge"
            ),
            None,
        )
        axis = _mjcf_vector(joint.get("axis")) if joint is not None else None
        joint_position = (
            _mjcf_vector(joint.get("pos"))
            if joint is not None else np.zeros(3, dtype=np.float64)
        )
        chain.append((_mjcf_vector(body.get("pos")), axis, joint_position))
        joint_count += int(joint is not None)
    if joint_count != 4:
        raise ValueError(f"Expected four PAMY2 joints, found {joint_count}")
    return chain, _mjcf_vector(end_effector_geometry.get("pos"))


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    cross = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    return (
        np.cos(angle) * np.eye(3)
        + (1.0 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * cross
    )


def _pam_kinematic_points(
    joint_positions: np.ndarray,
    chain: list[tuple[np.ndarray, Optional[np.ndarray], np.ndarray]],
    end_effector_offset: np.ndarray,
) -> np.ndarray:
    """Return body/joint origins and racket origin for one PAMY2 pose."""
    angles = np.asarray(joint_positions, dtype=np.float64).reshape(-1)
    if angles.size != 4:
        raise ValueError("PAMY2 forward kinematics expects four joint positions")
    rotation = np.eye(3)
    translation = np.zeros(3, dtype=np.float64)
    points = [translation.copy()]
    joint_index = 0
    for body_position, joint_axis, joint_position in chain:
        translation = translation + rotation @ body_position
        if joint_axis is not None:
            pivot = rotation @ joint_position
            rotation = rotation @ _axis_angle_matrix(
                joint_axis, float(angles[joint_index])
            )
            translation = translation + pivot - rotation @ joint_position
            joint_index += 1
        if not np.allclose(points[-1], translation):
            points.append(translation.copy())
    points.append(translation + rotation @ end_effector_offset)
    return np.asarray(points)


def _pam_kinematic_trajectory(joint_positions: np.ndarray) -> np.ndarray:
    chain, end_effector_offset = _load_pam_kinematic_chain()
    return np.stack([
        _pam_kinematic_points(sample, chain, end_effector_offset)
        for sample in np.asarray(joint_positions)
    ])


def _save_pam_joint_rollout_mp4(
    truth: np.ndarray,
    prediction: np.ndarray,
    time: np.ndarray,
    output: Path,
    behavior_label: str,
) -> str:
    """Animate measured and autonomous joint-coordinate rollouts."""
    count = min(len(truth), len(prediction), len(time))
    frame_count = min(count, ROLLOUT_ANIMATION_MAX_FRAMES)
    frame_indices = np.unique(np.linspace(0, count - 1, frame_count, dtype=int))
    colors = ("#365C8D", "#1FA187", "#F4C430", "#D1495B")
    figure, axes = plt.subplots(
        4, 1, figsize=(10, 9), sharex=True, facecolor="#F5F7FA"
    )
    truth_lines = []
    prediction_lines = []
    truth_markers = []
    prediction_markers = []
    combined = np.concatenate((truth[:count], prediction[:count]), axis=0)
    for joint, axis in enumerate(axes):
        lower = float(np.nanmin(combined[:, joint]))
        upper = float(np.nanmax(combined[:, joint]))
        margin = 0.08 * max(upper - lower, 1e-6)
        axis.set_xlim(float(time[0]), float(time[count - 1]))
        axis.set_ylim(lower - margin, upper + margin)
        axis.set_ylabel(f"q{joint + 1} (rad)")
        axis.grid(color="#DCE3EA", linewidth=0.7, alpha=0.75)
        axis.spines[["top", "right"]].set_visible(False)
        measured, = axis.plot([], [], color=colors[joint], linewidth=2.0)
        predicted, = axis.plot(
            [], [], color=colors[joint], linewidth=1.7, linestyle="--"
        )
        measured_marker, = axis.plot([], [], "o", color=colors[joint], markersize=5)
        predicted_marker, = axis.plot(
            [], [], "o", markerfacecolor="white", markeredgecolor=colors[joint],
            markersize=5,
        )
        truth_lines.append(measured)
        prediction_lines.append(predicted)
        truth_markers.append(measured_marker)
        prediction_markers.append(predicted_marker)
    axes[-1].set_xlabel("Time in held-out trajectory (s)")
    figure.suptitle(
        f"PAMY2 joint rollout · {behavior_label}\n"
        "solid: measured · dashed: autonomous model",
        weight="bold",
    )

    def update(frame: int):
        index = int(frame_indices[frame])
        visible = slice(0, index + 1)
        artists = []
        for joint in range(4):
            truth_lines[joint].set_data(time[visible], truth[visible, joint])
            prediction_lines[joint].set_data(time[visible], prediction[visible, joint])
            truth_markers[joint].set_data([time[index]], [truth[index, joint]])
            prediction_markers[joint].set_data(
                [time[index]], [prediction[index, joint]]
            )
            artists.extend((
                truth_lines[joint], prediction_lines[joint],
                truth_markers[joint], prediction_markers[joint],
            ))
        return artists

    movie = animation.FuncAnimation(
        figure, update, frames=len(frame_indices), interval=50, blit=False
    )
    writer = animation.FFMpegWriter(
        fps=20, bitrate=4000,
        metadata={"title": "PAMY2 joint rollout", "artist": "Matplotlib"},
    )
    movie.save(
        output, writer=writer, dpi=130,
        progress_callback=lambda frame, _: _animation_progress(
            frame + 1, len(frame_indices)
        ),
    )
    plt.close(figure)
    return str(output)


def _save_pam_kinematic_rollout_mp4(
    truth: np.ndarray,
    prediction: np.ndarray,
    time: np.ndarray,
    output: Path,
    behavior_label: str,
) -> str:
    """Animate the measured and predicted PAMY2 chains and racket paths."""
    count = min(len(truth), len(prediction), len(time))
    truth_points = _pam_kinematic_trajectory(truth[:count])
    prediction_points = _pam_kinematic_trajectory(prediction[:count])
    frame_count = min(count, ROLLOUT_ANIMATION_MAX_FRAMES)
    frame_indices = np.unique(np.linspace(0, count - 1, frame_count, dtype=int))
    all_points = np.concatenate((
        truth_points.reshape(-1, 3), prediction_points.reshape(-1, 3)
    ))
    lower = np.nanmin(all_points, axis=0)
    upper = np.nanmax(all_points, axis=0)
    center = (lower + upper) / 2.0
    radius = max(float(np.max(upper - lower)) * 0.62, 0.25)

    figure = plt.figure(figsize=(10, 8), facecolor="#F5F7FA")
    axis = figure.add_subplot(111, projection="3d")
    axis.set_facecolor("#F5F7FA")
    axis.set(
        xlim=(center[0] - radius, center[0] + radius),
        ylim=(center[1] - radius, center[1] + radius),
        zlim=(center[2] - radius, center[2] + radius),
        xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)",
    )
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=24, azim=-55)
    axis.grid(True, alpha=0.25)
    floor_x, floor_y = np.meshgrid(
        np.linspace(center[0] - radius, center[0] + radius, 2),
        np.linspace(center[1] - radius, center[1] + radius, 2),
    )
    floor_z = min(0.0, center[2] - radius)
    axis.plot_surface(
        floor_x, floor_y, np.full_like(floor_x, floor_z),
        color="#CBD5E1", alpha=0.18, shade=False,
    )
    truth_chain, = axis.plot(
        [], [], [], "-o", color="#F28C00", linewidth=6, markersize=5,
        label="Measured", solid_capstyle="round",
    )
    prediction_chain, = axis.plot(
        [], [], [], "--o", color="#2563EB", linewidth=4, markersize=4,
        label="Autonomous rollout",
    )
    truth_trail, = axis.plot([], [], [], color="#F28C00", linewidth=1.8, alpha=0.6)
    prediction_trail, = axis.plot(
        [], [], [], color="#2563EB", linewidth=1.5, alpha=0.55,
        linestyle="--",
    )
    time_text = axis.text2D(0.02, 0.95, "", transform=axis.transAxes, color="#64748B")
    axis.legend(loc="upper right", frameon=False)

    def set_line(line, points):
        line.set_data(points[:, 0], points[:, 1])
        line.set_3d_properties(points[:, 2])

    def update(frame: int):
        index = int(frame_indices[frame])
        set_line(truth_chain, truth_points[index])
        set_line(prediction_chain, prediction_points[index])
        set_line(truth_trail, truth_points[:index + 1, -1])
        set_line(prediction_trail, prediction_points[:index + 1, -1])
        time_text.set_text(f"t = {float(time[index] - time[0]):.2f} s")
        return (
            truth_chain, prediction_chain, truth_trail,
            prediction_trail, time_text,
        )

    movie = animation.FuncAnimation(
        figure, update, frames=len(frame_indices), interval=50, blit=False
    )
    writer = animation.FFMpegWriter(
        fps=20, bitrate=5000,
        metadata={"title": "PAMY2 kinematic rollout", "artist": "Matplotlib"},
    )
    movie.save(
        output, writer=writer, dpi=130,
        progress_callback=lambda frame, _: _animation_progress(
            frame + 1, len(frame_indices)
        ),
    )
    plt.close(figure)
    return str(output)


def _pam_test_behavior_labels(arrays, count: int) -> np.ndarray:
    labels = np.asarray(arrays.get("pam_test_behaviors", ())).astype(str).reshape(-1)
    if len(labels) == count:
        return labels
    combined = str(np.asarray(arrays.get("pam_behavior", "unknown")).item())
    return np.full(count, combined, dtype=object)


@torch.no_grad()
def plot_robot_arm_diagnostics(
    model,
    arrays: Dict[str, np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    steps: int,
    mode: str,
    outdir: Path,
    plot_holdouts: int = 6,
    animate: bool = False,
) -> Dict[str, Any]:
    """Create EDMD-style robot-arm diagnostics and separate rollout figures."""
    if "pam_window_size" in arrays and not arrays.get("pam_plot_selected", False):
        return plot_robot_arm_diagnostics(
            model, pamy_visualization_arrays(arrays), mean, std, device, steps, mode,
            outdir, 4, animate)
    test_norm = np.asarray(arrays["test_sequences"])
    count = min(int(steps) + 1, test_norm.shape[1])
    if len(test_norm) == 0 or count < 2:
        return {}
    initial = torch.from_numpy(test_norm[:, 0]).float().to(device)
    rollout = model_rollout(model, initial, count - 1, mode)
    prediction_norm = rollout["x_rollout"].detach().cpu().float().numpy()
    metric_truth = test_norm[:, 1:count]
    rollout_ndtw = _rollout_ndtw(metric_truth, prediction_norm[:, 1:count])
    truth = denorm(test_norm[:, :count], mean, std)
    prediction = denorm(prediction_norm, mean, std)

    one_step_inputs = test_norm[:, :count - 1].reshape(-1, test_norm.shape[-1])
    one_step_parts = []
    for start in range(0, len(one_step_inputs), 4096):
        stop = start + 4096
        batch = torch.from_numpy(one_step_inputs[start:stop]).float().to(device)
        predicted = model_rollout(model, batch, 1, mode)["x_rollout"][:, 1]
        one_step_parts.append(predicted.detach().cpu().float().numpy())
    one_step_norm = np.concatenate(one_step_parts).reshape(
        len(test_norm), count - 1, test_norm.shape[-1]
    )
    one_step = denorm(one_step_norm, mean, std)

    actual_one_step = truth[:, 1:].reshape(-1, truth.shape[-1])
    actual_rollout = truth.reshape(-1, truth.shape[-1])
    scale_one_step = np.std(actual_one_step, axis=0)
    scale_rollout = np.std(actual_rollout, axis=0)
    scale_one_step[scale_one_step < 1e-12] = 1.0
    scale_rollout[scale_rollout < 1e-12] = 1.0
    one_step_error = (
        np.sqrt(np.mean(
            (actual_one_step - one_step.reshape(-1, truth.shape[-1])) ** 2,
            axis=0,
        )) / scale_one_step
    )
    rollout_error = (
        np.sqrt(np.mean(
            (actual_rollout - prediction.reshape(-1, truth.shape[-1])) ** 2,
            axis=0,
        )) / scale_rollout
    )

    shown_count = (len(test_norm) if arrays.get("pam_plot_selected", False) else
                   min(max(int(plot_holdouts), 1), len(test_norm)))
    display_indices = np.linspace(0, len(test_norm) - 1, shown_count, dtype=int)
    test_time = np.asarray(arrays.get("pam_test_time"))
    if test_time.shape != test_norm.shape[:2]:
        dt = float(np.asarray(arrays.get("pam_dt", 1.0)))
        test_time = np.broadcast_to(
            np.arange(test_norm.shape[1], dtype=np.float64) * dt,
            test_norm.shape[:2],
        )
    behavior = str(np.asarray(arrays.get("pam_behavior", "unknown")).item())
    behavior_label = behavior.replace("_", " ").title()
    joint_colors = ("#365C8D", "#1FA187", "#F4C430", "#D1495B")

    def style_axis(axis) -> None:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#DCE3EA", linewidth=0.7, alpha=0.75)
        axis.tick_params(colors="#435160", labelsize=8)

    figure = plt.figure(figsize=(16, 12), facecolor="#F5F7FA")
    grid = figure.add_gridspec(
        3, 2, left=0.07, right=0.97, bottom=0.08, top=0.88,
        hspace=0.4, wspace=0.27,
    )
    ax_eigs = figure.add_subplot(grid[0, 0])
    ax_singular = figure.add_subplot(grid[0, 1])
    ax_position = figure.add_subplot(grid[1, 0])
    ax_velocity = figure.add_subplot(grid[1, 1], sharex=ax_position)
    ax_pressure = figure.add_subplot(grid[2, 0], sharex=ax_position)
    ax_error = figure.add_subplot(grid[2, 1])
    figure.text(
        0.07, 0.955, "ROBOT ARM  ·  KOOPMAN AUTOENCODER",
        fontsize=11, weight="bold", color="#4C78A8",
    )
    figure.text(
        0.07, 0.92, behavior_label,
        fontsize=21, weight="bold", color="#14213D",
    )
    figure.text(
        0.07, 0.89,
        f"state = [q, qdot, observed pressure] · {len(test_norm)} held-out intervals · "
        f"{count} rollout samples",
        fontsize=9, color="#64748B",
    )

    spectra = [np.linalg.eigvals(model.K.detach().cpu().float().numpy())]
    angle = np.linspace(0.0, 2.0 * np.pi, 300)
    ax_eigs.plot(
        np.cos(angle), np.sin(angle), linestyle="--",
        color="#94A3B8", linewidth=1.0,
    )
    ax_eigs.axhline(0.0, color="#CBD5E1", linewidth=0.7)
    ax_eigs.axvline(0.0, color="#CBD5E1", linewidth=0.7)
    for index, eigenvalues in enumerate(spectra):
        ax_eigs.scatter(
            eigenvalues.real, eigenvalues.imag, s=35,
            label=f"K[{index}]" if len(spectra) > 1 else "K",
        )
    eig_limit = max(
        1.1,
        max(float(np.max(np.abs(values))) for values in spectra) * 1.1,
    )
    ax_eigs.set(
        xlim=(-eig_limit, eig_limit), ylim=(-eig_limit, eig_limit),
        xlabel="Real", ylabel="Imaginary",
    )
    ax_eigs.set_aspect("equal", adjustable="box")
    ax_eigs.set_title("Learned Koopman eigenvalues", loc="left", weight="bold")
    if len(spectra) > 1:
        ax_eigs.legend(frameon=False, fontsize=7)
    style_axis(ax_eigs)

    train_states = np.asarray(arrays["train_sequences"]).reshape(-1, truth.shape[-1])
    singular_values = np.linalg.svd(train_states, full_matrices=False, compute_uv=False)
    energy = np.cumsum(singular_values ** 2) / max(np.sum(singular_values ** 2), 1e-15)
    components = np.arange(1, len(singular_values) + 1)
    ax_singular.semilogy(
        components, singular_values, marker="o", color="#4C78A8",
        linewidth=1.3, markersize=4,
    )
    ax_energy = ax_singular.twinx()
    ax_energy.plot(components, energy * 100.0, color="#F58518", linewidth=1.3)
    ax_energy.set_ylabel("Cumulative energy (%)", color="#F58518", fontsize=8)
    ax_energy.set_ylim(0.0, 105.0)
    ax_singular.set(xlabel="Component", ylabel="Singular value")
    ax_singular.set_title("Normalized training-state spectrum", loc="left", weight="bold")
    style_axis(ax_singular)

    for trajectory, index in enumerate(display_indices):
        time = test_time[index, :count]
        for joint, color in enumerate(joint_colors):
            measured_label = f"q{joint} measured" if trajectory == 0 else None
            ax_position.plot(
                time, truth[index, :, joint], color=color,
                linewidth=1.0, alpha=0.6, label=measured_label,
            )
            ax_position.plot(
                time, prediction[index, :, joint], color=color,
                linewidth=0.85, linestyle="--", alpha=0.7,
            )
            ax_velocity.plot(
                time, truth[index, :, 4 + joint], color=color,
                linewidth=0.9, alpha=0.6,
            )
            ax_velocity.plot(
                time, prediction[index, :, 4 + joint], color=color,
                linewidth=0.75, linestyle="--", alpha=0.7,
            )
            ago, antago = 8 + 2 * joint, 9 + 2 * joint
            pressure_label = f"Joint {joint}" if trajectory == 0 else None
            ax_pressure.plot(
                time, truth[index, :, ago] - truth[index, :, antago],
                color=color, linewidth=0.9, alpha=0.6,
                label=pressure_label,
            )
            ax_pressure.plot(
                time, prediction[index, :, ago] - prediction[index, :, antago],
                color=color, linewidth=0.75, linestyle="--", alpha=0.7,
            )
    ax_position.set_title(
        f"Held-out positions · {shown_count} rollouts", loc="left", weight="bold"
    )
    ax_velocity.set_title(
        f"Held-out velocities · {shown_count} rollouts", loc="left", weight="bold"
    )
    ax_pressure.set_title(
        f"Held-out differential pressures · {shown_count} rollouts",
        loc="left", weight="bold",
    )
    ax_position.set_ylabel("Position")
    ax_velocity.set(xlabel="Time in held-out interval (s)", ylabel="Velocity")
    ax_pressure.set(
        xlabel="Time in held-out interval (s)",
        ylabel="Agonist - antagonist pressure",
    )
    ax_position.legend(frameon=False, fontsize=7, ncol=2)
    ax_pressure.legend(frameon=False, fontsize=7, ncol=2)
    for axis in (ax_position, ax_velocity, ax_pressure):
        style_axis(axis)

    feature_indices = np.arange(truth.shape[-1])
    width = 0.38
    ax_error.bar(
        feature_indices - width / 2, one_step_error, width,
        color="#4C78A8", label="One step",
    )
    ax_error.bar(
        feature_indices + width / 2, rollout_error, width,
        color="#F58518", label="Rollout",
    )
    labels = [f"q{i}" for i in range(4)] + [f"dq{i}" for i in range(4)] + [
        f"p{i}{side}" for i in range(4) for side in ("+", "-")
    ]
    ax_error.set_xticks(feature_indices, labels, rotation=55, ha="right")
    ax_error.set_ylabel("NRMSE")
    ax_error.set_title(
        f"Held-out prediction error · mean nDTW {np.mean(rollout_ndtw):.3f}",
        loc="left", weight="bold",
    )
    ax_error.legend(frameon=False, fontsize=8)
    style_axis(ax_error)
    figure.text(
        0.97, 0.025, "Solid: measured · dashed: autonomous model rollout",
        ha="right", fontsize=8, color="#7B8794",
    )
    summary_path = outdir / "01_robot_arm_diagnostics.png"
    figure.savefig(
        summary_path, dpi=170, facecolor=figure.get_facecolor(),
        bbox_inches="tight",
    )
    plt.close(figure)

    behavior_labels = _pam_test_behavior_labels(arrays, len(test_norm))
    rollout_plots = {}
    titles = ("Joint positions", "Joint velocities", "Differential pressures")
    ylabels = ("Position", "Velocity", "Agonist - antagonist")
    for behavior_name in dict.fromkeys(behavior_labels.tolist()):
        candidates = np.flatnonzero(behavior_labels == behavior_name)
        shown = min(max(int(plot_holdouts), 1), len(candidates))
        selected = candidates[np.linspace(0, len(candidates)-1, shown, dtype=int)]
        truth_figure, truth_axes = plt.subplots(
            3, 1, figsize=(11, 9), sharex=True, facecolor="#F5F7FA"
        )
        model_figure, model_axes = plt.subplots(
            3, 1, figsize=(11, 9), sharex=True, facecolor="#F5F7FA"
        )
        for index in selected:
            time = test_time[index, :count]
            for joint, color in enumerate(joint_colors):
                truth_axes[0].plot(time, truth[index, :, joint], color=color, alpha=0.65)
                model_axes[0].plot(time, prediction[index, :, joint], color=color, alpha=0.65)
                truth_axes[1].plot(time, truth[index, :, 4+joint], color=color, alpha=0.65)
                model_axes[1].plot(time, prediction[index, :, 4+joint], color=color, alpha=0.65)
                ago, antago = 8+2*joint, 9+2*joint
                truth_axes[2].plot(time, truth[index, :, ago]-truth[index, :, antago], color=color, alpha=0.65)
                model_axes[2].plot(time, prediction[index, :, ago]-prediction[index, :, antago], color=color, alpha=0.65)
        group_values = (
            (truth[selected, ..., :4], prediction[selected, ..., :4]),
            (truth[selected, ..., 4:8], prediction[selected, ..., 4:8]),
            (truth[selected, ..., 8::2]-truth[selected, ..., 9::2],
             prediction[selected, ..., 8::2]-prediction[selected, ..., 9::2]),
        )
        for row, (truth_axis, model_axis) in enumerate(zip(truth_axes, model_axes)):
            finite = np.concatenate([value.reshape(-1) for value in group_values[row]])
            finite = finite[np.isfinite(finite)]
            if len(finite):
                lower, upper = float(finite.min()), float(finite.max())
                margin = 0.04*max(upper-lower, 1e-12)
                truth_axis.set_ylim(lower-margin, upper+margin)
                model_axis.set_ylim(lower-margin, upper+margin)
            for axis in (truth_axis, model_axis):
                axis.set_title(titles[row], loc="left", weight="bold")
                axis.set_ylabel(ylabels[row]); style_axis(axis)
        truth_axes[-1].set_xlabel("Time in held-out interval (s)")
        model_axes[-1].set_xlabel("Time in held-out interval (s)")
        label = behavior_name.replace("_", " ").title()
        truth_figure.suptitle(f"Robot-arm ground-truth holdouts · {label}", weight="bold")
        model_figure.suptitle(f"Robot-arm autonomous model rollouts · {label}", weight="bold")
        suffix = behavior_name.replace("fixed_movement_", "").replace("_", "-")
        rollout_plots[behavior_name] = {
            "ground_truth": savefig(truth_figure, outdir/f"02_robot_arm_rollouts_ground_truth_{suffix}.png"),
            "model": savefig(model_figure, outdir/f"02_robot_arm_rollouts_model_{suffix}.png"),
            "trajectory_indices": selected.tolist(),
        }
    animations_by_behavior = {}
    if animate:
        for behavior_name, plots in rollout_plots.items():
            animation_index = int(plots["trajectory_indices"][0])
            animation_time = test_time[animation_index, :count]
            label = behavior_name.replace("_", " ").title()
            suffix = behavior_name.replace(
                "fixed_movement_", ""
            ).replace("_", "-")
            animations_by_behavior[behavior_name] = {
                "joint_motion_2d": _save_pam_joint_rollout_mp4(
                    truth[animation_index, :, :4],
                    prediction[animation_index, :, :4],
                    animation_time,
                    outdir / f"03_robot_arm_joint_rollout_{suffix}.mp4",
                    label,
                ),
                "kinematic_motion_3d": _save_pam_kinematic_rollout_mp4(
                    truth[animation_index, :, :4],
                    prediction[animation_index, :, :4],
                    animation_time,
                    outdir / f"04_robot_arm_kinematic_rollout_{suffix}.mp4",
                    label,
                ),
                "trajectory_index": animation_index,
            }
    return {
        "diagnostic_plot": str(summary_path),
        "rollout_plots_by_behavior": rollout_plots,
        "animations_by_behavior": animations_by_behavior,
        "behavior": behavior,
        "n_test_rollouts": int(len(test_norm)),
        "n_displayed_rollouts": int(shown_count),
        "rollout_points": int(count),
        "one_step_nrmse": one_step_error.tolist(),
        "rollout_nrmse": rollout_error.tolist(),
        "mean_one_step_nrmse": float(np.mean(one_step_error)),
        "mean_rollout_nrmse": float(np.mean(rollout_error)),
        "rollout_ndtw": rollout_ndtw.tolist(),
        "mean_rollout_ndtw": float(np.mean(rollout_ndtw)),
        "median_rollout_ndtw": float(np.median(rollout_ndtw)),
    }


@torch.no_grad()


def _draw_3d_trajectories(
    axis, trajectories, *, colors=None, phases=None, phase_norm=None,
    linewidth=1.0, alpha=1.0, mark_initial=False,
):
    """Shared renderer for solid, phase-colored, and subspace-RGB curves."""
    artist = None
    for index, trajectory in enumerate(trajectories):
        color = colors[index] if colors is not None else "0.72"
        if phases is not None or np.ndim(color) == 2:
            valid = np.isfinite(trajectory[:-1]).all(axis=1) & np.isfinite(trajectory[1:]).all(axis=1)
            if phases is not None:
                valid &= np.isfinite(phases[index][:-1])
            if not valid.any():
                continue
            segments = np.stack((trajectory[:-1], trajectory[1:]), axis=1)[valid]
            artist = Line3DCollection(segments, linewidths=linewidth, alpha=alpha)
            if phases is None:
                artist.set_colors((0.5 * (color[:-1] + color[1:]))[valid])
            else:
                artist.set_cmap("twilight")
                artist.set_norm(phase_norm)
                artist.set_array(phases[index][:-1][valid])
            axis.add_collection3d(artist)
        else:
            artist, = axis.plot(*trajectory.T, color=color, linewidth=linewidth, alpha=alpha)
            if mark_initial:
                axis.scatter(*trajectory[0], color=color, s=18)
    return artist


@torch.no_grad()
def plot_3d_system_diagnostics(
    model, arrays, mean, std, device, horizon, mode, outdir, system_name,
    *, max_points=None, dt=1.0, n_test_traj=None, animate=False,
) -> Dict[str, Any]:
    """Evaluate 3-D rollouts once; select the dataset's presentation and metadata."""
    is_goodwin = system_name in {"goodwin", "goodwin_oscillator", "goodwin-oscillator"}
    is_lorenz = system_name == "lorenz"
    sequences = _system_plot_sequences(arrays, system_name)
    max_points = sequences.shape[1] if max_points is None else max(int(max_points), 2)
    count = (min(int(horizon) + 1, sequences.shape[1]) if is_lorenz or is_goodwin
             else sequences.shape[1])
    if is_goodwin:
        count = min(count, max_points)
        if count < 2:
            return {}
    limit = (len(sequences) if n_test_traj is None else max(int(n_test_traj), 1)) if is_goodwin else (
        6 if is_lorenz else DYSTS_PLOT_HOLDOUTS
    )
    shown = len(sequences) if is_goodwin or canonical_system(system_name) in MULTISTABLE_3D else min(limit, len(sequences))
    truth_norm = sequences[:shown, :count]
    batch_size = shown if is_goodwin else 1
    predictions = []
    for start in range(0, shown, batch_size):
        initial = torch.from_numpy(truth_norm[start:start + batch_size, 0]).to(device)
        rollout = model_rollout(model, initial, count - 1, mode)["x_rollout"]
        predictions.append(rollout.detach().cpu().float().numpy())
    prediction_norm = np.concatenate(predictions)
    truth, prediction = (denorm(values, mean, std) for values in (truth_norm, prediction_norm))
    if is_goodwin:
        return _plot_goodwin_phase_diagnostics(
            model, arrays, mean, std, outdir, truth_norm, prediction_norm,
            truth, prediction, max_points, dt, animate,
        )
    display_name = ("Lorenz-63" if is_lorenz
                    else str(np.asarray(arrays.get("system_display_name", arrays.get("dysts_system", "3-D system"))).item()))
    figure = plt.figure(figsize=(12.0, 5.5))
    palette = plt.cm.viridis(np.linspace(0.08, 0.92, shown))
    for panel, (trajectories, title) in enumerate(
        ((truth, "Ground truth"), (prediction, "Koopman rollout")), start=1,
    ):
        axis = figure.add_subplot(1, 2, panel, projection="3d")
        _draw_3d_trajectories(axis, trajectories, colors=palette,
                              linewidth=1.0 if is_lorenz else 0.9, mark_initial=True)
        if not is_lorenz:
            _style_3d_axis(axis)
        axis.set(xlabel="x", ylabel="y", zlabel="z", title=title)
    figure.suptitle(f"{display_name} held-out trajectories", weight="bold")
    path = savefig(figure, outdir / f"01_{system_name}_3d_rollouts.png")
    report = {"plot": path, "n_test_rollouts": shown, "rollout_points": count}
    if is_lorenz:
        report.update({name: float(np.asarray(arrays[f"lorenz_{name}"]))
                       for name in ("sigma", "rho", "beta")})
    else:
        report["parameters"] = json.loads(str(np.asarray(
            arrays.get("system_parameters", arrays.get("dysts_parameters", "{}"))
        ).item()))
    if "stable_equilibria" in arrays:
        report["stable_equilibria"] = np.asarray(arrays["stable_equilibria"]).tolist()
    return report


@torch.no_grad()
def plot_lotka_volterra_diagnostics(model, arrays, mean, std, device, horizon, mode, outdir):
    """Compare held-out competitive population trajectories in the phase plane."""
    sequences = _heldout_plot_sequences(arrays)
    count = min(int(horizon) + 1, sequences.shape[1])
    prediction = model_rollout(model, torch.from_numpy(sequences[:, 0]).to(device),
                               count - 1, mode)["x_rollout"].cpu().numpy()
    truth, prediction = (denorm(x, mean, std) for x in (sequences[:, :count], prediction))
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    palette = plt.cm.viridis(np.linspace(0.08, 0.92, len(truth)))
    for axis, trajectories, title in zip(axes, (truth, prediction), ("Ground truth", "Koopman rollout")):
        for trajectory, color in zip(trajectories, palette):
            axis.plot(*trajectory.T, color=color, linewidth=1)
        axis.scatter([1, 0], [0, 1], marker="*", s=100, color="black", label="Attractors")
        axis.scatter([1/3], [1/3], marker="x", color="black", label="Saddle")
        axis.plot([0, 1.5], [0, 1.5], "--", color="0.7", linewidth=0.8)
        axis.set(xlabel="x population", ylabel="y population", title=title)
        axis.legend(frameon=False)
    return {"plot": savefig(figure, outdir / "01_lotka_volterra_phase_portraits.png"),
            "n_test_rollouts": len(truth), "rollout_points": count}


@torch.no_grad()
def _plot_lotka_volterra_grassmann(model, arrays, mean, std, device, outdir):
    """Color 2-D population model rollouts by their local selected subspaces."""
    sequences = _heldout_plot_sequences(arrays)
    was_training = model.training
    model.eval()
    try:
        sequences = _grassmann_plot_rollouts(model, sequences, device)
        parts = []
        flat = sequences.reshape(-1, 2)
        for start in range(0, len(flat), 4096):
            basis = model.predict_subspace(torch.from_numpy(flat[start:start + 4096]).to(device))
            parts.append((basis @ basis.transpose(-2, -1)).flatten(start_dim=-2).cpu().numpy())
        features = np.concatenate(parts)
        colors, _ = _apply_projector_color_embedding(
            features, _fit_projector_color_embedding(features, model.grassmann_rank),
        )
        figure, axis = plt.subplots(figsize=(7, 6))
        physical = denorm(sequences, mean, std)
        for trajectory, rgb in zip(physical, colors.reshape(*sequences.shape[:2], 3)):
            axis.add_collection(LineCollection(np.stack((trajectory[:-1], trajectory[1:]), axis=1),
                                               colors=0.5 * (rgb[:-1] + rgb[1:]), linewidths=1.5))
        axis.update_datalim(physical.reshape(-1, 2))
        axis.autoscale_view()
        axis.scatter([1, 0], [0, 1], marker="*", s=100, color="black")
        axis.set(xlabel="x population", ylabel="y population",
                 title="Lotka–Volterra trajectories colored by local Grassmann subspace")
        return {"rgb_subspace_plot": savefig(figure, outdir / "05c_lotka_volterra_state_conditioned_grassmann_rgb.png"),
                "rgb_trajectory_count": len(sequences)}
    finally:
        model.train(was_training)


def plot_duffing_diagnostics(
    model, arrays, mean, std, device, horizon, mode, outdir,
) -> Dict[str, Any]:
    """Plot held-out Duffing truth and autonomous rollouts in phase space."""
    sequences = np.asarray(arrays["test_sequences"], dtype=np.float32)
    count = min(int(horizon) + 1, sequences.shape[1])
    shown = min(12, len(sequences))
    predictions = []
    for sequence in sequences[:shown]:
        initial = torch.from_numpy(sequence[None, 0]).float().to(device)
        with torch.no_grad():
            rollout = model_rollout(model, initial, count - 1, mode)["x_rollout"]
        predictions.append(rollout[0].detach().cpu().float().numpy())
    truth = denorm(sequences[:shown, :count], mean, std)
    prediction = denorm(np.stack(predictions), mean, std)
    well = np.sqrt(
        float(np.asarray(arrays["duffing_linear_stiffness"]))
        / float(np.asarray(arrays["duffing_cubic_stiffness"]))
    )
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), sharex=True, sharey=True)
    true_well = truth[:, -1, 0] >= 0.0
    predicted_well = prediction[:, -1, 0] >= 0.0
    well_colors = {False: "#2563EB", True: "#DC2626"}
    well_labels = {
        False: "initial condition → left well in ground truth",
        True: "initial condition → right well in ground truth",
    }
    labeled = [{False: False, True: False}, {False: False, True: False}]
    for index in range(shown):
        well_key = bool(true_well[index])
        for axis_index, trajectory in enumerate((truth[index], prediction[index])):
            label = None
            if not labeled[axis_index][well_key]:
                label = well_labels[well_key]
                labeled[axis_index][well_key] = True
            axes[axis_index].plot(
                *trajectory.T, color=well_colors[well_key], linewidth=1.2,
                alpha=0.85, label=label,
            )
            axes[axis_index].scatter(
                trajectory[-1, 0], trajectory[-1, 1], s=22,
                facecolor=well_colors[well_key], edgecolor="white",
                linewidth=0.7, zorder=4,
            )
    for axis, title in zip(axes, ("Ground truth", "Koopman rollout")):
        axis.scatter([-well], [0.0], marker="*", s=85, color=well_colors[False])
        axis.scatter([well], [0.0], marker="*", s=85, color=well_colors[True])
        axis.scatter([0.0], [0.0], marker="x", s=45, color="#111827")
        axis.set(xlabel="position", ylabel="velocity", title=title)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle("Duffing double-well held-out trajectories", weight="bold")
    path = outdir / "01_duffing_phase_portraits.png"
    savefig(figure, path)
    return {
        "plot": str(path), "n_test_rollouts": shown, "rollout_points": count,
        "well_accuracy": float(np.mean(true_well == predicted_well)),
        "stable_well_positions": [-float(well), float(well)],
    }


def main() -> None:
    args = build_parser().parse_args()
    from src.utils import configured_device, load_system_config
    system_cfg = load_system_config(args.system_config)
    device = get_device(args.device or configured_device(system_cfg))
    checkpoint = auto_find_checkpoint(args.checkpoint, args.checkpoint_type)

    model, ckpt, model_cfg = load_checkpoint(checkpoint, device)
    system_hint = str((ckpt.get("data_config", {}) or {}).get("system", "")).lower()
    arrays = load_arrays(
        ckpt,
        reaching_seed_override=(
            0 if system_hint in {
                "monkey_reaching", "reaching", "comp_neurosci_reaching", "neural_reaching"
            } else None
        ),
    )
    mean = np.asarray(ckpt.get("mean", arrays["mean"]), dtype=np.float32)
    std = np.asarray(ckpt.get("std", arrays["std"]), dtype=np.float32)
    data_cfg = ckpt.get("data_config", {}) or {}
    dt = float(data_cfg.get("dt", 1.0))

    system = canonical_system(data_cfg.get("system", ""))
    is_goodwin = system in {"goodwin", "goodwin_oscillator", "goodwin-oscillator"}
    is_reaching = system in {"monkey_reaching", "reaching", "comp_neurosci_reaching", "neural_reaching"}
    is_robot_arm = system in {"pam", "pamy2", "pneumatic_robot_arm_pam"}
    is_lorenz = system in {"lorenz", "lorenz63", "lorenz_63", "lorenz-63"}
    is_duffing = system in {"duffing", "duffing_oscillator", "duffing-oscillator"}
    is_dysts_3d = system in DYSTS_3D_SYSTEMS
    is_ode_3d = is_dysts_3d or system in MULTISTABLE_3D
    is_lotka = system == "lotka_volterra"
    if not (is_goodwin or is_reaching or is_robot_arm
            or is_lorenz or is_duffing or is_ode_3d or is_lotka):
        raise ValueError(
            "Visualization supports Goodwin, Lorenz, Duffing, the configured "
            "Dysts systems, bistable Lorenz, Lotka–Volterra, "
            "reaching, and robot-arm data"
        )
    if is_goodwin:
        dt = float(np.asarray(arrays["goodwin_dt"]).item())
    elif is_reaching:
        dt = float(np.asarray(arrays["reaching_bin_width"]).item())
    elif is_robot_arm:
        dt = float(np.asarray(arrays.get("pam_dt", dt)))
    all_plot_holdouts = system in MULTISTABLE_3D or is_lotka or is_goodwin
    sequences = _system_plot_sequences(arrays, system)
    rollout_split = "validation" if is_goodwin else "test_validation" if all_plot_holdouts else "test"
    if is_reaching or all_plot_holdouts:
        trajectory_indices = list(range(len(sequences)))
    elif is_robot_arm:
        trajectory_indices = pamy_visualization_indices(arrays).tolist()
    elif is_ode_3d:
        trajectory_indices = list(range(min(
            DYSTS_GENERIC_ROLLOUTS, len(sequences)
        )))
    else:
        requested = max(1, int(data_cfg.get("n_test_traj", len(sequences))))
        trajectory_indices = list(range(min(requested, len(sequences))))

    outdir = Path(args.outdir) if args.outdir else checkpoint.parent / "diagnostics_latent"
    outdir.mkdir(parents=True, exist_ok=True)

    derived_max_points = int(sequences.shape[1])
    train_horizon = resolve_training_horizon(ckpt)
    if args.rollout_steps is not None:
        eval_horizon = int(args.rollout_steps)
    elif is_ode_3d:
        eval_horizon = min(
            derived_max_points - 1, DYSTS_DEFAULT_VISUALIZATION_STEPS
        )
    else:
        eval_horizon = derived_max_points - 1

    mode = str(
        args.rollout_mode or getattr(model, "rollout_mode", "latent")
    ).lower()
    if mode not in {"latent", "reencode"}:
        raise ValueError("rollout_mode must be latent or reencode")

    if args.reencode_every is not None:
        if int(args.reencode_every) < 1:
            raise ValueError("--reencode_every must be at least 1")
        model.reencode_every = int(args.reencode_every)

    summary: Dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "train_test_loss_plot": plot_training_loss_history(
            checkpoint, outdir
        ),
        "model_class": ckpt.get("model_class"),
        "model_config": model_cfg,
        "rollout_mode": mode,
        "reencode_every": int(getattr(model, "reencode_every", 1)),
        "training_horizon": train_horizon,
        "rollout_split": rollout_split,
        "trajectory_indices": trajectory_indices,
        "rollout_definition": (
            "strict latent recurrence; decode stored states afterward"
            if mode == "latent"
            else f"periodic decoder-encoder projection every {model.reencode_every} steps"
        ),
        "model_geometry": str(getattr(model, "model_type", "global_koopman")),
    }

    training_rollouts = []
    for display_index, trajectory_index in enumerate(trajectory_indices, start=1):
        seq_norm = sequences[trajectory_index]
        suffix = f"_{rollout_split}_{display_index:03d}"
        training_rollouts.append(run_one_horizon(
            model, seq_norm, mean, std, device, eval_horizon, mode,
            outdir, f"01_training_horizon{suffix}", dt, derived_max_points, data_cfg,
            plot_individual_diagnostics=False,
        ))

    summary["training_horizon_rollouts"] = training_rollouts
    if is_goodwin or is_lorenz or is_ode_3d:
        diagnostic_key = ("goodwin" if is_goodwin else "lorenz" if is_lorenz else system if system in MULTISTABLE_3D else "dysts")
        summary[f"{diagnostic_key}_diagnostics"] = plot_3d_system_diagnostics(
            model, arrays, mean, std, device, eval_horizon, mode, outdir,
            "goodwin" if is_goodwin else "lorenz" if is_lorenz else system.replace("-", "_"),
            max_points=derived_max_points, dt=dt,
            n_test_traj=max(1, int(data_cfg.get("n_test_traj", len(sequences)))),
            animate=args.animate,
        )
    if is_lotka:
        summary["lotka_volterra_diagnostics"] = plot_lotka_volterra_diagnostics(
            model, arrays, mean, std, device, eval_horizon, mode, outdir,
        )
    if is_duffing:
        summary["duffing_diagnostics"] = plot_duffing_diagnostics(
            model, arrays, mean, std, device, eval_horizon, mode, outdir,
        )
    if is_reaching:
        summary["reaching_diagnostics"] = plot_reaching_diagnostics(
            model, arrays, mean, std, device, eval_horizon, mode, outdir,
            derived_max_points, dt, args.animate,
            args.reaching_condition_average,
        )
    if is_robot_arm:
        summary["robot_arm_diagnostics"] = plot_robot_arm_diagnostics(
            model, arrays, mean, std, device, eval_horizon, mode, outdir,
            int(data_cfg.get("pam_plot_holdouts", 6)),
            args.animate,
        )
    if is_goodwin or is_robot_arm:
        summary["linearization_defect"] = plot_linearization_defect(
            model, arrays, mean, std, device, outdir, system,
            max_robot_trajectories=int(data_cfg.get("pam_plot_holdouts", 6)),
        )
    summary["spectrum"] = plot_spectrum(model, outdir / "08_koopman_spectrum.png")

    if (bool(getattr(model, "grassmann", False))
            and hasattr(model, "predict_subspace")):
        summary["state_conditioned_grassmann"] = plot_state_conditioned_grassmann(
            model, arrays, mean, std, device, outdir, system,
            animate=args.animate,
        )
    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved diagnostics to {outdir}")


if __name__ == "__main__":
    main()
