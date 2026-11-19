#!/usr/bin/env python3
"""Compare all fitted models for one dataset on held-out rollouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


SRC = Path(__file__).resolve().parent
PROJECT_ROOT = SRC.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils import (  # noqa: E402
    get_device,
    denorm,
    DEFAULT_SYSTEM_CONFIG,
    configured_device,
    load_system_config,
)
from utils import MULTISTABLE_NAMES as MULTISTABLE_SYSTEMS, canonical_system
from visualize import (  # noqa: E402
    _goodwin_nrmse,
    _normalized_dtw,
    load_arrays,
    load_checkpoint,
    model_rollout,
)

from utils import sequence_lengths


# Alignment-band half-width for every DTW/nDTW score, including horizon scores.
DTW_BAND_FRACTION = 0.01  # 0.1 = ±10%; 0.0 = diagonal-only for equal lengths.


def prediction_horizons(config: dict) -> list[float | str]:
    evaluation = config.get("evaluation", {})
    if not isinstance(evaluation, dict):
        raise ValueError("System config evaluation section must be a mapping")
    horizons = evaluation.get("prediction_horizons", [0.1, 0.25, 0.5, 0.75, "full"])
    if not isinstance(horizons, list) or not horizons:
        raise ValueError("evaluation.prediction_horizons must be a non-empty list")
    if any(not (value == "full" or (type(value) in (int, float) and 0 < value <= 1)) for value in horizons):
        raise ValueError("evaluation.prediction_horizons entries must be fractions in (0, 1] (0.1 = 10%) or 'full'")
    return list(dict.fromkeys(horizons))


def rollout_horizon_metrics(row, sequences, prediction_norm, truth, prediction, arrays, horizons):
    """Score prefixes specified as fractions of the dataset's average sample length."""
    system = row["system"]
    is_reaching = system in {"monkey_reaching", "reaching", "comp_neurosci_reaching", "neural_reaching"}
    groups = {"all": slice(None)}
    if is_reaching:
        neural_count = int(np.asarray(arrays["reaching_neural_unit_count"]).item())
        groups = {"combined": slice(0, neural_count + 4)}
    full_steps = sequences.shape[1] - 1
    average_length = float(row["average_trajectory_length"])
    # Reuse full-trajectory scores and repeated capped horizons, especially for DTW.
    cache = {full_steps: {
        name: {metric: row[(name + "_" if is_reaching else "") + metric]
               for metric in ("nrmse", "mean_dtw", "mean_ndtw")}
        for name in groups
    }}
    reports = {}
    for horizon in horizons:
        steps = full_steps if horizon == "full" else min(max(1, int(horizon * average_length)), full_steps)
        if steps not in cache:
            window = slice(0, steps + 1)
            metrics = {}
            for name, group in groups.items():
                normalized_truth = sequences[:, window, group]
                normalized_prediction = prediction_norm[:, window, group]
                physical_truth = truth[:, window, group]
                physical_prediction = prediction[:, window, group]
                metrics[name] = {
                    "nrmse": _goodwin_nrmse(
                        physical_truth,
                        physical_prediction,
                    ),
                    "mean_dtw": float(np.mean(_rollout_raw_dtw(physical_truth, physical_prediction))),
                    "mean_ndtw": float(np.mean(_rollout_ndtw(normalized_truth, normalized_prediction))),
                }
            cache[steps] = metrics
        label = "full" if horizon == "full" else f"{100 * horizon:g}%"
        reports[label] = {"steps": steps, "groups": cache[steps]}
    return reports


def fitted_datasets(
    outputs: Path, checkpoint_type: str = "best",
) -> dict[str, list[Path]]:
    """Return datasets and their selected best or latest checkpoints."""
    if checkpoint_type not in {"best", "last"}:
        raise ValueError("checkpoint_type must be 'best' or 'last'")
    found: dict[str, list[Path]] = {}
    if not outputs.is_dir():
        return found
    for dataset_dir in sorted(path for path in outputs.iterdir() if path.is_dir()):
        if checkpoint_type == "best":
            checkpoints = sorted(dataset_dir.glob("*/best.pt"))
        else:
            checkpoints = []
            for run_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
                latest = run_dir / "latest.pt"
                if latest.is_file():
                    checkpoints.append(latest)
                    continue
                snapshots = list(run_dir.glob("step_*.pt")) or list(run_dir.glob("epoch_*.pt"))
                if snapshots:
                    checkpoints.append(max(snapshots, key=lambda path: int(path.stem.split("_", 1)[1])))
        if checkpoints:
            found[dataset_dir.name] = checkpoints
    return found


def choose_dataset(available: dict[str, list[Path]]) -> str:
    names = sorted(available)
    if not names:
        raise FileNotFoundError("No fitted models found below outputs/")
    print("Available datasets:")
    for index, name in enumerate(names, start=1):
        print(f"  [{index}] {name} ({len(available[name])} models)")
    if not sys.stdin.isatty():
        raise RuntimeError("Interactive selection needs a terminal; use --dataset NAME")
    while True:
        answer = input("Dataset: ").strip()
        if answer in available:
            return answer
        try:
            selected = int(answer)
        except ValueError:
            selected = 0
        if 1 <= selected <= len(names):
            return names[selected - 1]
        print(f"Choose 1-{len(names)} or enter a dataset name.")


def dataset_shape_summary(arrays: dict[str, np.ndarray]) -> tuple[int, float]:
    """Return state dimension and trajectory-weighted mean length in samples."""
    sequence_splits = []
    for key in ("train_sequences", "val_sequences", "test_sequences"):
        if key not in arrays:
            continue
        split = np.asarray(arrays[key])
        if split.ndim != 3:
            raise ValueError(f"{key} must have shape [trajectory, time, state]")
        sequence_splits.append(split)
    if not sequence_splits:
        raise ValueError("dataset contains no trajectory sequence splits")

    state_dims = {int(split.shape[-1]) for split in sequence_splits}
    if len(state_dims) != 1:
        raise ValueError("dataset sequence splits have inconsistent state dimensions")
    trajectory_count = sum(int(split.shape[0]) for split in sequence_splits)
    if trajectory_count == 0:
        raise ValueError("dataset contains no trajectories")
    total_samples = sum(
        int(sequence_lengths(split).sum()) for split in sequence_splits
    )
    return state_dims.pop(), total_samples / trajectory_count


def model_parameter_counts(model: torch.nn.Module) -> tuple[int, int]:
    """Return parameter counts excluding and including the basis predictor."""
    combined = sum(parameter.numel() for parameter in model.parameters())
    subspace_net = getattr(model, "subspace_net", None)
    auxiliary = (
        sum(parameter.numel() for parameter in subspace_net.parameters())
        if isinstance(subspace_net, torch.nn.Module) else 0
    )
    return combined - auxiliary, combined


def _raw_dtw(
    reference: np.ndarray, estimate: np.ndarray,
    max_samples: int = 1000, band_fraction: float | None = None,
) -> float:
    """Accumulated multivariate DTW cost without path-length normalization."""
    if band_fraction is None:
        band_fraction = DTW_BAND_FRACTION
    stride = max(1, int(np.ceil(max(len(reference), len(estimate)) / max_samples)))
    reference = np.asarray(reference[::stride], dtype=np.float64)
    estimate = np.asarray(estimate[::stride], dtype=np.float64)
    n, m = len(reference), len(estimate)
    band = max(abs(n - m), int(np.ceil(max(n, m) * band_fraction)))
    previous = np.full(m + 1, np.inf)
    previous[0] = 0.0
    for i in range(1, n + 1):
        current = np.full(m + 1, np.inf)
        for j in range(max(1, i - band), min(m, i + band) + 1):
            distance = float(np.sqrt(np.mean(
                (reference[i - 1] - estimate[j - 1]) ** 2
            )))
            current[j] = distance + min(
                previous[j], current[j - 1], previous[j - 1]
            )
        previous = current
    return float(previous[m])


def _rollout_raw_dtw(reference: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    return np.asarray([
        _raw_dtw(truth[valid], prediction[valid])
        for truth, prediction in zip(reference, estimate)
        for valid in [np.isfinite(truth).all(axis=-1)]
    ])


def _rollout_ndtw(reference: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    return np.asarray([
        _normalized_dtw(truth[valid], prediction[valid], band_fraction=DTW_BAND_FRACTION)
        for truth, prediction in zip(reference, estimate)
        for valid in [np.isfinite(truth).all(axis=-1)]
    ])


def _rmse(reference: np.ndarray, estimate: np.ndarray) -> float:
    valid = np.isfinite(reference).all(axis=-1)
    return float(np.sqrt(np.mean((reference[valid].astype(np.float64) - estimate[valid].astype(np.float64)) ** 2)))


@torch.no_grad()
def state_conditioned_grassmann_diagnostics(model, sequences, device, max_samples=64):
    """Gauge-invariant representation diagnostics on held-out states."""
    flat = np.asarray(sequences, dtype=np.float32).reshape(-1, sequences.shape[-1])
    flat = flat[np.isfinite(flat).all(axis=-1)]
    indices = np.linspace(0, len(flat)-1, min(int(max_samples), len(flat)), dtype=int)
    states = torch.from_numpy(flat[indices]).to(device)
    psi = model.ambient_dictionary(states)
    basis = model.predict_subspace(states)
    gram = basis.transpose(-2, -1) @ basis
    identity = torch.eye(model.grassmann_rank, device=device)
    residual = torch.linalg.matrix_norm(gram-identity, ord="fro", dim=(-2,-1))
    projected = torch.einsum("...dr,...r->...d", basis,
        torch.einsum("...dr,...d->...r", basis, psi))
    energy = projected.square().sum(-1) / psi.square().sum(-1).clamp_min(1e-12)
    if len(basis) > 1:
        overlap = basis[:-1].transpose(-2,-1) @ basis[1:]
        chord = (model.grassmann_rank-overlap.square().sum((-2,-1))).clamp_min(0).sqrt()
    else:
        chord = residual.new_zeros(1)
    return {
        "grassmann_orthogonality_mean": float(residual.mean()),
        "grassmann_orthogonality_max": float(residual.max()),
        "grassmann_neighbor_chordal_mean": float(chord.mean()),
        "grassmann_projected_energy_mean": float(energy.mean()),
    }


@torch.no_grad()
def evaluate(
    checkpoint: Path,
    device: torch.device,
    arrays_cache: dict[str, dict[str, np.ndarray]] | None = None,
    horizons: list[float | str] | None = None,
) -> dict[str, object]:
    model, ckpt, _ = load_checkpoint(checkpoint, device)
    data_config = ckpt.get("data_config", {}) or {}
    cache_key = json.dumps(data_config, sort_keys=True, default=str)
    if arrays_cache is not None and cache_key in arrays_cache:
        arrays = arrays_cache[cache_key]
    else:
        arrays = load_arrays(ckpt)
        if arrays_cache is not None:
            arrays_cache[cache_key] = arrays
    split = "val" if len(arrays.get("val_sequences", ())) else "test"
    sequences = np.asarray(arrays[f"{split}_sequences"], dtype=np.float32)
    if len(sequences) == 0 or sequences.shape[1] < 2:
        raise ValueError(f"{split} split contains no usable rollout")

    mode = str(getattr(model, "rollout_mode", "latent")).lower()
    initial = torch.from_numpy(sequences[:, 0]).float().to(device)
    prediction_norm = model_rollout(
        model, initial, sequences.shape[1] - 1, mode,
    )["x_rollout"].detach().cpu().float().numpy()

    mean = np.asarray(ckpt.get("mean", arrays["mean"]), dtype=np.float32)
    std = np.asarray(ckpt.get("std", arrays["std"]), dtype=np.float32)
    truth = denorm(sequences, mean, std)
    prediction = denorm(prediction_norm, mean, std)
    state_dim, average_trajectory_length = dataset_shape_summary(arrays)
    model_size, combined_model_size = model_parameter_counts(model)
    row: dict[str, object] = {
        "model": checkpoint.parent.name,
        "model_size": model_size,
        "combined_model_size": combined_model_size,
        "state_dim": state_dim,
        "average_trajectory_length": average_trajectory_length,
        "trajectories": len(sequences),
        "split": split,
        "steps": sequences.shape[1] - 1,
        "mode": mode,
        "reencode_every": int(model.reencode_every),
        "grassmann": bool(getattr(model, "grassmann", False)),
        "model_class": model.__class__.__name__,
    }
    system = str(data_config.get("system", "")).lower()
    row["system"] = system
    if (bool(getattr(model, "grassmann", False))
            and model.__class__.__name__ == "KoopmanAutoencoder"
            and hasattr(model, "predict_subspace")):
        row.update(state_conditioned_grassmann_diagnostics(
            model, sequences, device
        ))
    if system in {"monkey_reaching", "reaching", "comp_neurosci_reaching", "neural_reaching"}:
        neural_count = int(np.asarray(arrays["reaching_neural_unit_count"]).item())
        groups = {
            "combined": slice(0, neural_count + 4),
        }
        for name, group in groups.items():
            group_ndtw = _rollout_ndtw(
                sequences[..., group], prediction_norm[..., group]
            )
            group_dtw = _rollout_raw_dtw(
                truth[..., group], prediction[..., group]
            )
            row[f"{name}_nrmse"] = _goodwin_nrmse(
                truth[..., group], prediction[..., group]
            )
            row[f"{name}_mean_ndtw"] = float(np.mean(group_ndtw))
            row[f"{name}_rmse"] = _rmse(
                truth[..., group], prediction[..., group]
            )
            row[f"{name}_mean_dtw"] = float(np.mean(group_dtw))
    else:
        ndtw = _rollout_ndtw(sequences, prediction_norm)
        dtw = _rollout_raw_dtw(truth, prediction)
        row.update(
            nrmse=_goodwin_nrmse(truth, prediction),
            mean_ndtw=float(np.mean(ndtw)),
            median_ndtw=float(np.median(ndtw)),
            rmse=_rmse(truth, prediction),
            mean_dtw=float(np.mean(dtw)),
        )
        if canonical_system(system) in MULTISTABLE_SYSTEMS:
            attractors = np.asarray(arrays["stable_equilibria"])
            true_basin = np.linalg.norm(truth[:, -1, None] - attractors, axis=-1).argmin(axis=1)
            predicted_basin = np.linalg.norm(prediction[:, -1, None] - attractors, axis=-1).argmin(axis=1)
            row["basin_accuracy"] = float(np.mean(true_basin == predicted_basin))
            row["worst_basin_nrmse"] = max(
                _goodwin_nrmse(truth[true_basin == basin], prediction[true_basin == basin])
                for basin in np.unique(true_basin)
            )
            row["final_nrmse"] = _goodwin_nrmse(truth[:, -1:], prediction[:, -1:])
        if system in {"lorenz", "lorenz63", "lorenz_63", "lorenz-63"}:
            row["final_nrmse"] = _goodwin_nrmse(
                truth[:, -1:], prediction[:, -1:]
            )
        if system in {"duffing", "duffing_oscillator", "duffing-oscillator"}:
            true_well = truth[:, -1, 0] >= 0.0
            predicted_well = prediction[:, -1, 0] >= 0.0
            well_errors = [
                _goodwin_nrmse(truth[true_well == well], prediction[true_well == well])
                for well in (False, True) if np.any(true_well == well)
            ]
            row["worst_well_nrmse"] = float(max(well_errors))
            row["well_accuracy"] = float(np.mean(predicted_well == true_well))
    if horizons is not None:
        row["horizon_metrics"] = rollout_horizon_metrics(
            row, sequences, prediction_norm, truth, prediction, arrays, horizons,
        )
    return row


def _format_model_size(value: object) -> str:
    count = int(value)
    if count >= 1_000_000:
        return f"{count / 1_000_000:.2f}M"
    if count >= 1_000:
        return f"{count / 1_000:.1f}K"
    return str(count)


def _format_average_length(value: object) -> str:
    length = float(value)
    if length.is_integer():
        return str(int(length))
    return f"{length:.1f}"


def add_relative_improvements(
    rows: list[dict[str, object]], *, reaching: bool,
) -> None:
    """Pair named Grassmann variants with their non-Grassmann base runs."""
    by_name = {str(row["model"]): row for row in rows}
    rmse_key = "combined_rmse" if reaching else "rmse"
    dtw_key = "combined_mean_dtw" if reaching else "mean_dtw"
    for row in rows:
        row["relative_improvement"] = None
        if not bool(row.get("grassmann", False)):
            continue
        name = str(row["model"])
        marker = "_grassmann_true"
        if marker not in name:
            continue
        base_name = name.split(marker, 1)[0] + "_grassmann_false"
        base = by_name.get(base_name)
        if base is None:
            continue
        base_rmse = float(base[rmse_key])
        base_dtw = float(base[dtw_key])
        row["relative_improvement"] = (
            100.0 * (base_rmse - float(row[rmse_key])) / max(abs(base_rmse), 1e-12),
            100.0 * (base_dtw - float(row[dtw_key])) / max(abs(base_dtw), 1e-12),
        )


def _format_relative_improvement(value: object) -> str:
    if value is None:
        return "—"
    rmse, dtw = value
    return f"{float(rmse):+.1f}% / {float(dtw):+.1f}%"


def horizon_advantages(row, baseline, horizon, group):
    """Percentage error reduction against the matching baseline; positive is better."""
    if baseline is None or row.get("split") != baseline.get("split"):
        return {}
    report = row["horizon_metrics"][horizon]
    reference = baseline.get("horizon_metrics", {}).get(horizon)
    if reference is None or report["steps"] != reference["steps"] or group not in reference["groups"]:
        return {}
    improvements = {}
    for metric in ("nrmse", "mean_dtw", "mean_ndtw"):
        base_value = float(reference["groups"][group][metric])
        value = float(report["groups"][group][metric])
        if np.isfinite(base_value) and np.isfinite(value) and base_value > 0:
            improvements[metric] = 100 * (base_value - value) / base_value
    return improvements


def print_horizon_table(dataset, rows):
    headers = ("model", "split", "horizon", "steps", "group", "avg traj len", "NRMSE", "mean DTW", "mean nDTW",
               "adv. NRMSE", "adv. DTW", "adv. nDTW")
    by_name = {str(row["model"]): row for row in rows}
    rendered = []
    for row in sorted(rows, key=lambda row: str(row["model"])):
        name = str(row["model"])
        baseline = None
        if row.get("grassmann") and "_grassmann_true" in name:
            base_name = name.split("_grassmann_true", 1)[0] + "_grassmann_false"
            baseline = by_name.get(base_name)
        for horizon, report in row["horizon_metrics"].items():
            for group, metrics in report["groups"].items():
                advantages = horizon_advantages(row, baseline, horizon, group)
                rendered.append((
                    name, str(row["split"]), horizon, str(report["steps"]), group,
                    _format_average_length(row["average_trajectory_length"]),
                    *(f"{metrics[key]:.5g}" for key in ("nrmse", "mean_dtw", "mean_ndtw")),
                    *(f"{advantages[key]:+.1f}%" if key in advantages else "—"
                      for key in ("nrmse", "mean_dtw", "mean_ndtw")),
                ))
    widths = [max(len(header), *(len(row[index]) for row in rendered))
              for index, header in enumerate(headers)]

    def line(values):
        return "  ".join(value.ljust(width) if index in (0, 1, 4) else value.rjust(width)
                         for index, (value, width) in enumerate(zip(values, widths)))

    print(f"\n{dataset} held-out autonomous rollouts by prediction horizon")
    print(line(headers))
    print(line(tuple("-" * width for width in widths)))
    for values in rendered:
        print(line(values))
    print("Horizon = percentage of average trajectory length in samples (all splits); steps round down, minimum 1, capped at available prediction steps.")
    print("Advantage = 100 × (baseline − Grassmann) / baseline; positive is better.")
    print("—: baseline row, no matching baseline/split/length, or zero/nonfinite baseline error.")


def print_table(dataset: str, rows: list[dict[str, object]]) -> None:
    if rows and all("horizon_metrics" in row for row in rows):
        print_horizon_table(dataset, rows)
        return
    systems = {str(row.get("system", "")).lower() for row in rows}
    is_reaching = bool(systems & {"monkey_reaching", "reaching", "comp_neurosci_reaching", "neural_reaching"})
    is_lorenz = bool(systems & {"lorenz", "lorenz63", "lorenz_63", "lorenz-63"})
    is_duffing = bool(systems & {"duffing", "duffing_oscillator", "duffing-oscillator"})
    add_relative_improvements(rows, reaching=is_reaching)
    if any(canonical_system(system) in MULTISTABLE_SYSTEMS for system in systems):
        headers = ("model", "NRMSE", "final NRMSE", "worst basin NRMSE", "basin accuracy",
                   "mean nDTW", "median nDTW", "rel. improvement RMSE / DTW", "trajs", "steps", "mode")
        rendered = [(
            str(row["model"]), f"{row['nrmse']:.5g}", f"{row['final_nrmse']:.5g}",
            f"{row['worst_basin_nrmse']:.5g}", f"{100 * row['basin_accuracy']:.1f}%",
            f"{row['mean_ndtw']:.5g}", f"{row['median_ndtw']:.5g}",
            _format_relative_improvement(row["relative_improvement"]),
            str(row["trajectories"]), str(row["steps"]), str(row["mode"]),
        ) for row in sorted(rows, key=lambda row: row["nrmse"])]
    elif is_reaching:
        headers = (
            "model", "size", "size + aux", "state dim", "avg traj len", "combined NRMSE", "combined nDTW",
            "rel. improvement RMSE / DTW",
            "trajs", "steps", "mode",
        )
        rendered = [
            (
                str(row["model"]), _format_model_size(row["model_size"]),
                _format_model_size(row["combined_model_size"]),
                str(row["state_dim"]),
                _format_average_length(row["average_trajectory_length"]),
                f"{float(row['combined_nrmse']):.5g}",
                f"{float(row['combined_mean_ndtw']):.5g}",
                _format_relative_improvement(row["relative_improvement"]),
                str(row["trajectories"]), str(row["steps"]), str(row["mode"]),
            )
            for row in sorted(rows, key=lambda item: float(item["combined_nrmse"]))
        ]
    elif is_lorenz:
        headers = (
            "model", "size", "size + aux", "state dim", "avg traj len",
            "NRMSE", "final NRMSE", "mean nDTW", "median nDTW",
            "rel. improvement RMSE / DTW", "trajs", "steps", "mode",
        )
        rendered = [
            (
                str(row["model"]), _format_model_size(row["model_size"]),
                _format_model_size(row["combined_model_size"]),
                str(row["state_dim"]),
                _format_average_length(row["average_trajectory_length"]),
                f"{float(row['nrmse']):.5g}",
                f"{float(row['final_nrmse']):.5g}",
                f"{float(row['mean_ndtw']):.5g}",
                f"{float(row['median_ndtw']):.5g}",
                _format_relative_improvement(row["relative_improvement"]),
                str(row["trajectories"]), str(row["steps"]), str(row["mode"]),
            )
            for row in sorted(rows, key=lambda item: float(item["nrmse"]))
        ]
    elif is_duffing:
        headers = (
            "model", "size", "size + aux", "state dim", "avg traj len",
            "NRMSE", "worst-well NRMSE", "well accuracy", "mean nDTW",
            "rel. improvement RMSE / DTW", "trajs", "steps", "mode",
        )
        rendered = [
            (
                str(row["model"]), _format_model_size(row["model_size"]),
                _format_model_size(row["combined_model_size"]), str(row["state_dim"]),
                _format_average_length(row["average_trajectory_length"]),
                f"{float(row['nrmse']):.5g}", f"{float(row['worst_well_nrmse']):.5g}",
                f"{100.0*float(row['well_accuracy']):.1f}%",
                f"{float(row['mean_ndtw']):.5g}",
                _format_relative_improvement(row["relative_improvement"]),
                str(row["trajectories"]), str(row["steps"]), str(row["mode"]),
            )
            for row in sorted(rows, key=lambda item: float(item["nrmse"]))
        ]
    else:
        headers = (
            "model", "size", "size + aux", "state dim", "avg traj len", "NRMSE",
            "mean nDTW", "median nDTW", "rel. improvement RMSE / DTW",
            "trajs", "steps", "mode",
        )
        rendered = [
            (
                str(row["model"]), _format_model_size(row["model_size"]),
                _format_model_size(row["combined_model_size"]),
                str(row["state_dim"]),
                _format_average_length(row["average_trajectory_length"]),
                f"{float(row['nrmse']):.5g}",
                f"{float(row['mean_ndtw']):.5g}",
                f"{float(row['median_ndtw']):.5g}",
                _format_relative_improvement(row["relative_improvement"]),
                str(row["trajectories"]), str(row["steps"]), str(row["mode"]),
            )
            for row in sorted(rows, key=lambda item: float(item["nrmse"]))
        ]
    widths = [
        max(len(headers[column]), *(len(row[column]) for row in rendered))
        for column in range(len(headers))
    ]

    def line(values: tuple[str, ...]) -> str:
        cells = []
        for index, value in enumerate(values):
            cells.append(
                value.ljust(widths[index])
                if index in {0, len(values) - 1}
                else value.rjust(widths[index])
            )
        return "  ".join(cells)

    print(f"\n{dataset} full-trajectory held-out autonomous rollouts")
    print(line(headers))
    print(line(tuple("-" * width for width in widths)))
    for row in rendered:
        print(line(row))

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize validation rollout metrics (test fallback) for all fitted models."
    )
    parser.add_argument("--dataset", help="Skip the interactive dataset prompt")
    parser.add_argument("--outputs", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument(
        "--system-config", type=Path, default=DEFAULT_SYSTEM_CONFIG,
        help="Shared runtime configuration (default: src/configs/system_config.yaml).",
    )
    parser.add_argument(
        "--device", default=None,
        choices=("auto", "cpu", "cuda", "cuda:0", "cuda:1", "gpu_0", "gpu_1"),
        help="Override compute.device from the system config.",
    )
    parser.add_argument(
        "--checkpoint_type", choices=("best", "last"), default="best",
        help=(
            "Checkpoint to summarize for every run: best.pt (default), or "
            "latest.pt with step_*.pt (or legacy epoch_*.pt) fallback."
        ),
    )
    args = parser.parse_args()

    available = fitted_datasets(args.outputs, args.checkpoint_type)
    dataset = args.dataset or choose_dataset(available)
    if dataset not in available:
        choices = ", ".join(sorted(available)) or "none"
        raise SystemExit(f"No fitted dataset {dataset!r}; available: {choices}")

    system_cfg = load_system_config(args.system_config)
    horizons = prediction_horizons(system_cfg)
    device = get_device(args.device or configured_device(system_cfg))
    arrays_cache: dict[str, dict[str, np.ndarray]] = {}
    rows = []
    failures = []
    for checkpoint in available[dataset]:
        try:
            rows.append(evaluate(checkpoint, device, arrays_cache, horizons=horizons))
        except Exception as error:  # keep the comparison useful if one fit is stale
            failures.append((checkpoint.parent.name, str(error)))
    if not rows:
        details = "; ".join(f"{name}: {error}" for name, error in failures)
        raise SystemExit(f"No models could be evaluated. {details}")

    print_table(dataset, rows)
    for name, error in failures:
        print(f"warning: skipped {name}: {error}", file=sys.stderr)


if __name__ == "__main__":
    main()
