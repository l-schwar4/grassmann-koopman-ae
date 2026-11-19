"""Train the supported Koopman autoencoder models from YAML configs."""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import fcntl
import shlex
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
from torch import nn
import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import (  # noqa: E402
    get_device,
    make_grad_scaler,
    save_json,
    set_seed,
    unwrap_model,
    WindowDataset,
    make_data,
    make_loader,
    reencode_candidates,
    DEFAULT_SYSTEM_CONFIG,
    configured_device,
    load_system_config,
)

from utils import sequence_lengths, complete_trajectory_summary
from torch.utils.data import Dataset


# -----------------------------------------------------------------------------
# Config / dynamic import
# -----------------------------------------------------------------------------


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping")
    return data


def atomic_torch_save(payload: Dict[str, Any], path: str | Path) -> None:
    """Write a checkpoint completely before making it visible to readers."""
    destination = Path(path)
    temporary = destination.with_name(f".{destination.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def import_module_from_file(path: str | Path) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    module_name = path.stem.replace("-", "_") + "_imported"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module



def merge_section(defaults: Dict[str, Any], section: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = dict(defaults)
    if section:
        out.update(section)
    return out


def maybe_override(config: Dict[str, Any], args: argparse.Namespace, key: str) -> None:
    value = getattr(args, key, None)
    if value is not None:
        config[key] = value


# -----------------------------------------------------------------------------
# Sequence training and evaluation
# -----------------------------------------------------------------------------


class FixedSequenceDataset(Dataset):
    """Already prepared uniform windows, stacked without padding or GPU masks."""

    def __init__(self, sequences):
        self.sequences = torch.from_numpy(sequences).float()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index]

    @staticmethod
    def collate_fn(items):
        return dict(sequence=torch.stack(items), dense_all_starts=True)

def training_dataset(arrays, split, horizon, legacy_class):
    if bool(arrays.get("fixed_sequence_batches", False)):
        return FixedSequenceDataset(arrays[f"{split}_sequences"])
    return legacy_class(arrays[f"{split}_sequences"], horizon)


def sequence_batch_loss(loss_fn, model, sequence, *, dense_all_starts=False, **kwargs):
    if dense_all_starts:
        return fixed_sequence_loss(model, sequence, **kwargs)
    return loss_fn(model, sequence, **kwargs)


def fixed_sequence_loss(
    model,
    sequence,
    *,
    lusch_alpha1=1.0,
    lusch_linf_weight=0.0,
    lusch_l2_weight=0.0,
    prediction_steps=None,
    linearity_steps=None,
    temporal_consistency_weight=0.0,
    state_mse_group_sizes=None,
    rollout_mode=None,
    **_
):
    """Bounded multistep loss from every valid start in equal-length sequences.

    Encode observations and construct Grassmann bases once per batch. Slice
    contiguous time ranges at each lag instead of indexing masked trajectories.
    """
    try:
        from .model.koopman_grassmann_ae import state_space_mse
    except ImportError:
        from model.koopman_grassmann_ae import state_space_mse
    max_horizon = sequence.shape[1] - 1
    sp = min(
        max_horizon, max_horizon if prediction_steps is None else int(prediction_steps)
    )
    sl = min(
        max_horizon, max_horizon if linearity_steps is None else int(linearity_steps)
    )
    if sp < 1 or sl < 1:
        raise ValueError("prediction_steps and linearity_steps must be at least one")
    horizon = max(sp, sl)
    mode = model.rollout_mode if rollout_mode is None else str(rollout_mode).lower()
    if mode not in {"latent", "reencode"}:
        raise ValueError("rollout_mode must be latent or reencode")
    interval = int(model.reencode_every)
    if interval < 1:
        raise ValueError("reencode_every must be at least one")
    subspace_interval = int(getattr(model, "subspace_update_every", -1))
    batch, time, state_dim = sequence.shape
    dictionary = model.ambient_dictionary(sequence.reshape(-1, state_dim)).reshape(
        batch, time, -1
    )
    initial_x = sequence[:, :-1]
    if model.grassmann:
        basis_states = sequence if float(temporal_consistency_weight) else initial_x
        bases = model.predict_subspace(basis_states.reshape(-1, state_dim))
        bases = bases.reshape(batch, basis_states.shape[1], *bases.shape[-2:])
        context = bases[:, : time - 1]
        current = model.project_with_basis(dictionary[:, :-1], context)
    else:
        current, context = (dictionary[:, :-1], None)
    reconstruction_state = model.decode(current)
    reconstruction = state_space_mse(
        reconstruction_state, initial_x, state_mse_group_sizes
    )
    zero = reconstruction.new_zeros(())
    prediction_sum, linearity_sum = (zero, zero)
    prediction_count = linearity_count = 0
    linf = (
        (reconstruction_state - initial_x).abs().max()
        if float(lusch_linf_weight)
        else zero
    )
    for lag in range(1, horizon + 1):
        available = time - lag
        current = current[:, :available]
        if context is not None:
            context = context[:, :available]
        predicted = model.apply_operator(current, model.K)
        target_dictionary = dictionary[:, lag:]
        target_state = sequence[:, lag:]
        count = batch * available
        if lag <= sl:
            target = (
                model.project_with_basis(target_dictionary, context)
                if context is not None
                else target_dictionary
            )
            linearity_sum = (
                linearity_sum + torch.nn.functional.mse_loss(predicted, target) * count
            )
            linearity_count += count
        reencode = mode == "reencode" and lag % interval == 0 and (lag < horizon)
        update_subspace = (
            model.grassmann
            and subspace_interval > 0
            and (lag % subspace_interval == 0)
            and (lag < horizon)
        )
        need_state = (
            lag <= sp
            or reencode
            or update_subspace
            or (lag == 1 and float(lusch_linf_weight))
        )
        state = model.decode(predicted) if need_state else None
        if lag <= sp:
            prediction_sum = (
                prediction_sum
                + state_space_mse(state, target_state, state_mse_group_sizes) * count
            )
            prediction_count += count
        if lag == 1 and float(lusch_linf_weight):
            linf = linf + (state - target_state).abs().max()
        if reencode:
            current = model.encode(state)
        elif update_subspace:
            current = model.project_dictionary(predicted, x=state)
        else:
            current = predicted
    prediction = prediction_sum / prediction_count
    linearity = linearity_sum / linearity_count
    matrices = (
        [
            p.square().sum()
            for p in model.parameters()
            if p.requires_grad and p.ndim >= 2
        ]
        if float(lusch_l2_weight)
        else []
    )
    l2 = torch.stack(matrices).sum() if float(lusch_l2_weight) and matrices else zero
    if model.grassmann and float(temporal_consistency_weight):
        projectors = bases @ bases.transpose(-2, -1)
        differences = projectors[:, 1:] - projectors[:, :-1]
        temporal = (
            torch.linalg.matrix_norm(differences, ord="fro", dim=(-2, -1)).mean()
            / 2.0**0.5
        )
    else:
        temporal = zero
    loss = (
        float(lusch_alpha1) * (reconstruction + prediction)
        + linearity
        + float(lusch_linf_weight) * linf
        + float(lusch_l2_weight) * l2
        + float(temporal_consistency_weight) * temporal
    )
    return dict(
        loss=loss,
        reconstruction=reconstruction,
        prediction=prediction,
        linearity=linearity,
        linf=linf,
        l2=l2,
        temporal_consistency=temporal,
    )

@torch.no_grad()
def evaluate_sequence(
    model: nn.Module,
    sequence_loss_fn: Callable[..., Dict[str, torch.Tensor]],
    dl,
    device: torch.device,
    loss_kwargs: Dict[str, Any],
    *,
    optimizer_step: int,
    rollout_mode: Optional[str],
) -> Dict[str, float]:
    model.eval()
    sums: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}
    for batch in dl:
        seq = batch["sequence"].to(device, non_blocking=True)
        terms = sequence_batch_loss(
            sequence_loss_fn, unwrap_model(model), seq,
            optimizer_step=int(optimizer_step), rollout_mode=rollout_mode,
            dense_all_starts=batch.get("dense_all_starts", False), **loss_kwargs,
        )
        bs = (seq.shape[0] * (seq.shape[1] - 1) if batch.get("dense_all_starts", False)
              else seq.shape[0])
        for k, v in terms.items():
            value = v.detach().mean() * bs
            sums[k] = value if k not in sums else sums[k] + value
            counts[k] = counts.get(k, 0) + bs
    return {k: float((sums[k] / max(counts[k], 1)).cpu()) for k in sums}


@torch.no_grad()
def evaluate_rollout_nrmse(
    model: nn.Module,
    sequences: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    *,
    rollout_mode: Optional[str],
    batch_size: int,
) -> float:
    """Evaluate rollout NRMSE on held-out trajectories.

    Each rollout starts at trajectory time zero and is scored in denormalized
    physical coordinates over the complete stored trajectory.
    """
    model.eval()
    base = unwrap_model(model)
    selected_mode = str(
        rollout_mode or getattr(base, "rollout_mode", "latent")
    ).lower()
    if selected_mode not in {"latent", "reencode"}:
        raise ValueError("rollout_mode must be latent or reencode")
    reencode_every = int(getattr(base, "reencode_every", 1))
    if reencode_every < 1:
        raise ValueError("reencode_every must be at least 1")

    sequences = np.asarray(sequences)
    if sequences.ndim != 3 or sequences.shape[1] < 2:
        raise ValueError("sequences must have shape [trajectory,time,state]")
    scale = torch.as_tensor(std, dtype=torch.float64, device=device).reshape(-1)
    offset = torch.as_tensor(mean, dtype=torch.float64, device=device).reshape(-1)
    if scale.numel() != sequences.shape[-1] or offset.numel() != sequences.shape[-1]:
        raise ValueError("normalization statistics must match the state dimension")

    squared_error = torch.zeros((), dtype=torch.float64, device=device)
    target_sum: Optional[torch.Tensor] = None
    target_square_sum: Optional[torch.Tensor] = None
    samples_per_channel = 0

    for start in range(0, len(sequences), max(int(batch_size), 1)):
        sequence = torch.from_numpy(
            sequences[start : start + max(int(batch_size), 1)]
        ).float().to(device, non_blocking=True)
        steps = int(sequence.shape[1]) - 1
        target = sequence.double() * scale + offset
        batch_sum = target.sum(dim=(0, 1))
        batch_square_sum = target.square().sum(dim=(0, 1))
        target_sum = batch_sum if target_sum is None else target_sum + batch_sum
        target_square_sum = (
            batch_square_sum if target_square_sum is None else target_square_sum + batch_square_sum
        )
        samples_per_channel += int(target.shape[0] * target.shape[1])
        predicted = base.rollout(
            sequence[:, 0], steps, include_initial=False,
            rollout_mode=selected_mode, reencode_every=reencode_every,
        )["x_rollout"]
        squared_error += (predicted.double() * scale + offset - target[:, 1:]).square().sum()

    if target_sum is None or target_square_sum is None or samples_per_channel == 0:
        return float("nan")
    centered_square = (
        target_square_sum - target_sum.square() / float(samples_per_channel)
    ).sum().clamp_min(1.0e-24)
    return float(torch.sqrt(squared_error / centered_square).cpu())


@torch.no_grad()
def select_reencoding_frequency(model, sequences, mean, std, device, candidates, *, batch_size):
    """Choose inference cadence by full-trajectory physical RMSE without fitting weights."""
    base = unwrap_model(model)
    was_training = model.training
    previous = (base.rollout_mode, base.reencode_every, getattr(base, "subspace_update_every", -1))
    lengths = sequence_lengths(sequences)
    physical = sequences[np.isfinite(sequences).all(axis=-1)].astype(np.float64) * np.asarray(std) + np.asarray(mean)
    centered_square = float(np.square(physical - physical.mean(axis=0)).sum())
    denominator = max(int((lengths - 1).sum()) * physical.shape[-1], 1)
    scores = []
    try:
        base.rollout_mode = "reencode"
        base.subspace_update_every = -1
        model.eval()
        for interval in reencode_candidates(candidates):
            base.reencode_every = interval
            nrmse = evaluate_rollout_nrmse(
                model, sequences, mean, std, device,
                rollout_mode="reencode", batch_size=batch_size,
            )
            rmse = nrmse * math.sqrt(max(centered_square, 1e-24) / denominator)
            scores.append({"reencode_every": interval, "rmse": rmse, "nrmse": nrmse})
    finally:
        base.rollout_mode, base.reencode_every, base.subspace_update_every = previous
        model.train(was_training)
    finite = [score for score in scores if math.isfinite(score["rmse"])]
    if not finite:
        raise RuntimeError("All reencoding candidates produced non-finite RMSE")
    selected = min(finite, key=lambda score: score["rmse"])
    return {"selected_reencode_every": selected["reencode_every"],
            "rmse": selected["rmse"], "scores": scores}


def finalize_inference_checkpoints(model, outdir, arrays, device, candidates, batch_size):
    """Keep each saved snapshot's weights, attach its own selected inference policy."""
    split = "val" if len(arrays.get("val_sequences", ())) else "test"
    reports, by_step = {}, {}
    final_step = torch.load(outdir / "latest.pt", map_location="cpu", weights_only=False)["optimizer_step"]
    for path in (outdir / "best.pt", outdir / "latest.pt", outdir / f"step_{final_step:06d}.pt"):
        if not path.exists():
            continue
        payload = torch.load(path, map_location=device, weights_only=False)
        step = int(payload["optimizer_step"])
        if step not in by_step:
            unwrap_model(model).load_state_dict(payload["model"])
            report = select_reencoding_frequency(
                model, arrays[f"{split}_sequences"], arrays["mean"], arrays["std"], device,
                candidates, batch_size=batch_size,
            )
            by_step[step] = {**report, "split": split, "optimizer_step": step}
        report = by_step[step]
        payload["reencoding_selection"] = report
        payload["reencode_candidates"] = list(candidates)
        payload["selected_reencode_every"] = report["selected_reencode_every"]
        payload["training_rollout_mode"] = "latent"
        payload["model_config"] = {**payload["model_config"], "rollout_mode": "reencode",
                                   "reencode_every": report["selected_reencode_every"]}
        if "subspace_update_every" in payload["model_config"]:
            payload["model_config"]["subspace_update_every"] = -1
        atomic_torch_save(payload, path)
        reports[path.name] = report
        print(f"{path.name}: selected reencode_every={report['selected_reencode_every']} "
              f"on {split}, RMSE={report['rmse']:.6g}", flush=True)
    save_json(reports, outdir / "reencoding_selection.json")
    return reports


def train_sequence_step(
    model: nn.Module,
    sequence_loss_fn: Callable[..., Dict[str, torch.Tensor]],
    batch,
    opt: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    *,
    loss_kwargs: Dict[str, Any],
    training_cfg: Dict[str, Any],
    optimizer_step: int,
    rollout_mode: Optional[str],
) -> tuple[Dict[str, float], bool]:
    model.train()
    use_amp = bool(training_cfg["amp"] and device.type == "cuda")

    seq = batch["sequence"].to(device, non_blocking=True)
    opt.zero_grad(set_to_none=True)
    base = unwrap_model(model)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        terms = sequence_batch_loss(
            sequence_loss_fn, base, seq,
            optimizer_step=int(optimizer_step), rollout_mode=rollout_mode,
            dense_all_starts=batch.get("dense_all_starts", False), **loss_kwargs,
        )
        loss = terms["loss"]

    if not torch.isfinite(loss.detach()).all():
        finite_terms = {
            name: float(value.detach().mean().cpu())
            for name, value in terms.items()
        }
        raise FloatingPointError(
            f"Non-finite training loss at optimizer step {optimizer_step}: {finite_terms}. "
            "Lower the learning rate or rollout horizon and restart training."
        )

    scaler.scale(loss).backward()
    if float(training_cfg["grad_clip"]) > 0:
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(
            model.parameters(), float(training_cfg["grad_clip"]),
            error_if_nonfinite=True,
        )
    previous_scale = scaler.get_scale()
    scaler.step(opt)
    scaler.update()
    updated = scaler.get_scale() >= previous_scale

    return {key: float(value.detach().mean().cpu()) for key, value in terms.items()}, updated


# -----------------------------------------------------------------------------
# Parser
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train KAE and GKAE models from YAML configs.")

    p.add_argument("--config", type=str, required=True, help="YAML with data/model/loss/training sections")
    p.add_argument("--model_file", type=str, required=True, help="Python file containing model class and loss function")
    p.add_argument("--model_class", type=str, required=True, help="Model class name in model_file")
    p.add_argument("--outdir", type=str, default="./outputs/koopman_run")
    p.add_argument(
        "--system-config", type=Path, default=DEFAULT_SYSTEM_CONFIG,
        help="Shared runtime configuration (default: src/configs/system_config.yaml).",
    )

    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "cuda:0", "cuda:1", "gpu_0", "gpu_1"],
        default=None,
        help="Override training.device from the YAML config.",
    )
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--prefetch_factor", type=int, default=None)
    p.add_argument("--optimizer_steps", type=int, default=None,
                   help="Total optimizer updates; overrides training.optimizer_steps.")
    p.add_argument("--eval_every", type=int, default=None, help="Evaluate every N optimizer steps.")
    p.add_argument("--save_every", type=int, default=None)
    p.add_argument(
        "--cooldown-seconds",
        type=float,
        default=None,
        help=(
            "Override training.startup_cooldown_seconds from the system config "
            "(use 0 to disable)."
        ),
    )
    p.add_argument(
        "--grassmann",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the model.grassmann config option.",
    )

    return p


TRAINING_LOG_PATH = ROOT.parent / "log.txt"


def training_timestamp() -> str:
    return datetime.now(ZoneInfo("Europe/Berlin")).isoformat(timespec="seconds")


def append_training_log(run_id: str, message: str) -> None:
    """Append a complete event, locked across jobs and synced to disk."""
    with TRAINING_LOG_PATH.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(f"[{training_timestamp()}] [run {run_id}] {message}\n")
        handle.flush()
        os.fsync(handle.fileno())


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def run_training(args) -> None:
    system_cfg = load_system_config(args.system_config)
    logging_period = system_cfg.get("training", {}).get("logging_period", 10)
    if type(logging_period) is not int or logging_period < 1:
        raise ValueError("system training.logging_period must be a positive integer")
    cfg = load_yaml(args.config)

    cooldown_seconds = float(
        system_cfg.get("training", {}).get("startup_cooldown_seconds", 0.0)
        if args.cooldown_seconds is None else args.cooldown_seconds
    )
    if not math.isfinite(cooldown_seconds) or cooldown_seconds < 0:
        raise ValueError("--cooldown-seconds must be a finite non-negative number")
    if cooldown_seconds > 0:
        print(
            f"Waiting {cooldown_seconds:g} seconds before starting training...",
            flush=True,
        )
        time.sleep(cooldown_seconds)

    model_cfg = dict(cfg.get("model", {}))
    if args.grassmann is not None:
        model_cfg["grassmann"] = bool(args.grassmann)
    candidates = reencode_candidates(model_cfg.get("reencode_every"))
    model_cfg["rollout_mode"] = "latent"
    model_cfg["reencode_every"] = candidates[0]
    if model_cfg.get("grassmann"):
        model_cfg["subspace_update_every"] = -1
    loss_cfg = dict(cfg.get("loss", {}))
    loss_cfg.pop("rollout_mode", None)
    training_cfg = merge_section(
        {
            "optimizer_steps": 5000,
            "batch_size": 512,
            "lr": 1e-3,
            "weight_decay": 1e-6,
            "grad_clip": 1.0,
            "seed": 0,
            "device": "auto",
            "amp": True,
            "compile": False,
            "num_workers": 0,
            "prefetch_factor": 2,
            "eval_every": 1000,
            "save_every": 1000,
        },
        cfg.get("training", {}),
    )
    for key in list(training_cfg.keys()):
        maybe_override(training_cfg, args, key)
    if "epochs" in training_cfg:
        raise ValueError("training.epochs is no longer supported; set training.optimizer_steps")
    if type(training_cfg["optimizer_steps"]) is not int or training_cfg["optimizer_steps"] < 1:
        raise ValueError("training.optimizer_steps must be a positive integer")
    outdir = Path(args.outdir)
    training_cfg["device"] = (
        configured_device(system_cfg) if args.device is None else args.device
    )

    set_seed(int(training_cfg["seed"]))
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    device = get_device(str(training_cfg["device"]))
    if device.type == "cuda":
        torch.cuda.set_device(device)

    data_cfg = dict(cfg.get("data", {}))
    if not data_cfg:
        raise ValueError("The YAML config must define a non-empty data section")
    arrays = make_data(**data_cfg)
    if str(data_cfg.get("system", "")).lower() in {
        "monkey_reaching", "reaching", "comp_neurosci_reaching", "neural_reaching"
    }:
        balance_groups = loss_cfg.get("balance_reaching_groups", True)
        if type(balance_groups) is not bool:
            raise ValueError("loss.balance_reaching_groups must be true or false")
        if balance_groups:
            loss_cfg["state_mse_group_sizes"] = [int(arrays["reaching_neural_unit_count"]), 4]
            print("Reaching state losses: 50% neural-group MSE + 50% cursor-group MSE.")
        else:
            loss_cfg.pop("state_mse_group_sizes", None)
    print(complete_trajectory_summary(arrays))
    selection_split = "val" if len(arrays.get("val_sequences", ())) else "test"
    selection_sequences = arrays[f"{selection_split}_sequences"]
    model_cfg["state_dim"] = int(arrays["train_sequences"].shape[-1])
    if str(data_cfg.get("system", "")).lower() in {
        "lorenz", "lorenz63", "lorenz_63", "lorenz-63"
    }:
        print(
            "Lorenz-63 parameters: "
            f"sigma={float(arrays['lorenz_sigma']):g}, "
            f"rho={float(arrays['lorenz_rho']):g}, "
            f"beta={float(arrays['lorenz_beta']):g}"
        )
    if str(data_cfg.get("system", "")).lower() in {
        "duffing", "duffing_oscillator", "duffing-oscillator"
    }:
        well = np.sqrt(
            float(arrays["duffing_linear_stiffness"])
            / float(arrays["duffing_cubic_stiffness"])
        )
        print(
            "Duffing double well: "
            f"stable positions +/-{well:g}, "
            f"damping={float(arrays['duffing_damping']):g}"
        )
    if str(data_cfg.get("system", "")).lower() in {
        "monkey_reaching", "reaching", "comp_neurosci_reaching", "neural_reaching"
    }:
        neural_count = int(np.asarray(
            arrays["reaching_neural_unit_count"]
        ).item())
        print(
            "Reaching joint state: "
            f"{neural_count} neural rates + 2 cursor positions + "
            "2 cursor velocities"
        )

    loader_kwargs = dict(
        batch_size=int(training_cfg["batch_size"]),
        device=device,
        num_workers=int(training_cfg["num_workers"]),
        prefetch_factor=int(training_cfg["prefetch_factor"]),
    )

    requested_sequence_horizon = max(
        int(loss_cfg.get("prediction_steps", 1) or 1),
        int(loss_cfg.get("linearity_steps", 1) or 1),
        int(model_cfg.get("specialization_horizon", 1) or 1),
    )
    split_sequence_lengths = {
        split: int(arrays[f"{split}_sequences"].shape[1])
        for split in ("train", "test")
    }
    split_horizons = {
        split: min(requested_sequence_horizon, length - 1)
        for split, length in split_sequence_lengths.items()
    }
    capped = {
        split: horizon for split, horizon in split_horizons.items()
        if horizon < requested_sequence_horizon
    }
    if capped:
        print(
            f"Using requested sequence horizon {requested_sequence_horizon}; "
            f"shorter split horizons: {capped}."
        )
    seq_ds = training_dataset(arrays, "train", split_horizons["train"], WindowDataset)
    seq_dl = make_loader(seq_ds, shuffle=True,
                         collate_fn=getattr(seq_ds, "collate_fn", None), **loader_kwargs)
    if not len(seq_dl):
        raise ValueError("No training batches are available")
    print(
        f"Using {len(seq_ds)} sequence items in {len(seq_dl)} shuffled batches; "
        f"cycling through data for {int(training_cfg['optimizer_steps'])} optimizer steps."
    )
    test_seq_ds = training_dataset(arrays, "test", split_horizons["test"], WindowDataset)
    test_seq_dl = make_loader(test_seq_ds, shuffle=False,
                              collate_fn=getattr(test_seq_ds, "collate_fn", None), **loader_kwargs)

    model_module = import_module_from_file(args.model_file)
    ModelClass = getattr(model_module, args.model_class)
    sequence_loss_fn = getattr(model_module, "koopman_sequence_loss")

    model = ModelClass(**model_cfg).to(device)
    base_model = model
    if hasattr(base_model, "architecture_metadata"):
        model_cfg.update(base_model.architecture_metadata())

    rollout_mode = "latent"
    print(f"Training uses latent-only rollouts; inference candidates: {candidates}; selection split: {selection_split}")

    if bool(training_cfg["compile"]) and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]

    opt = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(training_cfg["lr"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )

    use_amp = bool(training_cfg["amp"] and device.type == "cuda")
    scaler = make_grad_scaler(use_amp)

    outdir.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "model_file": str(Path(args.model_file)),
            "model_class": args.model_class,
            "model": model_cfg,
            "loss": loss_cfg,
            "training": training_cfg,
            "data": data_cfg,
            "system": system_cfg,
            "reencode_candidates": candidates,
            "training_rollout_mode": "latent",
        },
        outdir / "resolved_config.json",
    )

    print(f"Device: {device}")
    print(f"Model: {args.model_class} from {args.model_file}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"System: {data_cfg['system']}")
    print(f"Train pairs: {arrays['train_x_t'].shape}")
    print(f"Test pairs:  {arrays['test_x_t'].shape}")
    if data_cfg["system"].lower() in {
        "goodwin", "goodwin_oscillator", "goodwin-oscillator"
    }:
        split_sizes = np.asarray(arrays["goodwin_split_sizes"]).astype(int).tolist()
        print(f"Goodwin trajectories (train/validation/test): {split_sizes}")
        print(f"Goodwin samples per trajectory: {arrays['train_sequences'].shape[1]}")
    elif data_cfg["system"].lower() in {
        "monkey_reaching", "reaching", "comp_neurosci_reaching", "neural_reaching"
    }:
        split_sizes = np.asarray(arrays["reaching_split_sizes"]).astype(int).tolist()
        targets = np.asarray(
            arrays["reaching_selected_targets"]
        ).astype(int).tolist()
        print(f"Reaching trajectories (train/test): {split_sizes}")
        print(f"Reaching path class: {arrays['reaching_path_class'].item()}")
        print(f"Reaching target clusters (fixed six): {targets}")
        print(f"Reaching state dimension: {arrays['train_sequences'].shape[-1]}")
    print(f"Sequence shapes (train/test): {arrays['train_sequences'].shape}, {arrays['test_sequences'].shape}")
    is_grassmann_model = bool(getattr(base_model, "grassmann", False)) or (
        getattr(base_model, "model_type", None) == "grassmann_koopman"
    )
    if is_grassmann_model:
        print("Single-subspace optimization: Euclidean optimization")
        print(f"latent_dim: {base_model.latent_dim}")
        print(f"subspace_dim: {base_model.subspace_dim}")
    print(f"Autonomous rollout mode: {rollout_mode or 'model default / baseline'}")
    best_validation_rollout_rmse = float("inf")
    last_train_eval_metrics = None
    last_test_metrics = None
    history = []

    def checkpoint_payload(optimizer_step: int) -> Dict[str, Any]:
        return {
            "model": unwrap_model(model).state_dict(),
            "model_config": model_cfg,
            "loss_config": loss_cfg,
            "training_config": training_cfg,
            "data_config": data_cfg,
            "model_file": str(Path(args.model_file).name),
            "model_class": args.model_class,
            "optimizer_step": int(optimizer_step),
            "mean": arrays["mean"],
            "std": arrays["std"],
            "training_rollout_mode": "latent",
            "reencode_candidates": candidates,
        }

    if not len(seq_dl):
        raise ValueError("No training batches are available")
    batches = iter(seq_dl)
    optimizer_step = 0
    while optimizer_step < int(training_cfg["optimizer_steps"]):
        try:
            batch = next(batches)
        except StopIteration:
            batches = iter(seq_dl)
            batch = next(batches)
        step_started_at = training_timestamp()
        train_metrics, updated = train_sequence_step(
            model, sequence_loss_fn, batch, opt, scaler, device,
            loss_kwargs=loss_cfg, training_cfg=training_cfg,
            optimizer_step=optimizer_step + 1, rollout_mode=rollout_mode,
        )
        if not updated:
            print(f"Optimizer update {optimizer_step + 1} skipped by AMP; retrying without counting it.", flush=True)
            continue
        optimizer_step += 1

        do_eval = (
            optimizer_step % max(int(training_cfg["eval_every"]), 1) == 0
            or optimizer_step == int(training_cfg["optimizer_steps"])
        )

        if do_eval:
            # Recompute both sides after optimization with the same model
            # snapshot, eval-mode operator, rollout mode, and complete weighted
            # objective. Online optimization metrics remain separate below.
            train_eval_metrics = evaluate_sequence(
                model, sequence_loss_fn, seq_dl, device, loss_cfg,
                optimizer_step=optimizer_step, rollout_mode=rollout_mode,
            )
            test_metrics = evaluate_sequence(
                model, sequence_loss_fn, test_seq_dl, device, loss_cfg,
                optimizer_step=optimizer_step, rollout_mode=rollout_mode,
            )
            selection = select_reencoding_frequency(
                model, selection_sequences, arrays["mean"], arrays["std"], device,
                candidates, batch_size=int(training_cfg["batch_size"]),
            )
            selection.update(split=selection_split, optimizer_step=optimizer_step)
            selected_score = next(
                score for score in selection["scores"]
                if score["reencode_every"] == selection["selected_reencode_every"]
            )
            test_metrics["rollout_nrmse"] = selected_score["nrmse"]
            test_metrics["rollout_rmse"] = selection["rmse"]
            test_metrics["selected_reencode_every"] = selection["selected_reencode_every"]
            last_train_eval_metrics = train_eval_metrics
            last_test_metrics = test_metrics
            if selection["rmse"] < best_validation_rollout_rmse:
                best_validation_rollout_rmse = selection["rmse"]
                payload = checkpoint_payload(optimizer_step)
                payload.update(selection_metric="validation_reencoded_rollout_rmse",
                               selection_metric_value=selection["rmse"],
                               selection_split=selection_split,
                               reencoding_selection=selection,
                               selected_reencode_every=selection["selected_reencode_every"])
                payload["model_config"] = {
                    **payload["model_config"], "rollout_mode": "reencode",
                    "reencode_every": selection["selected_reencode_every"],
                }
                atomic_torch_save(payload, outdir / "best.pt")

        do_log = (
            optimizer_step % logging_period == 0
            or optimizer_step == int(training_cfg["optimizer_steps"])
        )
        if do_log:
            timestamp = training_timestamp()
            row = {
                "optimizer_step": optimizer_step,
                "timestamp": timestamp,
                "step_started_at": step_started_at,
                **{f"optimization_train_{k}": v for k, v in train_metrics.items()},
            }
            if (
                do_eval and last_train_eval_metrics is not None
                and last_test_metrics is not None
            ):
                row.update({
                    f"train_{k}": v for k, v in last_train_eval_metrics.items()
                })
                row.update({
                    f"test_{k}": v for k, v in last_test_metrics.items()
                })
            history.append(row)
            # Persist one history entry per logging period.
            history_tmp = outdir / "history.json.tmp"
            save_json(history, history_tmp)
            os.replace(history_tmp, outdir / "history.json")

            msg = (
                f"[step {optimizer_step}/{int(training_cfg['optimizer_steps'])}] "
                f"optimization_train={train_metrics.get('loss', float('nan')):.6f}"
            )
            for k in [
                "reconstruction",
                "projected_reconstruction_error",
                "ambient_reconstruction_error",
                "grassmann_projected_energy",
                "grassmann_complement_energy",
                "prediction",
                "rollout",
                "rollout_velocity",
                "rollout_speed",
                "latent",
                "linearity",
                "linf",
                "projection",
                "chart",
                "dyn_chart",
                "dynamics",
                "fitted_operator_samples",
                "fitted_operator_spectral_radius",
                "grassmann_distance_min",
                "grassmann_distance_mean",
                "ambient_operator_frobenius_norm",
                "ambient_operator_spectral_radius",
                "local_operator_frobenius_norm",
                "local_operator_spectral_radius_max",
                "stability",
                "operator_frobenius_norm",
                "operator_spectral_radius",
            ]:
                if k in train_metrics:
                    msg += f" {k}={train_metrics[k]:.6f}"
            if do_eval and last_train_eval_metrics is not None and last_test_metrics is not None:
                msg += (
                    f" train={last_train_eval_metrics['loss']:.6f}"
                )
                msg += f" test={last_test_metrics['loss']:.6f}"
                msg += (
                    " validation_rollout_nrmse="
                    f"{last_test_metrics['rollout_nrmse']:.6f}"
                )
                msg += (
                    f" validation_rollout_rmse={last_test_metrics['rollout_rmse']:.6f}"
                    f" reencode_every={last_test_metrics['selected_reencode_every']}"
                )
            print(msg, flush=True)

        save_every = max(int(training_cfg.get("save_every", 1)), 1)
        if optimizer_step == 1 or optimizer_step % save_every == 0 or optimizer_step == int(training_cfg["optimizer_steps"]):
            payload = checkpoint_payload(optimizer_step)
            atomic_torch_save(payload, outdir / "latest.pt")
            if optimizer_step == int(training_cfg["optimizer_steps"]):
                atomic_torch_save(payload, outdir / f"step_{optimizer_step:06d}.pt")

    if not math.isfinite(best_validation_rollout_rmse):
        # Long unrefreshed rollouts may diverge even when periodic inference is viable.
        payload = checkpoint_payload(int(training_cfg["optimizer_steps"]))
        payload.update(selection_metric="final_step_fallback", selection_metric_value=None,
                       selection_split=selection_split)
        atomic_torch_save(payload, outdir / "best.pt")
        print("No finite reencoded rollout score; selecting inference cadence for the final weights.", flush=True)

    inference_reports = finalize_inference_checkpoints(
        model, outdir, arrays, device, candidates, int(training_cfg["batch_size"]),
    )

    test_metrics = {**last_test_metrics, "reencoding_selection": inference_reports}
    save_json(test_metrics, outdir / "test_metrics.json")

    print(
        "Done. Best validation reencoded rollout RMSE: "
        f"{best_validation_rollout_rmse:.6f}. "
        f"Final test loss: {test_metrics['loss']:.6f}. Outputs: {outdir}"
    )


def main() -> None:
    args = build_parser().parse_args()
    run_id = f"{os.getpid()}-{time.time_ns()}"
    command = shlex.join([sys.executable, *sys.argv])
    append_training_log(run_id, f"STARTED cwd={Path.cwd()} command={command}")
    try:
        run_training(args)
    except BaseException:
        append_training_log(run_id, "FAILED\n" + traceback.format_exc())
        raise
    else:
        append_training_log(run_id, "FINISHED")


if __name__ == "__main__":
    main()
