from __future__ import annotations

import fcntl
import hashlib
import json
from numbers import Integral
import os
import random
import re
import tempfile
from zipfile import BadZipFile
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import yaml
from scipy.integrate import solve_ivp
from torch.utils.data import DataLoader, Dataset

DEFAULT_REENCODE_CANDIDATES = (10, 25, 50, 100)


def reencode_candidates(value=None):
    values = list(DEFAULT_REENCODE_CANDIDATES if value is None else
                  value if isinstance(value, (list, tuple)) else [value])
    if not values or any(isinstance(v, bool) or not isinstance(v, Integral) or v < 1 for v in values):
        raise ValueError('reencode_every must be a nonempty array of positive integers')
    return list(dict.fromkeys(int(v) for v in values))


MULTISTABLE_NAMES = {'lotka_volterra', 'lorenz_bistable'}
MULTISTABLE_3D_SYSTEMS = {'lorenz_bistable'}
MULTISTABLE_DISPLAY_NAMES = {
    'lotka_volterra': 'Competitive Lotka–Volterra',
    'lorenz_bistable': 'Bistable Lorenz-63',
}


def canonical_system(system: str) -> str:
    return str(system).strip().lower()


def system_specification(system, *, lorenz_sigma=10.0,
                         lorenz_rho=10.0, lorenz_beta=8.0 / 3.0):
    """Return the vector field, initial-condition box, and stable equilibria."""
    key = canonical_system(system)
    if key == 'lotka_volterra':
        def rhs(x):
            return x * (1.0 - x - 2.0 * x[..., ::-1])
        return rhs, np.full(2, 0.05), np.full(2, 1.5), np.eye(2)
    if key == 'lorenz_bistable':
        if not np.isfinite([lorenz_sigma, lorenz_rho, lorenz_beta]).all():
            raise ValueError('Lorenz parameters must be finite')
        if lorenz_sigma <= 0 or lorenz_beta <= 0 or lorenz_rho <= 1:
            raise ValueError('Bistable Lorenz requires sigma > 0, beta > 0, rho > 1')
        coordinate = np.sqrt(lorenz_beta * (lorenz_rho - 1))
        attractors = np.array([[-coordinate, -coordinate, lorenz_rho - 1],
                               [coordinate, coordinate, lorenz_rho - 1]])
        def rhs(x):
            a, b, c = np.moveaxis(x, -1, 0)
            return np.stack((lorenz_sigma * (b - a), a * (lorenz_rho - c) - b,
                             a * b - lorenz_beta * c), axis=-1)
        return rhs, np.array([-10., -10., 0.]), np.array([10., 10., 15.]), attractors
    raise ValueError(f'Unknown multistable system: {system!r}')


def generate_multistable_sequences(system, n_traj, *, steps=1000, dt=0.02, seed=0,
                       noise_std=0.0, rtol=1e-9, atol=1e-11, **parameters):
    """Sample independent initial conditions; retain the initial state (no warmup)."""
    if n_traj < 0 or steps < 2 or not np.isfinite(dt) or dt <= 0:
        raise ValueError('n_traj must be nonnegative, steps >= 2, and dt positive')
    if noise_std < 0 or rtol <= 0 or atol <= 0:
        raise ValueError('noise_std must be nonnegative and solver tolerances positive')
    rhs, lower, upper, _ = system_specification(system, **parameters)
    rng = np.random.default_rng(seed)
    initial = rng.uniform(lower, upper, size=(n_traj, len(lower)))
    if not n_traj:
        return np.empty((0, steps, len(lower)), dtype=np.float32)
    time = np.arange(steps, dtype=np.float64) * dt
    shape = initial.shape
    solution = solve_ivp(
        lambda _, state: rhs(state.reshape(shape)).reshape(-1),
        (0.0, time[-1]), initial.reshape(-1), t_eval=time,
        method='DOP853', rtol=rtol, atol=atol,
    )
    if not solution.success:
        raise RuntimeError(f'{system} integration failed: {solution.message}')
    sequences = solution.y.T.reshape(steps, *shape).transpose(1, 0, 2).copy()
    if noise_std:
        sequences += rng.normal(0.0, noise_std, size=sequences.shape)
    if not np.isfinite(sequences).all():
        raise FloatingPointError(f'{system} produced non-finite states')
    return sequences.astype(np.float32)


DEFAULT_SYSTEM_CONFIG = Path(__file__).resolve().parent / "configs" / "system_config.yaml"


def load_system_config(path: str | Path = DEFAULT_SYSTEM_CONFIG) -> dict[str, Any]:
    """Load and minimally validate the shared system configuration."""
    config_path = Path(path)
    with open(config_path, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"System config root must be a mapping: {config_path}")

    compute = config.get("compute", {})
    if not isinstance(compute, dict):
        raise ValueError(f"System config compute section must be a mapping: {config_path}")

    training = config.get("training", {})
    if not isinstance(training, dict):
        raise ValueError(f"System config training section must be a mapping: {config_path}")
    return config


def configured_device(config: dict[str, Any]) -> str:
    device = config.get("compute", {}).get("device")
    return str(device).strip() if device is not None and str(device).strip() else "auto"


GOODWIN_NAMES = {"goodwin", "goodwin_oscillator", "goodwin-oscillator"}
REACHING_NAMES = {"monkey_reaching", "reaching", "comp_neurosci_reaching", "neural_reaching"}
PAM_NAMES = {"pam", "pamy2", "pneumatic_robot_arm_pam"}
LORENZ_NAMES = {"lorenz", "lorenz63", "lorenz_63", "lorenz-63"}
DUFFING_NAMES = {"duffing", "duffing_oscillator", "duffing-oscillator"}
DYSTS_SYSTEMS = {
    "chua": ("Chua", {"alpha": 15.6, "beta": 28.0, "m0": -1.142857, "m1": -0.71429}),
    "dadras": ("Dadras", {"c": 2.0, "e": 9.0, "o": 2.7, "p": 3.0, "r": 1.7}),
    "lu_chen_cheng": ("LuChenCheng", {"a": -10.0, "b": -4.0, "c": 18.1}),
    "shimizu_morioka": ("ShimizuMorioka", {"a": 0.85, "b": 0.5}),
}
DYSTS_ALIASES = {
    "lu-chen-cheng": "lu_chen_cheng", "luchencheng": "lu_chen_cheng",
    "shimizu-morioka": "shimizu_morioka", "shimizumorioka": "shimizu_morioka",
}
DYSTS_CACHE_SCHEMA = 1
LORENZ_CACHE_SCHEMA = 1


def _load_generated_cache(
    path: Path, specification: str,
) -> Optional[Dict[str, np.ndarray]]:
    """Load a complete, parameter-matched generated-data cache entry if available."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["cache_specification"].item()) != specification:
                return None
            return {
                key: np.asarray(archive[key])
                for key in archive.files if key != "cache_specification"
            }
    except (OSError, ValueError, KeyError, EOFError, BadZipFile):
        return None


def _cached_generated_data(
    *, cache_dir: str | Path, specification: Dict[str, Any], generate,
) -> Dict[str, np.ndarray]:
    """Load or atomically create one normalized dataset cache entry."""
    specification_json = json.dumps(
        specification, sort_keys=True, separators=(",", ":")
    )
    digest = hashlib.sha256(specification_json.encode("utf-8")).hexdigest()[:20]
    directory = Path(cache_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{specification['system']}_{digest}.npz"
    lock_path = directory / f"{specification['system']}_{digest}.lock"

    cached = _load_generated_cache(path, specification_json)
    if cached is not None:
        print(f"Loading cached {specification['system']} data: {path}")
        return cached

    with open(lock_path, "a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        cached = _load_generated_cache(path, specification_json)
        if cached is not None:
            print(f"Loading cached {specification['system']} data: {path}")
            return cached

        print(f"Generating {specification['system']} data and caching it at: {path}")
        arrays = generate()
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.stem}-", suffix=".npz", dir=directory
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            np.savez_compressed(
                temporary, cache_specification=np.asarray(specification_json),
                **arrays,
            )
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return arrays


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> torch.device:
    normalized = str(name).strip().lower()
    aliases = {"gpu_0": "cuda:0", "gpu_1": "cuda:1"}
    normalized = aliases.get(normalized, normalized)
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    if normalized.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested {normalized}, but CUDA is unavailable")
        device = torch.device(normalized)
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested {normalized}, but only {torch.cuda.device_count()} "
                "CUDA device(s) are available"
            )
        return device
    if normalized == "cpu":
        return torch.device("cpu")
    raise ValueError(
        "device must be auto, cpu, cuda, cuda:0, cuda:1, gpu_0, or gpu_1"
    )


def make_grad_scaler(enabled: bool):
    return torch.amp.GradScaler("cuda", enabled=enabled)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_orig_mod", model)


def denorm(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return x * std + mean


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


# -----------------------------------------------------------------------------
# Synthetic systems
# -----------------------------------------------------------------------------


def generate_lorenz_sequences(
    n_traj: int,
    steps: int = 1000,
    dt: float = 0.01,
    transient: int = 1000,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    noise_std: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Generate Lorenz-63 trajectories with a fourth-order Runge--Kutta step."""
    if n_traj < 1 or steps < 2:
        raise ValueError("n_traj must be positive and steps must be at least 2")
    if dt <= 0 or transient < 0:
        raise ValueError("dt must be positive and transient must be nonnegative")
    rng = np.random.default_rng(seed)
    state = rng.uniform((-15.0, -20.0, 5.0), (15.0, 20.0, 40.0), (n_traj, 3))

    def derivative(x: np.ndarray) -> np.ndarray:
        return np.stack((
            sigma * (x[:, 1] - x[:, 0]),
            x[:, 0] * (rho - x[:, 2]) - x[:, 1],
            x[:, 0] * x[:, 1] - beta * x[:, 2],
        ), axis=-1)

    def rk4(x: np.ndarray) -> np.ndarray:
        k1 = derivative(x)
        k2 = derivative(x + 0.5 * dt * k1)
        k3 = derivative(x + 0.5 * dt * k2)
        k4 = derivative(x + dt * k3)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    for _ in range(int(transient)):
        state = rk4(state)
    sequences = np.empty((n_traj, steps, 3), dtype=np.float64)
    for index in range(steps):
        sequences[:, index] = state
        state = rk4(state)
    if noise_std:
        sequences += rng.normal(0.0, noise_std, sequences.shape)
    if not np.isfinite(sequences).all():
        raise FloatingPointError("Lorenz integration produced non-finite states")
    return sequences.astype(np.float32)


def generate_duffing_sequences(
    n_traj: int,
    steps: int = 1000,
    dt: float = 0.02,
    transient: int = 0,
    damping: float = 0.3,
    linear_stiffness: float = 1.0,
    cubic_stiffness: float = 1.0,
    noise_std: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Generate the autonomous damped Duffing double-well system.

    The state is ``(position, velocity)`` and obeys
    ``x' = v, v' = a*x - b*x**3 - damping*v``. Positive ``a`` and ``b``
    produce stable wells at ``x = +/-sqrt(a/b)``.
    """
    if n_traj < 1 or steps < 2:
        raise ValueError("n_traj must be positive and steps must be at least 2")
    if dt <= 0 or transient < 0 or damping < 0:
        raise ValueError("dt must be positive; transient and damping must be nonnegative")
    if linear_stiffness <= 0 or cubic_stiffness <= 0:
        raise ValueError("Duffing stiffness parameters must be positive")
    rng = np.random.default_rng(seed)
    state = rng.uniform((-2.0, -1.5), (2.0, 1.5), size=(n_traj, 2))

    def derivative(y: np.ndarray) -> np.ndarray:
        position, velocity = y[:, 0], y[:, 1]
        return np.stack((
            velocity,
            linear_stiffness * position
            - cubic_stiffness * position ** 3
            - damping * velocity,
        ), axis=-1)

    def rk4(y: np.ndarray) -> np.ndarray:
        k1 = derivative(y)
        k2 = derivative(y + 0.5 * dt * k1)
        k3 = derivative(y + 0.5 * dt * k2)
        k4 = derivative(y + dt * k3)
        return y + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)

    for _ in range(int(transient)):
        state = rk4(state)
    sequences = np.empty((n_traj, steps, 2), dtype=np.float64)
    for index in range(steps):
        sequences[:, index] = state
        state = rk4(state)
    if noise_std:
        sequences += rng.normal(0.0, noise_std, sequences.shape)
    if not np.isfinite(sequences).all():
        raise FloatingPointError("Duffing integration produced non-finite states")
    return sequences.astype(np.float32)


def generate_dysts_splits(
    system: str,
    n_train_traj: int,
    n_val_traj: int,
    n_test_traj: int,
    steps: int,
    warmup_steps: int,
    seed: int,
    parameters: Optional[Dict[str, float]] = None,
    perturbation_scale: float = 0.2,
    dt_scale: float = 1.0,
    method: str = "DOP853",
    rtol: float = 1.0e-12,
    atol: float = 1.0e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate deterministic Dysts splits from perturbed default conditions."""
    try:
        from dysts import flows
    except ImportError as exc:
        raise ImportError(
            "The Dysts systems require the 'dysts' package (pip install dysts)."
        ) from exc

    key = DYSTS_ALIASES.get(system.lower(), system.lower())
    class_name, default_parameters = DYSTS_SYSTEMS[key]
    selected_parameters = dict(default_parameters)
    if parameters:
        selected_parameters.update(parameters)
    dynamical_system = getattr(flows, class_name)(parameters=selected_parameters)
    default_ic = np.asarray(dynamical_system.ic, dtype=np.float64).copy()
    coordinate_std = np.asarray(dynamical_system.std, dtype=np.float64)
    if dt_scale <= 0.0:
        raise ValueError("Dysts dt_scale must be positive")
    dynamical_system.dt = float(dynamical_system.dt) * float(dt_scale)
    stored_dt = float(dynamical_system.dt)
    total_steps = int(warmup_steps) + int(steps)
    split_rngs = [
        np.random.default_rng(child)
        for child in np.random.SeedSequence(seed).spawn(3)
    ]

    def integrate_split(
        count: int, rng: np.random.Generator, include_default: bool,
    ) -> np.ndarray:
        trajectories = np.empty(
            (count, steps, default_ic.size), dtype=np.float32
        )
        for index in range(count):
            if include_default and index == 0:
                initial = default_ic
            else:
                initial = default_ic + rng.normal(
                    size=default_ic.shape
                ) * perturbation_scale * coordinate_std
            # Assigning ``ic`` avoids version-specific init_cond handling while
            # still using Dysts for both the equations and integration.
            dynamical_system.ic = np.asarray(initial, dtype=np.float64)
            trajectory = dynamical_system.make_trajectory(
                total_steps, resample=False, postprocess=False,
                method=method, rtol=rtol, atol=atol,
            )
            if trajectory is None:
                raise RuntimeError(
                    f"Dysts failed to integrate {class_name} trajectory {index}"
                )
            trajectory = np.asarray(trajectory, dtype=np.float64)
            trajectories[index] = trajectory[
                warmup_steps:warmup_steps + steps
            ]
        return trajectories

    splits = tuple(
        integrate_split(count, rng, split_index == 0)
        for split_index, (count, rng) in enumerate(zip(
            (n_train_traj, n_val_traj, n_test_traj), split_rngs
        ))
    )
    metadata = {
        "class_name": class_name,
        "parameters": selected_parameters,
        "default_ic": default_ic,
        "coordinate_std": coordinate_std,
        "dt": stored_dt,
    }
    return (*splits, metadata)


def make_pairs(seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return seq[:, :-1].reshape(-1, seq.shape[-1]), seq[:, 1:].reshape(-1, seq.shape[-1])


def normalize_splits(train_raw: np.ndarray, val_raw: np.ndarray, test_raw: np.ndarray) -> Dict[str, np.ndarray]:
    flat = train_raw.reshape(-1, train_raw.shape[-1])
    mean, std = flat.mean(0).astype(np.float32), flat.std(0).astype(np.float32)
    std[std < 1e-6] = 1
    norm = lambda x: ((x - mean) / std).astype(np.float32)
    train, val, test = map(norm, (train_raw, val_raw, test_raw))
    tx, ty = make_pairs(train); vx, vy = make_pairs(val); sx, sy = make_pairs(test)
    return dict(
        train_sequences=train, val_sequences=val, test_sequences=test,
        train_x_t=tx, train_x_tp1=ty,
        val_x_t=vx, val_x_tp1=vy,
        test_x_t=sx, test_x_tp1=sy,
        mean=mean, std=std,
    )


def normalize_train_test(
    train_raw: np.ndarray, test_raw: np.ndarray
) -> Dict[str, np.ndarray]:
    """Normalize train/test data using statistics from training only."""
    flat = train_raw.reshape(-1, train_raw.shape[-1])
    mean, std = flat.mean(0).astype(np.float32), flat.std(0).astype(np.float32)
    std[std < 1e-6] = 1
    train = ((train_raw - mean) / std).astype(np.float32)
    test = ((test_raw - mean) / std).astype(np.float32)
    tx, ty = make_pairs(train)
    sx, sy = make_pairs(test)
    return dict(
        train_sequences=train, test_sequences=test,
        train_x_t=tx, train_x_tp1=ty,
        test_x_t=sx, test_x_tp1=sy,
        mean=mean, std=std,
    )


GOODWIN_DEFAULT_PRODUCTION = 20.0
GOODWIN_DEFAULT_HILL_EXPONENT = 10.0
GOODWIN_DEFAULT_DEGRADATION = (1.0, 1.0, 1.0)
GOODWIN_CACHE_SCHEMA = 2
GOODWIN_DEFAULT_PARAMETERS = {
    "dt": 0.06, "duration": 40.0, "transient": 0.0,
    "production": GOODWIN_DEFAULT_PRODUCTION,
    "hill_exponent": GOODWIN_DEFAULT_HILL_EXPONENT,
    "degradation": GOODWIN_DEFAULT_DEGRADATION,
    "trajectory_count": 500, "seed": 6,
    "train_trajectories": 180, "val_trajectories": 20, "test_trajectories": 20,
    "split_seed": 1, "test_seed": 7,
    "initial_x_min": 0.02, "initial_x_max": 8.0,
    "initial_y_min": 0.01, "initial_y_max": 1.5,
    "initial_z_min": 0.02, "initial_z_max": 1.8,
}


def goodwin_rhs(
    state: np.ndarray,
    production: float = GOODWIN_DEFAULT_PRODUCTION,
    hill_exponent: float = GOODWIN_DEFAULT_HILL_EXPONENT,
    degradation: tuple[float, float, float] = GOODWIN_DEFAULT_DEGRADATION,
) -> np.ndarray:
    """Goodwin negative-feedback oscillator vector field.

    dx/dt = production / (1 + z**hill) - d1*x
    dy/dt = x - d2*y
    dz/dt = y - d3*z
    """
    x, y, z = np.moveaxis(np.asarray(state), -1, 0)
    d1, d2, d3 = degradation
    repression = production / (1.0 + np.maximum(z, 0.0) ** hill_exponent)
    return np.stack((repression - d1 * x, x - d2 * y, y - d3 * z), axis=-1)


def simulate_goodwin_batch(
    initial_states: np.ndarray,
    dt: float = 0.01,
    duration: float = 120.0,
    transient: float = 200.0,
    production: float = GOODWIN_DEFAULT_PRODUCTION,
    hill_exponent: float = GOODWIN_DEFAULT_HILL_EXPONENT,
    degradation: tuple[float, float, float] = GOODWIN_DEFAULT_DEGRADATION,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate every initial condition together with vectorized RK4."""
    if not np.isfinite([dt, duration, transient, production, hill_exponent]).all():
        raise ValueError("Goodwin simulation parameters must be finite")
    if dt <= 0.0 or duration <= 0.0 or transient < 0.0:
        raise ValueError("dt and duration must be positive; transient must be non-negative")
    if production <= 0.0 or hill_exponent <= 0.0:
        raise ValueError("production and hill_exponent must be positive")
    if (np.shape(degradation) != (3,) or not np.isfinite(degradation).all()
            or np.any(np.asarray(degradation) <= 0)):
        raise ValueError("degradation must contain three finite positive rates")
    if (np.ndim(initial_states) != 2 or np.shape(initial_states)[1] != 3
            or not np.isfinite(initial_states).all()):
        raise ValueError("initial_states must have shape (trajectories, 3) and be finite")

    kept_steps = int(round(duration / dt))
    if kept_steps < 1:
        raise ValueError("duration must retain at least two samples")
    transient_steps = int(round(transient / dt))
    total_steps = transient_steps + kept_steps
    states = np.empty(
        (len(initial_states), kept_steps + 1, 3), dtype=np.float64
    )
    state = np.asarray(initial_states, dtype=np.float64).copy()
    rhs_options = (production, hill_exponent, degradation)

    output_index = 0
    for step in range(total_steps + 1):
        if step >= transient_steps:
            states[:, output_index] = state
            output_index += 1
        if step == total_steps:
            break

        k1 = goodwin_rhs(state, *rhs_options)
        k2 = goodwin_rhs(state + 0.5 * dt * k1, *rhs_options)
        k3 = goodwin_rhs(state + 0.5 * dt * k2, *rhs_options)
        k4 = goodwin_rhs(state + dt * k3, *rhs_options)
        state += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    if not np.isfinite(states).all():
        raise FloatingPointError("Goodwin integration produced non-finite states; reduce dt")
    return dt * np.arange(kept_steps + 1), states


def generate_goodwin_dataset(
    trajectory_count: int = 500,
    seed: int = 6,
    train_trajectories: int = 180,
    val_trajectories: int = 20,
    test_trajectories: int = 20,
    split_seed: int = 1,
    test_seed: int = 7,
    initial_states: np.ndarray | None = None,
    initial_x_min: float = 0.02,
    initial_x_max: float = 8.0,
    initial_y_min: float = 0.01,
    initial_y_max: float = 1.5,
    initial_z_min: float = 0.02,
    initial_z_max: float = 1.8,
    **simulation_options: object,
) -> Dict[str, np.ndarray]:
    """Generate the canonical Goodwin trajectories and deterministic train/test split."""
    if any(isinstance(n, bool) or not isinstance(n, Integral)
           for n in (trajectory_count, train_trajectories, val_trajectories, test_trajectories)):
        raise ValueError("Goodwin trajectory counts must be integers")
    if trajectory_count < 1:
        raise ValueError("trajectory_count must be positive")
    lows = np.asarray((initial_x_min, initial_y_min, initial_z_min))
    highs = np.asarray((initial_x_max, initial_y_max, initial_z_max))
    if (not np.isfinite([lows, highs]).all()
            or np.any(lows <= 0.0) or np.any(highs <= lows)):
        raise ValueError("each initial range must satisfy 0 < minimum < maximum")
    if initial_states is None:
        rng = np.random.default_rng(seed)
        x_low, y_low, z_low = lows
        x_high, y_high, z_high = highs
        corners = np.asarray(
            [
                (x_low, y_low, z_low),
                (x_high, y_low, z_low),
                (x_low, y_high, z_low),
                (x_low, y_low, z_high),
                (x_high, y_high, z_low),
                (x_high, y_low, z_high),
                (x_low, y_high, z_high),
                (x_high, y_high, z_high),
            ],
            dtype=float,
        )
        if trajectory_count <= len(corners):
            initial_states = corners[:trajectory_count]
        else:
            extra = np.exp(
                rng.uniform(
                    np.log(lows),
                    np.log(highs),
                    size=(trajectory_count - len(corners), 3),
                )
            )
            initial_states = np.vstack((corners, extra))
    else:
        initial_states = np.asarray(initial_states, dtype=float)
        if initial_states.ndim != 2 or initial_states.shape[1] != 3:
            raise ValueError("initial_states must have shape (trajectories, 3)")
        trajectory_count = len(initial_states)

    if (train_trajectories < 1 or val_trajectories < 1 or test_trajectories < 1
            or train_trajectories + val_trajectories > trajectory_count):
        raise ValueError("Goodwin train/validation counts must fit within the original pool; all split counts must be positive")
    if test_seed == seed:
        raise ValueError("Goodwin test_seed must differ from the original pool seed")
    options = {"dt": 0.06, "duration": 40.0, "transient": 0.0, **simulation_options}
    time, states = simulate_goodwin_batch(initial_states, **options)
    split_order = np.random.default_rng(split_seed).permutation(len(states))
    train_indices = split_order[:train_trajectories]
    # Never resize or reshuffle the original pool: its old test split is now validation.
    val_indices = split_order[train_trajectories:train_trajectories + val_trajectories]
    test_initial_states = np.exp(np.random.default_rng(test_seed).uniform(
        np.log(lows), np.log(highs), size=(test_trajectories, 3),
    ))
    if set(map(tuple, test_initial_states)) & set(map(tuple, initial_states)):
        raise ValueError("Goodwin test initial states overlap the original pool; change test_seed")
    _, test_states = simulate_goodwin_batch(test_initial_states, **options)
    test_indices = np.arange(trajectory_count, trajectory_count + test_trajectories)
    all_states = np.concatenate((states, test_states))
    return dict(
        time=time,
        states=all_states,
        train_states=states[train_indices],
        val_states=states[val_indices],
        test_states=test_states,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        split_seed=np.asarray(split_seed, dtype=np.int64),
        dt=float(time[1] - time[0]),
        duration=float(options["duration"]),
        transient=float(options["transient"]),
        production=float(simulation_options.get("production", GOODWIN_DEFAULT_PRODUCTION)),
        hill_exponent=float(
            simulation_options.get("hill_exponent", GOODWIN_DEFAULT_HILL_EXPONENT)
        ),
        degradation=np.asarray(
            simulation_options.get("degradation", GOODWIN_DEFAULT_DEGRADATION), dtype=float
        ),
        initial_states=np.concatenate((initial_states, test_initial_states)),
        test_seed=np.asarray(test_seed, dtype=np.int64),
        initial_ranges=np.column_stack((lows, highs)),
        seed=seed,
    )


def _normalize_goodwin_data(archive, steps, noise_std):
    """Normalize generated train/validation/test data using training statistics only."""
    required = {"train_states", "val_states", "test_states"}
    missing = sorted(required.difference(archive.keys()))
    if missing:
        raise KeyError(
            f"Goodwin data has no canonical split arrays: {missing}"
        )
    train_raw = np.asarray(archive["train_states"], dtype=np.float32)
    test_raw = np.asarray(archive["test_states"], dtype=np.float32)
    val_raw = np.asarray(archive["val_states"], dtype=np.float32)
    time = np.asarray(archive["time"], dtype=np.float64)
    train_indices = np.asarray(
        archive["train_indices"], dtype=np.int64
    )
    test_indices = np.asarray(
        archive["test_indices"], dtype=np.int64
    )
    split_seed = int(np.asarray(archive["split_seed"]).item())
    metadata = {
        key: np.asarray(archive[key])
        for key in ("dt", "production", "hill_exponent", "degradation")
        if key in archive
    }

    sequence_length = (
        train_raw.shape[1]
        if steps is None
        else min(max(int(steps), 2), train_raw.shape[1])
    )
    train_raw = train_raw[:, :sequence_length].copy()
    test_raw = test_raw[:, :sequence_length].copy()
    val_raw = val_raw[:, :sequence_length].copy()
    time = time[:sequence_length].copy()

    if noise_std:
        noise_rng = np.random.default_rng(split_seed + 1)
        # Preserve the old noise stream for train and the former test (now val).
        for split in (train_raw, val_raw, test_raw):
            split += noise_rng.normal(
                0.0, noise_std, size=split.shape
            ).astype(np.float32)

    arrays = normalize_splits(train_raw, val_raw, test_raw)
    indices = [
        train_indices, np.asarray(archive["val_indices"], dtype=np.int64), test_indices,
    ]
    split_sizes = [len(train_raw), len(val_raw), len(test_raw)]
    arrays.update(
        goodwin_time=time,
        goodwin_split_indices=np.concatenate(indices),
        goodwin_split_sizes=np.asarray(
            split_sizes, dtype=np.int64
        ),
        data_kind=np.asarray("goodwin"),
        **{f"goodwin_{key}": value for key, value in metadata.items()},
    )
    return arrays


def make_goodwin_data(
    *, goodwin_cache_dir: str | Path = "./data/goodwin/cache",
    goodwin_parameters: Optional[Dict[str, Any]] = None,
    steps: Optional[int] = None, noise_std: float = 0.0,
    n_train_traj: Optional[int] = None, n_val_traj: Optional[int] = None,
    n_test_traj: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Load cached Goodwin data or generate a parameter-matched three-split dataset."""
    if not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std must be finite and nonnegative")
    parameters = {**GOODWIN_DEFAULT_PARAMETERS, **(goodwin_parameters or {})}
    for key, value in (("train_trajectories", n_train_traj),
                       ("val_trajectories", n_val_traj), ("test_trajectories", n_test_traj)):
        if value is not None:
            parameters[key] = value
    unknown = parameters.keys() - GOODWIN_DEFAULT_PARAMETERS.keys()
    if unknown:
        raise ValueError(f"Unknown Goodwin generation parameters: {sorted(unknown)}")
    specification = {
        "schema": GOODWIN_CACHE_SCHEMA, "system": "goodwin",
        "parameters": parameters, "steps": steps, "noise_std": float(noise_std),
    }
    return _cached_generated_data(
        cache_dir=goodwin_cache_dir, specification=specification,
        generate=lambda: _normalize_goodwin_data(
            generate_goodwin_dataset(**parameters), steps, noise_std,
        ),
    )


REACHING_TARGET_CLUSTER_COUNT = 6


def _reaching_target_clusters(
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the fixed six deterministic clusters to actual target coordinates."""
    cluster_count = REACHING_TARGET_CLUSTER_COUNT
    if len(positions) < cluster_count:
        raise ValueError("Fewer target samples than fixed reaching target clusters")
    center = positions.mean(axis=0)
    centers = [positions[np.argmin(np.sum((positions - center) ** 2, axis=1))]]
    for _ in range(1, cluster_count):
        distance = np.min(
            np.stack([np.sum((positions - item) ** 2, axis=1) for item in centers]),
            axis=0,
        )
        centers.append(positions[int(np.argmax(distance))])
    centers = np.asarray(centers, dtype=np.float64)
    for _ in range(100):
        distance = np.sum(
            (positions[:, None, :] - centers[None, :, :]) ** 2, axis=2
        )
        labels = np.argmin(distance, axis=1)
        updated = centers.copy()
        for index in range(cluster_count):
            members = positions[labels == index]
            if len(members):
                updated[index] = members.mean(axis=0)
        if np.allclose(updated, centers, rtol=0.0, atol=1e-10):
            centers = updated
            break
        centers = updated
    angles = np.mod(
        np.arctan2(centers[:, 1] - centers[:, 1].mean(),
                   centers[:, 0] - centers[:, 0].mean()),
        2 * np.pi,
    )
    order = np.argsort(angles)
    remap = np.empty(cluster_count, dtype=np.int64)
    remap[order] = np.arange(cluster_count)
    return remap[labels], centers[order]


def make_reaching_data(
    *,
    reaching_data_path: str | Path,
    reaching_path_class: str = "unobstructed",
    reaching_selected_targets: Optional[Sequence[int]] = None,
    reaching_max_units: int = 192,
    reaching_train_fraction: float = 0.85,
    steps: Optional[int] = None,
    seed: int = 0,
    **_: Any,
) -> Dict[str, np.ndarray]:
    """Load the movement-aligned reaching baseline without interactive input.

    Eligible trials in the selected zero-based target-coordinate clusters are
    split independently within each target and path class. Neural units are
    sampled evenly, matching the EDMD benchmark, and cursor position/velocity
    are kept.
    """
    path = Path(reaching_data_path)
    if not path.exists():
        raise FileNotFoundError(f"Reaching dataset not found: {path}")

    path_class = str(reaching_path_class).strip().lower()
    if path_class not in {"unobstructed", "obstructed", "both"}:
        raise ValueError(
            "reaching_path_class must be 'unobstructed', 'obstructed', or 'both'"
        )
    if int(reaching_max_units) < 1:
        raise ValueError("reaching_max_units must be positive")
    train_fraction = float(reaching_train_fraction)
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("reaching_train_fraction must lie strictly between 0 and 1")

    with h5py.File(path, "r") as handle:
        if "baseline" not in handle:
            raise ValueError(
                "Processed reaching file has no /baseline group; rerun "
                "data/monkey_reaching/process_reaching.py --overwrite"
            )
        baseline = handle["baseline"]
        required = {"states", "target_position", "obstructed", "time"}
        missing = sorted(required.difference(baseline.keys()))
        if missing:
            raise KeyError(f"Reaching baseline is missing datasets: {missing}")

        target_positions = np.asarray(
            baseline["target_position"], dtype=np.float64
        )
        obstructed = np.asarray(baseline["obstructed"], dtype=bool)
        finite_targets = np.all(np.isfinite(target_positions), axis=1)
        if path_class == "both":
            eligible = np.flatnonzero(finite_targets)
        else:
            select_obstructed = path_class == "obstructed"
            eligible = np.flatnonzero(
                (obstructed == select_obstructed) & finite_targets
            )
        target_ids, target_centers = _reaching_target_clusters(
            target_positions[eligible]
        )
        grouped = {
            target: eligible[target_ids == target]
            for target in range(REACHING_TARGET_CLUSTER_COUNT)
        }

        path_values = (False, True) if path_class == "both" else (
            path_class == "obstructed",
        )
        grouped_by_path = {
            (target, is_obstructed): grouped[target][
                obstructed[grouped[target]] == is_obstructed
            ]
            for target in range(REACHING_TARGET_CLUSTER_COUNT)
            for is_obstructed in path_values
        }

        if reaching_selected_targets is None:
            selected_targets = [
                target for target in range(REACHING_TARGET_CLUSTER_COUNT)
                if all(
                    len(grouped_by_path[target, is_obstructed]) >= 3
                    for is_obstructed in path_values
                )
            ]
        else:
            selected_targets = list(dict.fromkeys(
                int(direction) for direction in reaching_selected_targets
            ))
        if not selected_targets:
            raise ValueError("At least one reaching target must be selected")
        invalid = [
            target for target in selected_targets
            if target < 0 or target >= REACHING_TARGET_CLUSTER_COUNT
        ]
        if invalid:
            raise ValueError(
                f"Selected reaching targets must lie in [0, {REACHING_TARGET_CLUSTER_COUNT - 1}]; "
                f"got {invalid}"
            )
        sparse = {
            (target, "obstructed" if is_obstructed else "unobstructed"): int(
                len(grouped_by_path[target, is_obstructed])
            )
            for target in selected_targets
            for is_obstructed in path_values
            if len(grouped_by_path[target, is_obstructed]) < 3
        }
        if sparse:
            raise ValueError(
                "Every selected reaching target/path class needs at least three trials; "
                f"available counts: {sparse}"
            )

        rng = np.random.default_rng(seed)
        split_indices: list[list[int]] = [[], []]
        split_targets: list[list[int]] = [[], []]
        split_path_classes: list[list[bool]] = [[], []]
        for target in selected_targets:
            for is_obstructed in path_values:
                indices = np.asarray(
                    grouped_by_path[target, is_obstructed], dtype=np.int64
                )
                chosen = rng.permutation(indices)
                train_count = max(1, int(np.floor(len(chosen) * train_fraction)))
                train_count = min(train_count, len(chosen) - 1)
                pieces = np.split(chosen, (train_count,))
                for split, piece in enumerate(pieces):
                    split_indices[split].extend(piece.tolist())
                    split_targets[split].extend([target] * len(piece))
                    split_path_classes[split].extend(
                        [is_obstructed] * len(piece)
                    )

        unit_count = int(baseline.attrs["neural_unit_count"])
        selected_unit_count = min(int(reaching_max_units), unit_count)
        unit_indices = np.linspace(
            0, unit_count - 1, selected_unit_count, dtype=np.int64
        )
        sequence_length = (
            baseline["states"].shape[1]
            if steps is None
            else min(max(int(steps), 2), baseline["states"].shape[1])
        )

        raw_splits = []
        for indices in split_indices:
            sorted_indices = sorted(indices)
            states = np.asarray(
                baseline["states"][sorted_indices], dtype=np.float32
            )
            # h5py requires increasing indices; restore the randomized order.
            position = {
                index: offset for offset, index in enumerate(sorted_indices)
            }
            states = states[
                [position[index] for index in indices], :sequence_length
            ]
            states = np.concatenate(
                (states[..., unit_indices], states[..., unit_count:unit_count + 4]),
                axis=-1,
            )
            if not np.isfinite(states).all():
                raise ValueError("Selected reaching states contain non-finite values")
            raw_splits.append(states)

        time = np.asarray(baseline["time"][:sequence_length], dtype=np.float64)
        bin_width = float(baseline.attrs["bin_width"])
        smooth_ms = float(baseline.attrs["smooth_ms"])

    cursor_splits = [states[..., -4:] for states in raw_splits]
    # Train reaching models on the complete measured state: neural firing
    # rates followed by cursor position and velocity. Normalization remains
    # train-only and is applied independently to every state coordinate.
    arrays = normalize_train_test(*raw_splits)
    arrays.update(
        reaching_train_cursor=cursor_splits[0],
        reaching_test_cursor=cursor_splits[1],
        reaching_neural_unit_count=np.asarray(
            selected_unit_count, dtype=np.int64
        ),
        reaching_time=time,
        reaching_bin_width=np.asarray(bin_width),
        reaching_smooth_ms=np.asarray(smooth_ms),
        reaching_path_class=np.asarray(path_class),
        reaching_target_cluster_count=np.asarray(
            REACHING_TARGET_CLUSTER_COUNT, dtype=np.int64
        ),
        reaching_target_cluster_centers=target_centers.astype(np.float32),
        reaching_selected_targets=np.asarray(
            selected_targets, dtype=np.int64
        ),
        # Compatibility aliases for existing visualization artifacts. These now
        # contain target-cluster labels, never angular direction bins.
        reaching_direction_bins=np.asarray(
            REACHING_TARGET_CLUSTER_COUNT, dtype=np.int64
        ),
        reaching_selected_directions=np.asarray(
            selected_targets, dtype=np.int64
        ),
        reaching_unit_indices=unit_indices,
        reaching_split_indices=np.asarray(
            [index for split in split_indices for index in split],
            dtype=np.int64,
        ),
        reaching_split_targets=np.asarray(
            [direction for split in split_targets for direction in split],
            dtype=np.int64,
        ),
        reaching_split_directions=np.asarray(
            [target for split in split_targets for target in split],
            dtype=np.int64,
        ),
        reaching_split_obstructed=np.asarray(
            [value for split in split_path_classes for value in split],
            dtype=bool,
        ),
        reaching_split_path_classes=np.asarray([
            "obstructed" if value else "unobstructed"
            for split in split_path_classes for value in split
        ]),
        reaching_split_sizes=np.asarray(
            [len(indices) for indices in split_indices], dtype=np.int64
        ),
        reaching_split_offsets=np.concatenate((
            np.zeros(1, dtype=np.int64),
            np.cumsum(
                [len(indices) for indices in split_indices], dtype=np.int64
            ),
        )),
        data_kind=np.asarray("reaching"),
    )
    return arrays


PAM_STATE_COLUMNS = ([f'position_{j}' for j in range(4)]
                     + [f'velocity_{j}' for j in range(4)]
                     + [f'observed_pressure_{j}_{side}' for j in range(4) for side in ('ago', 'antago')])

def _resample_pamy_segments(time, states, dt):
    valid = np.isfinite(time) & np.isfinite(states).all(axis=1)
    delta = np.diff(time)
    positive = delta[np.isfinite(delta) & (delta > 0)]
    if not len(positive):
        return
    breaks = (
        ~valid[:-1]
        | ~valid[1:]
        | ~np.isfinite(delta)
        | (delta <= 0)
        | (delta > 3.0 * np.median(positive))
    )
    edges = np.r_[0, np.flatnonzero(breaks) + 1, len(time)]
    for left, right in zip(edges[:-1], edges[1:]):
        if right - left < 2 or not valid[left]:
            continue
        local = time[left:right] - time[left]
        count = int(np.floor(local[-1] / dt + 1.0e-9)) + 1
        if count < 2:
            continue
        grid = np.arange(count, dtype=np.float64) * dt
        values = np.column_stack(
            [
                np.interp(grid, local, states[left:right, j])
                for j in range(states.shape[1])
            ]
        ).astype(np.float32)
        yield values, grid

def make_pam_data(
    *,
    pam_data_path,
    pam_behavior="fixed_movement_fast+fixed_movement_slow",
    pam_train_fraction=0.85,
    pam_sample_dt=0.01,
    pam_timestamp_scale=1e-09,
    pam_cache_dir=None,
    seed=0,
    steps=None,
    pam_window_size=100,
    **_,
):
    path = Path(pam_data_path).expanduser().resolve()
    stat = path.stat()
    behaviors = tuple(
        dict.fromkeys(
            (
                item.strip().lower()
                for item in re.split("[+,]", pam_behavior)
                if item.strip()
            )
        )
    )
    allowed = {
        "random_movement",
        "fixed_movement_fast",
        "fixed_movement_slow",
        "move_to_random_pressures",
        "reset",
    }
    if not behaviors or not set(behaviors) <= allowed:
        raise ValueError(f"Unknown PAMy behaviors: {pam_behavior!r}")
    if not 0 < pam_train_fraction < 1:
        raise ValueError("pam_train_fraction must lie between zero and one")
    if (
        not np.isfinite(pam_sample_dt)
        or pam_sample_dt <= 0
        or (not np.isfinite(pam_timestamp_scale))
        or (pam_timestamp_scale <= 0)
    ):
        raise ValueError(
            "PAMy sample dt and timestamp scale must be finite and positive"
        )
    if steps is not None:
        raise ValueError("Use pam_window_size instead of data.steps for PAMy")
    if (
        isinstance(pam_window_size, bool)
        or int(pam_window_size) != pam_window_size
        or pam_window_size < 2
    ):
        raise ValueError("pam_window_size must be an integer of at least two samples")
    pam_window_size = int(pam_window_size)
    specification = dict(
        system="pamy_windows",
        schema=2,
        source=str(path),
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        ctime_ns=stat.st_ctime_ns,
        behaviors=behaviors,
        train_fraction=pam_train_fraction,
        dt=pam_sample_dt,
        timestamp_scale=pam_timestamp_scale,
        seed=seed,
        window_size=pam_window_size,
    )

    def prepare():
        trajectories, times, identities, recording_ids, labels, clocks = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        names, strata = ([], {})
        with h5py.File(path, "r") as handle:
            for recording_index, key in enumerate(sorted(handle["recordings"])):
                recording = handle["recordings"][key]
                available = {
                    str(item.attrs["column_name"]): item
                    for item in recording["columns"].values()
                }
                missing = set(["iteration", *PAM_STATE_COLUMNS]) - available.keys()
                if missing:
                    raise ValueError(
                        f"PAMy recording {key} is missing {sorted(missing)}"
                    )
                iteration = np.asarray(available["iteration"], dtype=np.float64)
                if not np.isfinite(iteration).all() or np.any(np.diff(iteration) <= 0):
                    raise ValueError(
                        "PAMy iteration values must be finite and increasing"
                    )
                names.append(str(recording.attrs.get("source_name", key)))
                occupied = []
                for behavior in behaviors:
                    annotations = recording.get(f"annotations/{behavior}")
                    if annotations is None:
                        continue
                    for movement, (left, right) in enumerate(
                        np.asarray(annotations).reshape(-1, 2)
                    ):
                        start = int(np.searchsorted(iteration, left, side="left"))
                        stop = int(np.searchsorted(iteration, right, side="right"))
                        if stop - start < 2:
                            continue
                        if any(
                            (start < end and stop > begin for begin, end in occupied)
                        ):
                            raise ValueError(
                                "Overlapping selected PAMy movements would leak across splits"
                            )
                        occupied.append((start, stop))
                        states = np.column_stack(
                            [available[name][start:stop] for name in PAM_STATE_COLUMNS]
                        )
                        if "time_stamp" in available:
                            stamp = np.asarray(
                                available["time_stamp"][start:stop], dtype=np.longdouble
                            )
                            finite = stamp[np.isfinite(stamp)]
                            if not len(finite):
                                continue
                            time = np.asarray(
                                (stamp - finite[0]) * pam_timestamp_scale,
                                dtype=np.float64,
                            )
                            clock = "time_stamp"
                        else:
                            if "frequency" not in available:
                                raise ValueError(
                                    "PAMy needs timestamps or frequency to establish physical time"
                                )
                            rates = np.asarray(
                                available["frequency"][start:stop], dtype=np.float64
                            )
                            rates = rates[np.isfinite(rates) & (rates > 0)]
                            if not len(rates):
                                raise ValueError(
                                    "PAMy frequency has no finite positive rate"
                                )
                            time = (
                                iteration[start:stop] - iteration[start]
                            ) / np.median(rates)
                            clock = "iteration/median_frequency"
                        identity = f"{key}:{behavior}:{movement}"
                        for values, grid in _resample_pamy_segments(
                            time, states, pam_sample_dt
                        ):
                            trajectories.append(values)
                            times.append(grid)
                            identities.append(identity)
                            recording_ids.append(recording_index)
                            labels.append(behavior)
                            clocks.append(clock)
                            strata.setdefault((recording_index, behavior), set()).add(
                                identity
                            )
        rng = np.random.default_rng(seed)
        heldout = set()
        for groups in strata.values():
            shuffled = rng.permutation(sorted(groups))
            if len(shuffled) >= 2:
                train_count = min(
                    max(int(np.floor(len(shuffled) * pam_train_fraction)), 1),
                    len(shuffled) - 1,
                )
                heldout.update(shuffled[train_count:])
        indices = [
            [
                i
                for i, identity in enumerate(identities)
                if (identity in heldout) == split
            ]
            for split in (False, True)
        ]
        if not indices[0] or not indices[1]:
            raise ValueError(
                "PAMy needs at least two movements in a recording/behavior stratum for held-out data"
            )
        window_trajectories, window_times = ([], [])
        window_identities, window_recordings, window_labels, window_clocks = (
            [],
            [],
            [],
            [],
        )
        window_indices = [[], []]
        for split, chosen in enumerate(indices):
            for i in chosen:
                for start in range(
                    0, len(trajectories[i]) - pam_window_size + 1, pam_window_size
                ):
                    stop = start + pam_window_size
                    window_indices[split].append(len(window_trajectories))
                    window_trajectories.append(trajectories[i][start:stop])
                    clock = times[i][start:stop]
                    window_times.append(clock - clock[0])
                    window_identities.append(identities[i])
                    window_recordings.append(recording_ids[i])
                    window_labels.append(labels[i])
                    window_clocks.append(clocks[i])
        if any((not chosen for chosen in window_indices)):
            raise ValueError(
                "No complete PAMy windows in a split; reduce pam_window_size"
            )
        trajectories, times = (window_trajectories, window_times)
        identities, recording_ids = (window_identities, window_recordings)
        labels, clocks, indices = (window_labels, window_clocks, window_indices)
        flat = np.concatenate([trajectories[i] for i in indices[0]]).astype(np.float64)
        mean, std = (flat.mean(0).astype(np.float32), flat.std(0).astype(np.float32))
        std[std < 1e-06] = 1.0
        arrays = dict(mean=mean, std=std)
        for split, chosen in zip(("train", "test"), indices):
            sequences = np.stack(
                [((trajectories[i] - mean) / std).astype(np.float32) for i in chosen]
            )
            arrays.update(
                {
                    f"{split}_sequences": sequences,
                    f"{split}_x_t": sequences[:, :-1].reshape(
                        -1, len(PAM_STATE_COLUMNS)
                    ),
                    f"{split}_x_tp1": sequences[:, 1:].reshape(
                        -1, len(PAM_STATE_COLUMNS)
                    ),
                    f"pam_{split}_time": np.stack([times[i] for i in chosen]),
                    f"pam_{split}_trajectory_indices": np.asarray(
                        chosen, dtype=np.int64
                    ),
                    f"pam_{split}_behaviors": np.asarray([labels[i] for i in chosen]),
                }
            )
        arrays.update(
            fixed_sequence_batches=np.asarray(True),
            pam_dt=np.asarray(pam_sample_dt),
            pam_behavior=np.asarray("+".join(behaviors)),
            pam_state_columns=np.asarray(PAM_STATE_COLUMNS),
            pam_split_mode=np.asarray("movement"),
            pam_split_sizes=np.asarray([len(i) for i in indices]),
            pam_sequence_lengths=np.asarray(
                [max((len(trajectories[i]) for i in chosen)) for chosen in indices]
            ),
            pam_recording_names=np.asarray(names),
            pam_trajectory_recordings=np.asarray(recording_ids),
            pam_trajectory_behaviors=np.asarray(labels),
            pam_trajectory_interval_ids=np.asarray(identities),
            pam_trajectory_clocks=np.asarray(clocks),
            data_kind=np.asarray("pam"),
        )
        arrays["pam_window_size"] = np.asarray(pam_window_size)
        return arrays

    return _cached_generated_data(
        cache_dir=pam_cache_dir or path.parent / "cache",
        specification=specification,
        generate=prepare,
    )

def pamy_checkpoint_config(config):
    """Translate saved window-run metadata to the standard PAMy loader settings."""
    config = dict(config)
    if 'pam_window_size' not in config:
        raise ValueError('This PAMy checkpoint predates fixed windows; retrain with the current PAMy config')
    cache_dir = config.pop('pam_masked_cache_dir', None)
    if cache_dir is not None:
        config.setdefault('pam_cache_dir', cache_dir)
    for obsolete in ('pam_masked_trajectories', 'pam_max_snapshots', 'pam_val_fraction'):
        config.pop(obsolete, None)
    return config


def pamy_workspace_windows(points, candidates, count, *, grid_size=16):
    """Greedily cover occupied 3-D workspace cells with complete windows.

    points contains sampled end-effector positions for every held-out window.
    Each chosen window adds as many previously unseen spatial cells as possible;
    once covered, farthest-centroid selection supplies additional diverse windows.
    """
    lower = points.min(axis=(0, 1))
    span = np.maximum(points.max(axis=(0, 1)) - lower, 1.e-12)
    cells = np.clip(((points - lower) / span * grid_size).astype(np.int64),
                    0, grid_size - 1)
    cell_ids = (cells[..., 0] * grid_size + cells[..., 1]) * grid_size + cells[..., 2]
    occupied = {int(i): set(cell_ids[i].tolist()) for i in candidates}
    centers = points.mean(axis=1)
    center = points[candidates].mean(axis=(0, 1))
    distance = np.square(centers - center).sum(axis=-1)
    remaining = [int(i) for i in candidates]
    selected, covered = [], set()
    for _ in range(min(int(count), len(remaining))):
        index = max(remaining, key=lambda i: (len(occupied[i] - covered), distance[i]))
        selected.append(index)
        remaining.remove(index)
        covered.update(occupied[index])
        separation = np.square(centers - centers[index]).sum(axis=-1)
        distance = separation if len(selected) == 1 else np.minimum(distance, separation)
    return np.asarray(selected, dtype=np.int64)


def pamy_visualization_indices(arrays):
    """Choose up to four random holdouts per behavior, shared by every plot type."""
    rng = np.random.default_rng(0)
    labels = np.asarray(arrays["pam_test_behaviors"]).astype(str)
    selected = []
    for behavior in dict.fromkeys(labels.tolist()):
        candidates = np.flatnonzero(labels == behavior)
        selected.extend(
            rng.choice(candidates, size=min(4, len(candidates)), replace=False)
        )
    return np.asarray(selected, dtype=np.int64)

def pamy_visualization_arrays(arrays):
    """Select four held-out windows per behavior for the original shared plots."""
    chosen = pamy_visualization_indices(arrays)
    subset = dict(arrays)
    for key in (
        "test_sequences",
        "pam_test_time",
        "pam_test_behaviors",
        "pam_test_trajectory_indices",
    ):
        subset[key] = arrays[key][chosen]
    subset["pam_plot_selected"] = True
    return subset


def make_data(
    *,
    system: str = "lorenz",
    n_train_traj: Optional[int] = None,
    n_val_traj: Optional[int] = None,
    n_test_traj: Optional[int] = None,
    steps: Optional[int] = None,
    dt: float = 0.015,
    noise_std: float = 0.0,
    seed: int = 0,
    lorenz_sigma: float = 10.0,
    lorenz_rho: float = 28.0,
    lorenz_beta: float = 8.0 / 3.0,
    lorenz_transient: int = 1000,
    lorenz_cache_dir: str | Path = "./data/lorenz/cache",
    duffing_damping: float = 0.3,
    duffing_linear_stiffness: float = 1.0,
    duffing_cubic_stiffness: float = 1.0,
    duffing_transient: int = 0,
    bistable_lorenz_sigma: float = 10.0,
    bistable_lorenz_rho: float = 10.0,
    bistable_lorenz_beta: float = 8.0 / 3.0,
    multistable_rtol: float = 1e-9,
    multistable_atol: float = 1e-11,
    dysts_parameters: Optional[Dict[str, float]] = None,
    dysts_warmup_steps: int = 2000,
    dysts_perturbation_scale: float = 0.2,
    dysts_dt_scale: float = 1.0,
    dysts_method: str = "DOP853",
    dysts_rtol: float = 1.0e-12,
    dysts_atol: float = 1.0e-12,
    dysts_cache_dir: str | Path = "./data/dysts/cache",
    goodwin_cache_dir: str | Path = "./data/goodwin/cache",
    goodwin_parameters: Optional[Dict[str, Any]] = None,
    reaching_data_path: Optional[str | Path] = "./data/monkey_reaching/reaching_data.h5",
    reaching_path_class: str = "unobstructed",
    reaching_selected_targets: Optional[Sequence[int]] = None,
    reaching_max_units: int = 192,
    reaching_train_fraction: float = 0.85,
    # Accepted for checkpoints created before train/test-only splitting.
    reaching_val_fraction: Optional[float] = None,
    pam_data_path: Optional[str | Path] = "./data/PAMy2/pamy_data.h5",
    pam_behavior: str = "fixed_movement_fast+fixed_movement_slow",
    pam_window_size: int = 100,
    pam_sample_dt: float = 0.01,
    pam_timestamp_scale: float = 1.e-9,
    pam_cache_dir: Optional[str | Path] = None,
    pam_train_fraction: float = 0.85,
) -> Dict[str, np.ndarray]:
    if system.lower() in GOODWIN_NAMES:
        return make_goodwin_data(
            goodwin_cache_dir=goodwin_cache_dir, goodwin_parameters=goodwin_parameters,
            steps=steps, noise_std=noise_std, n_train_traj=n_train_traj,
            n_val_traj=n_val_traj, n_test_traj=n_test_traj,
        )
    n_train_traj = 128 if n_train_traj is None else n_train_traj
    n_val_traj = 12 if n_val_traj is None else n_val_traj
    n_test_traj = 12 if n_test_traj is None else n_test_traj
    multistable_key = canonical_system(system)
    if multistable_key in MULTISTABLE_NAMES:
        parameters = dict(lorenz_sigma=bistable_lorenz_sigma,
                          lorenz_rho=bistable_lorenz_rho,
                          lorenz_beta=bistable_lorenz_beta)
        count = 1000 if steps is None else int(steps)
        splits = [generate_multistable_sequences(
            multistable_key, n, steps=count, dt=dt, seed=seed + offset,
            noise_std=noise_std, rtol=multistable_rtol, atol=multistable_atol,
            **parameters,
        ) for n, offset in ((n_train_traj, 10), (n_val_traj, 20), (n_test_traj, 30))]
        arrays = normalize_splits(*splits)
        arrays.update(
            data_kind=np.asarray(multistable_key),
            system_display_name=np.asarray(MULTISTABLE_DISPLAY_NAMES[multistable_key]),
            stable_equilibria=system_specification(multistable_key, **parameters)[-1],
            system_parameters=np.asarray(json.dumps(parameters)),
            time=np.arange(count, dtype=np.float64) * dt,
        )
        return arrays
    dysts_key = DYSTS_ALIASES.get(system.lower(), system.lower())
    if dysts_key in DYSTS_SYSTEMS:
        synthetic_steps = 30_001 if steps is None else int(steps)
        selected_parameters = dict(DYSTS_SYSTEMS[dysts_key][1])
        if dysts_parameters:
            selected_parameters.update(dysts_parameters)
        cache_specification = {
            "schema": DYSTS_CACHE_SCHEMA,
            "system": dysts_key,
            "parameters": selected_parameters,
            "n_train_traj": int(n_train_traj),
            "n_val_traj": int(n_val_traj),
            "n_test_traj": int(n_test_traj),
            "steps": synthetic_steps,
            "warmup_steps": int(dysts_warmup_steps),
            "seed": int(seed),
            "perturbation_scale": float(dysts_perturbation_scale),
            "dt_scale": float(dysts_dt_scale),
            "method": str(dysts_method),
            "rtol": float(dysts_rtol),
            "atol": float(dysts_atol),
        }

        def generate() -> Dict[str, np.ndarray]:
            train_raw, val_raw, test_raw, metadata = generate_dysts_splits(
                dysts_key, n_train_traj, n_val_traj, n_test_traj,
                synthetic_steps, dysts_warmup_steps, seed,
                parameters=dysts_parameters,
                perturbation_scale=dysts_perturbation_scale,
                dt_scale=dysts_dt_scale,
                method=dysts_method, rtol=dysts_rtol, atol=dysts_atol,
            )
            generated = normalize_splits(train_raw, val_raw, test_raw)
            generated.update(
                dysts_system=np.asarray(metadata["class_name"]),
                dysts_parameters=np.asarray(json.dumps(
                    metadata["parameters"], sort_keys=True
                )),
                dysts_default_initial_condition=metadata["default_ic"],
                dysts_coordinate_std=metadata["coordinate_std"],
                dysts_dt=np.asarray(metadata["dt"]),
                dysts_time=np.arange(
                    synthetic_steps, dtype=np.float64
                ) * metadata["dt"],
                dysts_warmup_steps=np.asarray(dysts_warmup_steps),
                data_kind=np.asarray(dysts_key),
            )
            return generated

        return _cached_generated_data(
            cache_dir=dysts_cache_dir,
            specification=cache_specification,
            generate=generate,
        )
    if system.lower() in DUFFING_NAMES:
        synthetic_steps = 1000 if steps is None else int(steps)
        common = dict(
            steps=synthetic_steps, dt=dt, transient=duffing_transient,
            damping=duffing_damping,
            linear_stiffness=duffing_linear_stiffness,
            cubic_stiffness=duffing_cubic_stiffness,
            noise_std=noise_std,
        )
        arrays = normalize_splits(
            generate_duffing_sequences(n_traj=n_train_traj, seed=seed + 10, **common),
            generate_duffing_sequences(n_traj=n_val_traj, seed=seed + 20, **common),
            generate_duffing_sequences(n_traj=n_test_traj, seed=seed + 30, **common),
        )
        arrays.update(
            duffing_damping=np.asarray(duffing_damping),
            duffing_linear_stiffness=np.asarray(duffing_linear_stiffness),
            duffing_cubic_stiffness=np.asarray(duffing_cubic_stiffness),
            duffing_dt=np.asarray(dt),
            duffing_time=np.arange(synthetic_steps, dtype=np.float64) * dt,
            data_kind=np.asarray("duffing"),
        )
        return arrays
    if system.lower() in LORENZ_NAMES:
        synthetic_steps = 1000 if steps is None else int(steps)
        common = dict(
            steps=synthetic_steps, dt=dt, transient=lorenz_transient,
            sigma=lorenz_sigma, rho=lorenz_rho, beta=lorenz_beta,
            noise_std=noise_std,
        )
        specification = {
            "schema": LORENZ_CACHE_SCHEMA, "system": "lorenz",
            "n_train_traj": n_train_traj, "n_val_traj": n_val_traj,
            "n_test_traj": n_test_traj, "seed": seed, **common,
        }

        def generate() -> Dict[str, np.ndarray]:
            arrays = normalize_splits(
                generate_lorenz_sequences(n_traj=n_train_traj, seed=seed + 10, **common),
                generate_lorenz_sequences(n_traj=n_val_traj, seed=seed + 20, **common),
                generate_lorenz_sequences(n_traj=n_test_traj, seed=seed + 30, **common),
            )
            arrays.update(
                lorenz_sigma=np.asarray(lorenz_sigma),
                lorenz_rho=np.asarray(lorenz_rho),
                lorenz_beta=np.asarray(lorenz_beta),
                lorenz_dt=np.asarray(dt),
                lorenz_time=np.arange(synthetic_steps, dtype=np.float64) * dt,
                data_kind=np.asarray("lorenz"),
            )
            return arrays

        return _cached_generated_data(
            cache_dir=lorenz_cache_dir, specification=specification, generate=generate,
        )
    if system.lower() in REACHING_NAMES:
        if reaching_data_path is None:
            raise ValueError("reaching_data_path is required for reaching data")
        return make_reaching_data(**locals())
    if system.lower() in PAM_NAMES:
        if pam_data_path is None:
            raise ValueError("pam_data_path is required for PAM data")
        return make_pam_data(**locals())
    raise ValueError(
        "system must be 'goodwin', 'reaching', 'pam', "
        "'lorenz', 'duffing', 'lotka_volterra', "
        "'lorenz_bistable', or a supported Dysts system"
    )


# -----------------------------------------------------------------------------
# PyTorch datasets/loaders
# -----------------------------------------------------------------------------


class WindowDataset(Dataset):
    """Expose every valid fixed-horizon start from complete trajectories."""

    def __init__(self, sequences: np.ndarray, horizon: int) -> None:
        if horizon < 1 or sequences.shape[1] <= horizon:
            raise ValueError("horizon must satisfy 1 <= horizon < sequence length")
        self.sequences = torch.from_numpy(sequences).float()
        self.horizon = int(horizon)
        self.trajectory_count, self.sequence_length = sequences.shape[:2]
        self.windows_per_trajectory = self.sequence_length - self.horizon

    def __len__(self) -> int:
        return self.trajectory_count * self.windows_per_trajectory

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        trajectory, start = divmod(i, self.windows_per_trajectory)
        return {
            "sequence": self.sequences[
                trajectory, start : start + self.horizon + 1
            ],
        }


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int = 512,
    shuffle: bool,
    device: Optional[torch.device] = None,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    collate_fn=None,
) -> DataLoader:
    kw = dict(
        batch_size=batch_size, shuffle=shuffle, drop_last=False,
        num_workers=num_workers,
        pin_memory=device is not None and device.type == "cuda",
        collate_fn=collate_fn,
    )
    if num_workers:
        kw.update(persistent_workers=True, prefetch_factor=prefetch_factor)
    return DataLoader(dataset, **kw)


def sequence_lengths(sequences):
    if sequences.ndim != 3 or sequences.shape[1] < 2 or not np.isfinite(sequences).all():
        raise ValueError('Expected finite sequences with shape [trajectory,time,state] and at least two states')
    return np.full(len(sequences), sequences.shape[1], dtype=np.int64)



def complete_trajectory_summary(arrays: Dict[str, Any]) -> str:
    """Summarize full-trajectory length and physical sampling when available."""
    lengths = []
    weights = []
    for split in ("train", "val", "test"):
        key = f"{split}_sequences"
        if key not in arrays:
            continue
        sequences = np.asarray(arrays[key])
        if sequences.ndim < 2 or not len(sequences):
            continue
        lengths.append(float(sequence_lengths(sequences).mean()))
        weights.append(float(sequences.shape[0]))
    if not lengths:
        return "Complete trajectories: unavailable"

    average_samples = float(np.average(lengths, weights=weights))
    average_steps = average_samples - 1.0
    message = (
        "Complete trajectories: "
        f"average {average_samples:.1f} samples "
        f"({average_steps:.1f} transition steps)"
    )

    timestep = None
    for key in (
        "reaching_bin_width", "pam_dt", "lorenz_dt", "duffing_dt", "dysts_dt",
    ):
        if key in arrays:
            value = float(np.asarray(arrays[key]).reshape(()))
            if np.isfinite(value) and value > 0:
                timestep = value
                break
    if timestep is None:
        for key in ("reaching_time", "goodwin_time", "pam_test_time"):
            if key not in arrays:
                continue
            times = np.asarray(arrays[key])
            differences = np.diff(times, axis=-1).reshape(-1)
            if np.issubdtype(differences.dtype, np.timedelta64):
                differences = differences / np.timedelta64(1, "s")
            try:
                differences = differences.astype(np.float64)
            except (TypeError, ValueError):
                continue
            differences = differences[np.isfinite(differences) & (differences > 0)]
            if len(differences):
                timestep = float(differences.mean())
                break
    if timestep is not None:
        message += f"; average temporal step {timestep:.6g} s"
    return message
