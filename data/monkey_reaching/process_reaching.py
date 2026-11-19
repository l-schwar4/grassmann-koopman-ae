#!/usr/bin/env python3
"""Convert reaching NWB files into one compact, consistent HDF5 dataset."""

from __future__ import annotations

import argparse
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from pynwb import NWBHDF5IO
from scipy.ndimage import gaussian_filter1d


HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE / "reaching_data" / "sub-Jenkins"
DEFAULT_OUTPUT = HERE / "reaching_data.h5"
COPY_CHUNK = 250_000
PHASE_ORDER = ("priming", "cue", "execution")
PHASE_START_EVENT = {
    "priming": "target_presentation",
    "cue": "go_cue",
    "execution": "move_start",
}
PHASE_END_EVENT = {
    "priming": "go_cue",
    "cue": "move_start",
    "execution": "move_end",
}


def parse_phases(value: str) -> tuple[str, ...]:
    """Parse one contiguous phase interval in physiological time order."""
    text = value.strip().lower()
    if text == "all":
        return PHASE_ORDER
    phases = tuple(dict.fromkeys(
        item.strip() for item in text.split(",") if item.strip()
    ))
    if not phases or any(phase not in PHASE_ORDER for phase in phases):
        raise argparse.ArgumentTypeError(
            "--phases must use priming,cue,execution (or all)"
        )
    indices = [PHASE_ORDER.index(phase) for phase in phases]
    expected = list(range(indices[0], indices[-1] + 1))
    if indices != expected:
        raise argparse.ArgumentTypeError(
            "selected phases must be ordered and contiguous; ignoring an "
            "intermediate phase would introduce an artificial state jump"
        )
    return phases


def first_column(table: Any, names: tuple[str, ...]) -> str | None:
    return next((name for name in names if name in table.colnames), None)


def values_or_nan(table: Any, names: tuple[str, ...], count: int) -> np.ndarray:
    name = first_column(table, names)
    return (np.asarray(table[name].data[:], dtype=np.float64) if name
            else np.full(count, np.nan))


def array_2d(value: Any, width: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    try:
        return array.reshape(-1, width)
    except ValueError:
        return np.empty((0, width))


def write_ragged(group: h5py.Group, name: str, values: list[np.ndarray]) -> None:
    offsets = np.zeros(len(values) + 1, dtype=np.int64)
    for index, value in enumerate(values):
        offsets[index + 1] = offsets[index] + value.size
    flat = np.concatenate([value.ravel() for value in values]) if offsets[-1] else np.array([])
    group.create_dataset(f"{name}_offsets", data=offsets, compression="lzf")
    options = {"compression": "lzf"} if len(flat) else {}
    group.create_dataset(f"{name}_values", data=flat, **options)


def copy_cursor(source: Any, target: h5py.Group) -> None:
    count = int(source.data.shape[0])
    data = target.create_dataset(
        "data", shape=(count, 2), dtype=np.float32,
        chunks=(min(COPY_CHUNK, max(count, 1)), 2), compression="lzf",
    )
    timestamps = target.create_dataset(
        "timestamps", shape=(count,), dtype=np.float64,
        chunks=(min(COPY_CHUNK, max(count, 1)),), compression="lzf",
    )
    for left in range(0, count, COPY_CHUNK):
        right = min(count, left + COPY_CHUNK)
        data[left:right] = np.asarray(source.data[left:right, :2], dtype=np.float32)
        if source.timestamps is not None:
            timestamps[left:right] = np.asarray(
                source.timestamps[left:right], dtype=np.float64
            )
        else:
            timestamps[left:right] = (
                float(source.starting_time)
                + np.arange(left, right, dtype=np.float64) / float(source.rate)
            )


def cursor_at_events(
    source: Any, event_times: np.ndarray,
    known_timestamps: np.ndarray | None = None,
) -> np.ndarray:
    """Read nearest cursor samples for many events in one indexed operation."""
    if source.timestamps is not None:
        timestamps = (known_timestamps if known_timestamps is not None
                      else np.asarray(source.timestamps[:], dtype=np.float64))
        indices = np.searchsorted(timestamps, event_times)
    else:
        indices = np.rint(
            (event_times - float(source.starting_time)) * float(source.rate)
        ).astype(np.int64)
    indices = np.clip(indices, 0, source.data.shape[0] - 1)
    valid = np.isfinite(event_times)
    positions = np.full((len(event_times), 2), np.nan)
    unique, inverse = np.unique(indices[valid], return_inverse=True)
    if len(unique):
        sampled = np.asarray(source.data[unique, :2], dtype=np.float64)
        positions[valid] = sampled[inverse]
    return positions


def cursor_on_grid(
    source: Any, times: np.ndarray, known_timestamps: np.ndarray | None
) -> np.ndarray:
    """Interpolate a time block from one contiguous cursor read."""
    if source.timestamps is not None:
        timestamps = known_timestamps
        left = max(0, int(np.searchsorted(timestamps, times[0])) - 1)
        right = min(len(timestamps), int(np.searchsorted(timestamps, times[-1])) + 2)
        source_times = timestamps[left:right]
    else:
        rate = float(source.rate)
        origin = float(source.starting_time)
        left = max(0, int(np.floor((times[0] - origin) * rate)) - 1)
        right = min(source.data.shape[0],
                    int(np.ceil((times[-1] - origin) * rate)) + 2)
        source_times = origin + np.arange(left, right) / rate
    source_data = np.asarray(source.data[left:right, :2], dtype=np.float64)
    return np.column_stack([
        np.interp(times, source_times, source_data[:, dimension])
        for dimension in range(2)
    ])


def _infer_movement_interval(
    source: Any, trial_start: float, trial_stop: float, go_cue: float,
    known_timestamps: np.ndarray | None,
) -> tuple[float, float]:
    """Infer movement from sustained cursor speed within one trial."""
    if source.timestamps is not None:
        timestamps = known_timestamps
        left = int(np.searchsorted(timestamps, trial_start, side="left"))
        right = int(np.searchsorted(timestamps, trial_stop, side="right"))
        times = timestamps[left:right]
    else:
        rate = float(source.rate)
        origin = float(source.starting_time)
        left = max(0, int(np.floor((trial_start - origin) * rate)))
        right = min(source.data.shape[0], int(np.ceil((trial_stop - origin) * rate)) + 1)
        times = origin + np.arange(left, right, dtype=np.float64) / rate
    if right - left < 8:
        return np.nan, np.nan
    position = np.asarray(source.data[left:right, :2], dtype=np.float64)
    valid = np.isfinite(times) & np.all(np.isfinite(position), axis=1)
    times, position = times[valid], position[valid]
    if len(times) < 8 or times[-1] <= times[0]:
        return np.nan, np.nan
    dt = float(np.median(np.diff(times)))
    if not np.isfinite(dt) or dt <= 0:
        return np.nan, np.nan
    position = gaussian_filter1d(position, max(1.0, 0.01 / dt), axis=0, mode="nearest")
    speed = np.linalg.norm(np.gradient(position, times, axis=0), axis=1)
    search_start = go_cue if np.isfinite(go_cue) else trial_start
    search = np.flatnonzero(times >= search_start)
    if not len(search):
        return np.nan, np.nan
    peak_index = int(search[np.argmax(speed[search])])
    peak_speed = float(speed[peak_index])
    if not np.isfinite(peak_speed) or peak_speed <= 0:
        return np.nan, np.nan
    baseline = speed[times < search_start]
    baseline = baseline[np.isfinite(baseline)]
    noise = (float(np.median(baseline)) + 6.0 * float(np.median(
        np.abs(baseline - np.median(baseline))
    ))) if len(baseline) else 0.0
    threshold = max(0.12 * peak_speed, noise)
    active = speed >= threshold
    sustain = max(2, int(round(0.02 / dt)))
    active = np.convolve(active.astype(np.int8), np.ones(sustain, dtype=np.int8), mode="same") > 0
    before_peak = np.flatnonzero(active & (np.arange(len(times)) <= peak_index) & (times >= search_start))
    after_peak = np.flatnonzero(active & (np.arange(len(times)) >= peak_index))
    if not len(before_peak) or not len(after_peak):
        return np.nan, np.nan
    return float(times[before_peak[0]]), float(times[after_peak[-1]])


def repair_phase_events(
    arrays: dict[str, np.ndarray], cursor_source: Any,
    cursor_timestamps: np.ndarray | None, bin_width: float,
) -> np.ndarray:
    """Repair small clock-order errors and infer genuinely missing motion events."""
    start, stop = arrays["start_time"], arrays["stop_time"]
    target = arrays["target_presentation"]
    go = arrays["go_cue"]
    move = arrays["move_start"]
    end = arrays["move_end"]
    repaired = np.zeros((len(start), 4), dtype=np.uint8)

    # Some target and go timestamps differ by only one acquisition sample.
    jitter = np.isfinite(target) & np.isfinite(go) & (target > go) & (target - go <= bin_width)
    target[jitter] = go[jitter]
    repaired[jitter, 0] = 1

    valid_execution = (
        np.isfinite(move) & np.isfinite(end) & (start <= move)
        & (move < end) & (end <= stop)
    )
    for index in np.flatnonzero(~valid_execution & arrays["success"].astype(bool)):
        inferred_start, inferred_end = _infer_movement_interval(
            cursor_source, float(start[index]), float(stop[index]),
            float(go[index]), cursor_timestamps,
        )
        if not np.isfinite(move[index]) or not start[index] <= move[index] < stop[index]:
            if np.isfinite(inferred_start):
                move[index] = inferred_start
                repaired[index, 2] = 1
        if not np.isfinite(end[index]) or not move[index] < end[index] <= stop[index]:
            if np.isfinite(inferred_end) and inferred_end > move[index]:
                end[index] = inferred_end
                repaired[index, 3] = 1

    def median_positive(values: np.ndarray, fallback: float) -> float:
        values = values[np.isfinite(values) & (values > 0)]
        return float(np.median(values)) if len(values) else fallback

    priming_duration = median_positive(go - target, 0.5)
    cue_duration = median_positive(move - go, 0.35)
    execution_duration = median_positive(end - move, 0.4)
    for index in range(len(start)):
        if not np.isfinite(target[index]) or not start[index] <= target[index] <= stop[index]:
            target[index] = (go[index] - priming_duration if np.isfinite(go[index])
                             else start[index])
            repaired[index, 0] = 1
        if not np.isfinite(go[index]) or not target[index] <= go[index] <= stop[index]:
            if np.isfinite(move[index]) and np.isfinite(arrays["reaction_time"][index]):
                go[index] = move[index] - arrays["reaction_time"][index]
            else:
                go[index] = target[index] + priming_duration
            repaired[index, 1] = 1
        if not np.isfinite(move[index]) or not go[index] <= move[index] <= stop[index]:
            reaction = arrays["reaction_time"][index]
            move[index] = go[index] + (reaction if np.isfinite(reaction) and reaction > 0
                                       else cue_duration)
            repaired[index, 2] = 1
        if not np.isfinite(end[index]) or not move[index] < end[index] <= stop[index]:
            end[index] = min(stop[index], move[index] + execution_duration)
            repaired[index, 3] = 1

    arrays["reaction_time"] = move - go
    return repaired


def copy_units(units: Any, target: h5py.Group) -> None:
    """Copy the NWB ragged spike vector in large contiguous blocks."""
    spike_index = units["spike_times"]
    offsets = np.empty(len(units) + 1, dtype=np.int64)
    offsets[0] = 0
    offsets[1:] = np.asarray(spike_index.data[:], dtype=np.int64)
    source = spike_index.target.data
    total = int(offsets[-1])
    options = ({
        "chunks": (min(COPY_CHUNK, total),), "compression": "lzf"
    } if total else {})
    spikes = target.create_dataset(
        "spike_times", shape=(total,), dtype=np.float64, **options
    )
    for left in range(0, total, COPY_CHUNK):
        right = min(total, left + COPY_CHUNK)
        spikes[left:right] = np.asarray(source[left:right], dtype=np.float64)
    target.create_dataset("spike_offsets", data=offsets, compression="lzf")


def bin_unit(
    unit_index: int, spike_times: np.ndarray, start: float,
    bin_width: float, bin_count: int,
) -> tuple[int, np.ndarray]:
    indices = np.floor((spike_times - start) / bin_width).astype(np.int64)
    indices = indices[(indices >= 0) & (indices < bin_count)]
    values = np.bincount(indices, minlength=bin_count)
    return unit_index, np.minimum(values, np.iinfo(np.uint16).max).astype(np.uint16)


def write_binned_spikes(
    units: Any, session: h5py.Group, start: float, stop: float,
    bin_width: float, workers: int, cursor_source: Any,
    cursor_timestamps: np.ndarray | None,
) -> None:
    """Cache population spike counts once so EDMD need not re-bin each run."""
    bin_count = int(np.ceil((stop - start) / bin_width))
    unit_count = len(units)
    cached = session.create_group("binned_spikes")
    cached.attrs["start_time"] = start
    cached.attrs["bin_width"] = bin_width
    counts = cached.create_dataset(
        "counts", shape=(bin_count, unit_count), dtype=np.uint16,
        chunks=(min(8192, bin_count), min(32, unit_count)), compression="lzf",
    )
    cursor_position = cached.create_dataset(
        "cursor_position", shape=(bin_count, 2), dtype=np.float32,
        chunks=(min(8192, bin_count), 2), compression="lzf",
    )
    for left in range(0, bin_count, COPY_CHUNK):
        right = min(bin_count, left + COPY_CHUNK)
        centers = start + (np.arange(left, right) + 0.5) * bin_width
        cursor_position[left:right] = cursor_on_grid(
            cursor_source, centers, cursor_timestamps
        )
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Read NWB spike arrays on the main thread, then bin one bounded batch
        # in parallel. HDF5 output writes also remain on the main thread.
        for batch_start in range(0, unit_count, workers):
            batch_stop = min(unit_count, batch_start + workers)
            futures = [
                executor.submit(
                    bin_unit, unit_index,
                    np.asarray(units["spike_times"][unit_index], dtype=np.float64),
                    start, bin_width, bin_count,
                )
                for unit_index in range(batch_start, batch_stop)
            ]
            results = sorted((future.result() for future in as_completed(futures)),
                             key=lambda item: item[0])
            counts[:, batch_start:batch_stop] = np.column_stack(
                [values for _, values in results]
            )
            completed = batch_stop
            print(f"\r  Binning units: {completed}/{unit_count}",
                  end="\n" if completed == unit_count else "", flush=True)


def smooth_matrix(values: np.ndarray, sigma_bins: float) -> np.ndarray:
    """Gaussian-smooth every neural column in one compiled pass."""
    if sigma_bins <= 0:
        return values
    return gaussian_filter1d(
        values, sigma_bins, axis=0, mode="nearest", truncate=4.0
    )


def interpolate_uniform(
    values: np.ndarray, source_start: float, target_time: np.ndarray,
    bin_width: float,
) -> np.ndarray:
    """Vectorized linear interpolation on the uniformly binned cache."""
    position = (target_time - source_start) / bin_width
    left = np.clip(np.floor(position).astype(np.int64), 0, len(values) - 2)
    weight = np.clip(position - left, 0.0, 1.0)[:, None]
    return values[left] * (1.0 - weight) + values[left + 1] * weight


def write_aligned_baseline(
    handle: h5py.File, bin_width: float, smooth_ms: float,
    phases: tuple[str, ...], post_movement_ms: float,
) -> None:
    """Create a fixed-shape baseline for one contiguous selection of phases."""
    sessions = [handle["sessions"][key] for key in sorted(handle["sessions"])]
    common_units = min(len(session["units/spike_offsets"]) - 1
                       for session in sessions)
    unit_indices = np.vstack([
        np.linspace(0, len(session["units/spike_offsets"]) - 2,
                    common_units, dtype=np.int32)
        for session in sessions
    ])
    start_event = PHASE_START_EVENT[phases[0]]
    end_event = PHASE_END_EVENT[phases[-1]]
    post_movement_seconds = (
        float(post_movement_ms) / 1000.0 if phases[-1] == "execution" else 0.0
    )
    candidates: list[tuple[int, int, float, float, float]] = []
    durations: list[float] = []
    for session_index, session in enumerate(sessions):
        trials = session["trials"]
        trial_start = np.asarray(trials["start_time"], dtype=np.float64)
        trial_stop = np.asarray(trials["stop_time"], dtype=np.float64)
        phase_start = np.asarray(trials[start_event], dtype=np.float64)
        phase_end = np.asarray(trials[end_event], dtype=np.float64)
        phase_stop = phase_end + post_movement_seconds
        success = np.asarray(trials["success"], dtype=bool)
        reach_angle = np.asarray(trials["reach_angle"], dtype=np.float64)
        valid = np.flatnonzero(
            success
            & np.isfinite(trial_start) & np.isfinite(trial_stop)
            & np.isfinite(phase_start) & np.isfinite(phase_end)
            & np.isfinite(reach_angle)
            & (trial_start <= phase_start) & (phase_start < phase_end)
            & (phase_stop <= trial_stop)
            & (phase_stop - phase_start >= 4 * bin_width)
        )
        for trial_index in valid:
            duration = float(phase_stop[trial_index] - phase_start[trial_index])
            candidates.append((session_index, int(trial_index),
                               float(phase_start[trial_index]),
                               float(trial_stop[trial_index]),
                               float(phase_end[trial_index]
                                     - phase_start[trial_index])))
            durations.append(duration)

    if not candidates:
        raise ValueError(f"No successful trials contain phases {phases}")
    if post_movement_seconds > 0.0:
        # A start-aligned fixed window must end after the latest retained
        # movement, not after the earliest one. Retain the lower 95% of
        # movement-end offsets and reserve the requested tail after that
        # boundary. Earlier movements consequently have at least (rather than
        # exactly) the requested amount of post-movement data.
        end_offsets = np.asarray(
            [item[4] for item in candidates], dtype=np.float64
        )
        common_phase_end = float(np.quantile(end_offsets, 0.95))
        common_stop = common_phase_end + post_movement_seconds
    else:
        # Without a requested tail, reject only unusually short trials while
        # retaining a useful common duration for approximately 95 percent.
        common_phase_end = np.inf
        common_stop = float(np.quantile(durations, 0.05))
    sample_count = int(np.floor(common_stop / bin_width)) + 1
    if sample_count < 4:
        raise ValueError("Selected phases have no useful common aligned duration")
    relative_time = np.arange(sample_count, dtype=np.float64) * bin_width
    if post_movement_seconds > 0.0:
        retained = [
            item for item in candidates
            if item[4] <= common_phase_end + 1.0e-9
            and item[3] - item[2] + 1.0e-9 >= relative_time[-1]
        ]
    else:
        retained = [
            item for item in candidates
            if item[4] + 1.0e-9 >= relative_time[-1]
        ]

    baseline = handle.create_group("baseline")
    phase_text = ",".join(phases)
    baseline.attrs["description"] = (
        f"{phase_text} states aligned to {start_event} and stored at fixed duration"
    )
    baseline.attrs["phase"] = phase_text
    baseline.attrs["phases"] = phase_text
    baseline.attrs["alignment"] = start_event
    baseline.attrs["bin_width"] = bin_width
    baseline.attrs["smooth_ms"] = smooth_ms
    baseline.attrs["post_movement_ms"] = post_movement_seconds * 1000.0
    baseline.attrs["neural_unit_count"] = common_units
    baseline.attrs["candidate_trial_count"] = len(candidates)
    baseline.attrs["retained_trial_count"] = len(retained)
    baseline.attrs["state_layout"] = (
        "neural_rates[0:U], cursor_position[U:U+2], "
        "cursor_velocity[U+2:U+4]"
    )
    baseline.create_dataset("time", data=relative_time)
    baseline.create_dataset("unit_indices", data=unit_indices, compression="lzf")
    baseline.create_dataset(
        "phase_event_names",
        data=np.asarray(
            ["target_presentation", "go_cue", "move_start", "move_end"],
            dtype=h5py.string_dtype("utf-8"),
        ),
    )
    state_dimension = common_units + 4
    state_data = baseline.create_dataset(
        "states", shape=(len(retained), sample_count, state_dimension),
        dtype=np.float32, chunks=(1, sample_count, min(64, state_dimension)),
        compression="lzf",
    )
    session_ids = np.empty(len(retained), dtype=np.int32)
    trial_ids = np.empty(len(retained), dtype=np.int32)
    reach_angles = np.empty(len(retained), dtype=np.float32)
    obstructed = np.empty(len(retained), dtype=np.uint8)
    targets = np.empty((len(retained), 2), dtype=np.float32)
    phase_event_offsets = np.empty((len(retained), 4), dtype=np.float32)
    event_repaired = np.empty((len(retained), 4), dtype=np.uint8)
    sigma_bins = smooth_ms / (1000.0 * bin_width)
    margin = max(2, int(np.ceil(4.0 * sigma_bins)) + 2)
    event_names = ("target_presentation", "go_cue", "move_start", "move_end")
    for output_index, (
        session_index, trial_index, anchor, _, _
    ) in enumerate(retained):
        session = sessions[session_index]
        cached = session["binned_spikes"]
        trials = session["trials"]
        cache_start = float(cached.attrs["start_time"])
        first_position = (anchor - cache_start) / bin_width - 0.5
        last_position = (anchor + relative_time[-1] - cache_start) / bin_width - 0.5
        left = max(0, int(np.floor(first_position)) - margin)
        right = min(cached["counts"].shape[0],
                    int(np.ceil(last_position)) + margin + 2)
        source_time = cache_start + (np.arange(left, right) + 0.5) * bin_width
        rates = np.asarray(
            cached["counts"][left:right, unit_indices[session_index]],
            dtype=np.float64,
        ) / bin_width
        rates = smooth_matrix(rates, sigma_bins)
        cursor = np.asarray(cached["cursor_position"][left:right], dtype=np.float64)
        target_time = anchor + relative_time
        rates = interpolate_uniform(rates, float(source_time[0]), target_time, bin_width)
        cursor = interpolate_uniform(cursor, float(source_time[0]), target_time, bin_width)
        velocity = np.gradient(cursor, bin_width, axis=0)
        state_data[output_index] = np.column_stack((rates, cursor, velocity))
        session_ids[output_index] = session_index
        trial_ids[output_index] = trial_index
        reach_angles[output_index] = trials["reach_angle"][trial_index]
        obstructed[output_index] = trials["obstructed"][trial_index]
        targets[output_index] = trials["target_position"][trial_index]
        phase_event_offsets[output_index] = [
            float(trials[name][trial_index]) - anchor for name in event_names
        ]
        event_repaired[output_index] = trials["event_repaired"][trial_index]
        if (output_index + 1) % 100 == 0 or output_index + 1 == len(retained):
            print(f"\r  Writing aligned baseline: {output_index + 1}/{len(retained)}",
                  end="\n" if output_index + 1 == len(retained) else "", flush=True)
    for name, values in (
        ("session_index", session_ids), ("trial_index", trial_ids),
        ("reach_angle", reach_angles), ("obstructed", obstructed),
        ("target_position", targets), ("phase_event_offsets", phase_event_offsets),
        ("event_repaired", event_repaired),
    ):
        baseline.create_dataset(name, data=values, compression="lzf")


def convert_session(
    path: Path, session: h5py.Group, bin_width: float, workers: int
) -> None:
    with NWBHDF5IO(str(path), mode="r", load_namespaces=True) as io:
        nwb = io.read()
        if nwb.trials is None or nwb.units is None:
            raise ValueError(f"{path.name} has no trials or sorted units")
        trials = nwb.trials
        count = len(trials)
        session.attrs["source_name"] = path.name
        session.attrs["identifier"] = str(nwb.identifier)
        session.attrs["subject"] = str(getattr(nwb.subject, "subject_id", "unknown"))
        session.attrs["session_start_time"] = str(nwb.session_start_time)

        trial_group = session.create_group("trials")
        start = np.asarray(trials["start_time"].data[:], dtype=np.float64)
        stop = np.asarray(trials["stop_time"].data[:], dtype=np.float64)
        success_name = first_column(trials, ("task_success", "is_successful", "success"))
        success = (np.asarray(trials[success_name].data[:]).astype(bool)
                   if success_name else np.ones(count, dtype=bool))
        discard_name = first_column(trials, ("discard_trial",))
        if discard_name:
            success &= ~np.asarray(trials[discard_name].data[:]).astype(bool)
        arrays = {
            "start_time": start,
            "stop_time": stop,
            "success": success.astype(np.uint8),
            "target_presentation": values_or_nan(
                trials, ("target_presentation_time", "target_on_time"), count
            ),
            "move_start": values_or_nan(
                trials, ("move_begins_time", "movement_onset_time", "reach_time"), count
            ),
            "move_end": values_or_nan(
                trials, ("move_ends_time", "movement_end_time", "target_acquire_time"), count
            ),
            "go_cue": values_or_nan(trials, ("go_cue_time",), count),
            "reaction_time": values_or_nan(trials, ("reaction_time",), count),
            "target_size": values_or_nan(trials, ("target_size",), count),
        }
        target_name = first_column(
            trials, ("hit_target_position", "target_pos", "target_position")
        )
        targets = np.full((count, 2), np.nan)
        if target_name:
            for index in range(count):
                value = array_2d(trials[target_name][index], 2)
                if len(value):
                    targets[index] = value[0]
        trial_group.create_dataset("target_position", data=targets, compression="lzf")

        barrier_name = first_column(trials, ("barrier_info", "barrier_points"))
        target_list_name = first_column(trials, ("target_positions",))
        barrier_values = [
            array_2d(trials[barrier_name][index], 4)
            if barrier_name else np.empty((0, 4))
            for index in range(count)
        ]
        write_ragged(trial_group, "barriers", barrier_values)
        barrier_count_name = first_column(trials, ("maze_num_barriers",))
        barrier_counts = (
            np.asarray(trials[barrier_count_name].data[:], dtype=np.int16)
            if barrier_count_name
            else np.asarray([len(value) for value in barrier_values], dtype=np.int16)
        )
        trial_group.create_dataset(
            "maze_num_barriers", data=barrier_counts, compression="lzf"
        )
        trial_group.create_dataset(
            "obstructed", data=(barrier_counts > 0).astype(np.uint8),
            compression="lzf",
        )
        maze_condition_name = first_column(trials, ("maze_condition",))
        if maze_condition_name:
            trial_group.create_dataset(
                "maze_condition",
                data=np.asarray(trials[maze_condition_name].data[:]),
                compression="lzf",
            )
        write_ragged(trial_group, "all_targets", [
            array_2d(trials[target_list_name][index], 2)
            if target_list_name else np.empty((0, 2)) for index in range(count)
        ])

        position = nwb.processing["behavior"].data_interfaces["Position"]
        series_name = ("Cursor" if "Cursor" in position.spatial_series
                       else list(position.spatial_series)[0])
        cursor_source = position.spatial_series[series_name]
        cursor_timestamps = (np.asarray(cursor_source.timestamps[:], dtype=np.float64)
                             if cursor_source.timestamps is not None else None)
        repaired = repair_phase_events(
            arrays, cursor_source, cursor_timestamps, bin_width
        )
        for name, values in arrays.items():
            trial_group.create_dataset(name, data=values, compression="lzf")
        trial_group.create_dataset(
            "event_repaired", data=repaired, compression="lzf"
        )
        move_start_position = cursor_at_events(
            cursor_source, arrays["move_start"], cursor_timestamps
        )
        move_end_position = cursor_at_events(
            cursor_source, arrays["move_end"], cursor_timestamps
        )
        displacement = move_end_position - move_start_position
        reach_angle = np.arctan2(displacement[:, 1], displacement[:, 0])
        reach_angle[np.linalg.norm(displacement, axis=1) < 1e-12] = np.nan
        trial_group.create_dataset("reach_angle", data=reach_angle,
                                   compression="lzf")
        behavior = session.create_group("behavior")
        behavior.attrs["series_name"] = series_name
        copy_cursor(cursor_source, behavior.create_group("cursor"))
        copy_units(nwb.units, session.create_group("units"))
        finite_start = float(np.nanmin(start))
        finite_stop = float(np.nanmax(stop))
        write_binned_spikes(
            nwb.units, session, finite_start, finite_stop, bin_width, workers,
            cursor_source, cursor_timestamps,
        )


def convert(
    paths: list[Path], output: Path, bin_width: float, smooth_ms: float,
    workers: int, phases: tuple[str, ...], post_movement_ms: float,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs["format"] = "shenoy-reaching"
            handle.attrs["format_version"] = 5
            sessions = handle.create_group("sessions", track_order=True)
            for index, path in enumerate(paths):
                print(f"[{index + 1}/{len(paths)}] Processing {path.name} ...",
                      flush=True)
                convert_session(
                    path, sessions.create_group(f"{index:06d}"), bin_width, workers
                )
            # A compact global index combines sessions without copying their
            # large cursor or spike-count arrays.
            combined = handle.create_group("combined")
            combined.attrs["session_count"] = len(sessions)
            combined.attrs["bin_width"] = bin_width
            names = [session.attrs["source_name"] for session in sessions.values()]
            combined.create_dataset(
                "source_names",
                data=np.asarray(names, dtype=h5py.string_dtype("utf-8")),
            )
            trial_counts = [len(session["trials/start_time"])
                            for session in sessions.values()]
            combined.create_dataset(
                "trial_session",
                data=np.repeat(
                    np.arange(len(trial_counts), dtype=np.int32), trial_counts
                ),
                compression="lzf",
            )
            combined.create_dataset(
                "trial_index",
                data=np.concatenate([
                    np.arange(count, dtype=np.int32) for count in trial_counts
                ]),
                compression="lzf",
            )
            unit_counts = [len(session["units/spike_offsets"]) - 1
                           for session in sessions.values()]
            combined.create_dataset(
                "unit_session",
                data=np.repeat(
                    np.arange(len(unit_counts), dtype=np.int32), unit_counts
                ),
                compression="lzf",
            )
            combined.create_dataset(
                "unit_index",
                data=np.concatenate([
                    np.arange(count, dtype=np.int32) for count in unit_counts
                ]),
                compression="lzf",
            )
            print(f"Creating aligned {','.join(phases)} baseline ...", flush=True)
            write_aligned_baseline(
                handle, bin_width, smooth_ms, phases, post_movement_ms
            )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bin-ms", type=float, default=10.0,            # was 15
                        help="spike-count cache resolution")
    parser.add_argument("--smooth-ms", type=float, default=10.0,         # was 15
                        help="Gaussian smoothing sigma for baseline neural rates")
    parser.add_argument(
        "--phases", type=parse_phases, default=("execution",),
        help=("contiguous phases to retain: priming, cue, execution, all "
              "comma-separated contiguous range, or all; default: execution"),
    )
    parser.add_argument(
        "--post-movement-ms", type=float, default=0.0,
        help=(
            "time retained after movement end when the selected interval ends "
            "in execution; trials without enough recorded tail are excluded "
            "(default: 0)"
        ),
    )
    parser.add_argument("--workers", type=int,
                        default=min(24, os.cpu_count() or 1),
                        help="parallel spike-binning workers")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = sorted(args.data_dir.glob("*.nwb"))
    if not paths:
        raise SystemExit(f"No .nwb files found in {args.data_dir}")
    if args.bin_ms <= 0 or args.smooth_ms < 0 or args.post_movement_ms < 0:
        raise SystemExit(
            "--bin-ms must be positive; --smooth-ms and "
            "--post-movement-ms must be non-negative"
        )
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"{args.output} exists; pass --overwrite to replace it")
    convert(paths, args.output, args.bin_ms / 1000, args.smooth_ms,
            args.workers, args.phases, args.post_movement_ms)
    print(f"Saved {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
