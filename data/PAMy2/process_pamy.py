#!/usr/bin/env python3
"""Combine PAMY2 pandas pickles into one sliceable HDF5 container.

Pickles are opened sequentially because they cannot be streamed. Only load
source pickle files that you trust. Columns and behavior annotations are
stored without interpolation or changes to their sampling times.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import quote

import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE / "pam_data_1-6"
DEFAULT_DATA = HERE / "pamy_data.h5"
BEHAVIOR_KEYS = (
    "random_movement",
    "fixed_movement_fast",
    "fixed_movement_slow",
    "move_to_random_pressures",
    "reset",
)


def natural_key(path: Path) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.name)]


def clean_intervals(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0, 2), dtype=np.float64)
    try:
        items = list(value)
        if len(items) == 2 and all(np.isscalar(item) for item in items):
            items = [items]
        result = []
        for item in items:
            if len(item) >= 2:
                left, right = float(item[0]), float(item[1])
                if np.isfinite(left) and np.isfinite(right):
                    result.append((min(left, right), max(left, right)))
        return np.asarray(result, dtype=np.float64).reshape(-1, 2)
    except (TypeError, ValueError):
        return np.empty((0, 2), dtype=np.float64)


def write_column(group: h5py.Group, number: int, name: str,
                 series: pd.Series) -> None:
    key = f"{number:04d}_{quote(name, safe='')}"
    values = series.to_numpy(copy=False)
    if values.dtype.kind not in "biufc":
        converted = pd.to_numeric(series, errors="coerce")
        if converted.notna().sum() == series.notna().sum():
            values = converted.to_numpy(dtype=np.float64, na_value=np.nan)
        else:
            values = series.fillna("").astype(str).to_numpy(dtype=object)

    compression = {} if len(values) == 0 else {
        "compression": "gzip", "compression_opts": 1
    }
    if values.dtype.kind in "OUS":
        dataset = group.create_dataset(
            key, data=values, dtype=h5py.string_dtype("utf-8"), **compression,
        )
    else:
        compression = {} if len(values) == 0 else {
            "compression": "lzf", "shuffle": True
        }
        dataset = group.create_dataset(
            key, data=values, **compression,
        )
    dataset.attrs["column_name"] = name


def convert(paths: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs["format"] = "pamy2-dataframes"
            handle.attrs["format_version"] = 1
            handle.attrs["source_count"] = len(paths)
            recordings = handle.create_group("recordings", track_order=True)
            for index, path in enumerate(paths):
                print(f"[{index + 1}/{len(paths)}] Converting {path.name} ...",
                      flush=True)
                frame = pd.read_pickle(path)
                try:
                    recording = recordings.create_group(f"{index:06d}")
                    recording.attrs["source_name"] = path.name
                    recording.attrs["row_count"] = len(frame)
                    recording.attrs["column_order"] = json.dumps(
                        [str(column) for column in frame.columns]
                    )
                    column_group = recording.create_group("columns", track_order=True)
                    for number, column in enumerate(frame.columns):
                        write_column(column_group, number, str(column),
                                     frame.iloc[:, number])
                    annotations = recording.create_group("annotations")
                    for behavior in BEHAVIOR_KEYS:
                        annotations.create_dataset(
                            behavior, data=clean_intervals(frame.attrs.get(behavior))
                        )
                finally:
                    del frame
                    gc.collect()
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = sorted(args.data_dir.glob("pamy_data_*.df"), key=natural_key)
    if not paths:
        raise SystemExit(f"No pamy_data_*.df files found in {args.data_dir}")
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"{args.output} exists; pass --overwrite to replace it")
    convert(paths, args.output)
    print(f"Saved {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
