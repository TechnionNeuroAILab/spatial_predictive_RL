"""Parquet loading, pivoting, centre parsing and JSON output.

Tabular run products (behaviour, activations, Skaggs) are Parquet tables written
by :func:`save_frame`. Dicts of numpy arrays (hidden-state samples, replay
trajectories) are one Parquet file per dict, written by :func:`save_payload`.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_PAYLOAD_META = b"srl_payload"


def ensure_dir(path: str | Path) -> Path:
    """Create `path` (and parents) if needed and return it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, payload: dict) -> None:
    """Write `payload` as indented JSON (plain Python types only)."""
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def as_parquet_path(path: str | Path) -> Path:
    """Parquet path for a frame or payload. A ``.pkl`` suffix is rewritten."""
    p = Path(path)
    if p.suffix.lower() == ".pkl":
        return p.with_suffix(".parquet")
    return p


def save_frame(path: str | Path, df: pd.DataFrame) -> Path:
    """Write a DataFrame as Parquet without the index."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(df)}")
    dest = as_parquet_path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".partial")
    try:
        df.to_parquet(tmp, index=False, compression="snappy")
        os.replace(tmp, dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return dest


def load_frame(
    path: str | Path, required_columns: Sequence[str] | None = None
) -> pd.DataFrame:
    """Load a Parquet DataFrame and check it has `required_columns`."""
    p = as_parquet_path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    df = pd.read_parquet(p)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame in {p}, got {type(df)}")
    if required_columns:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"File {p} is missing required columns: {missing}")
    return df


def load_pickle(
    path: str | Path, required_columns: Sequence[str] | None = None
) -> pd.DataFrame:
    """Load a run DataFrame. Name kept for callers; the file is Parquet."""
    return load_frame(path, required_columns)


def _scalar_spec(value: object) -> dict:
    if value is None:
        return {"kind": "none"}
    if isinstance(value, (bool, np.bool_)):
        return {"kind": "bool", "value": bool(value)}
    if isinstance(value, (int, np.integer)):
        return {"kind": "int", "value": int(value)}
    if isinstance(value, (float, np.floating)):
        return {"kind": "float", "value": float(value)}
    if isinstance(value, str):
        return {"kind": "str", "value": value}
    if isinstance(value, tuple):
        return {"kind": "tuple", "value": [_json_scalar(v) for v in value]}
    raise TypeError(f"Unsupported payload value type: {type(value)}")


def _json_scalar(value: object):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, tuple):
        return [_json_scalar(v) for v in value]
    return value


def _list_array(values: np.ndarray) -> pa.Array:
    flat = np.ascontiguousarray(values).reshape(-1)
    if flat.size > np.iinfo(np.int32).max:
        raise ValueError(f"Array is too long to store as one Parquet list ({flat.size})")
    arrow = pa.array(flat, type=pa.from_numpy_dtype(flat.dtype))
    offsets = pa.array([0, flat.size], type=pa.int32())
    return pa.ListArray.from_arrays(offsets, arrow)


def save_payload(path: str | Path, payload: dict) -> Path:
    """Write a dict of scalars, tuples, and numpy arrays to one Parquet file.

    Each array is a length-1 list column of its raveled values. Shapes, dtypes,
    and scalars are stored in the schema metadata under ``srl_payload``.
    """
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict, got {type(payload)}")
    spec: dict[str, dict] = {}
    columns: dict[str, pa.Array] = {}
    for key, value in payload.items():
        if isinstance(value, np.ndarray):
            arr = np.ascontiguousarray(value)
            spec[key] = {"kind": "ndarray", "dtype": str(arr.dtype), "shape": list(arr.shape)}
            columns[key] = _list_array(arr)
        else:
            spec[key] = _scalar_spec(value)
    if not columns:
        columns["_present"] = pa.array([1], type=pa.int32())
    table = pa.table(columns)
    meta = dict(table.schema.metadata or {})
    meta[_PAYLOAD_META] = json.dumps(spec).encode()
    table = table.replace_schema_metadata(meta)
    dest = as_parquet_path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".partial")
    try:
        pq.write_table(table, tmp, compression="snappy")
        os.replace(tmp, dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return dest


def load_payload(path: str | Path) -> dict:
    """Inverse of :func:`save_payload`."""
    p = as_parquet_path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    table = pq.read_table(p)
    meta = table.schema.metadata or {}
    if _PAYLOAD_META not in meta:
        raise ValueError(f"{p} has no srl_payload metadata")
    spec = json.loads(meta[_PAYLOAD_META])
    out: dict = {}
    for key, info in spec.items():
        kind = info["kind"]
        if kind == "ndarray":
            values = table.column(key).combine_chunks().values
            flat = values.to_numpy(zero_copy_only=False)
            out[key] = np.asarray(flat, dtype=info["dtype"]).reshape(info["shape"])
        elif kind == "tuple":
            out[key] = tuple(info["value"])
        elif kind == "none":
            out[key] = None
        elif kind == "int":
            out[key] = int(info["value"])
        elif kind == "float":
            out[key] = float(info["value"])
        elif kind == "bool":
            out[key] = bool(info["value"])
        elif kind == "str":
            out[key] = str(info["value"])
        else:
            raise ValueError(f"Unknown payload kind {kind!r} in {p}")
    return out


def load_behavior_activity(
    data_dir: str | Path, period: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Behaviour, activity and their merge on timestep for one period."""
    data_dir = Path(data_dir)
    behavior = load_frame(
        data_dir / f"df_behavior_{period}.parquet", required_columns=["timestep", "x", "y"]
    )
    activity = load_frame(
        data_dir / f"df_activations_{period}.parquet",
        required_columns=["timestep", "neuron_id", "activity"],
    )
    merged = pd.merge(activity, behavior, on="timestep", how="inner")
    if merged.empty:
        raise ValueError(f"Merged DataFrame is empty for period '{period}'.")

    return behavior, activity, merged


def pivot_activity_matrix(
    activity_df: pd.DataFrame,
    behavior_df: pd.DataFrame | None = None,
    fill_value: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """(timestep x neuron) activity matrix, optionally aligned to behaviour; missing entries become fill_value."""
    X_df = activity_df.pivot(index="timestep", columns="neuron_id", values="activity")
    X_df = X_df.sort_index().fillna(fill_value)

    if behavior_df is None:
        return X_df, None
    # behaviour is already one row per timestep; .first() collapses any repeats
    behavior_aligned = behavior_df.groupby("timestep")[["x", "y"]].first().sort_index()
    # keep only timesteps present in both, so the two frames line up row-for-row
    common = X_df.index.intersection(behavior_aligned.index)
    if len(common) == 0:
        raise ValueError("No common timesteps between activity and behavior matrices.")

    return X_df.loc[common], behavior_aligned.loc[common]


def center_to_xy(center: object) -> tuple[float, float] | None:
    """Parse one stored field centre into an (x, y) pair, or None if it is unusable."""
    if center is None:
        return None
    try:
        arr = np.asarray(center, dtype=float)
    except Exception:
        return None

    if arr.ndim == 1 and arr.size == 2:
        return float(arr[0]), float(arr[1])

    # multi-component fit: take the first component
    if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] >= 1:
        return float(arr[0, 0]), float(arr[0, 1])

    return None


def extract_single_centers(
    centers_df: pd.DataFrame,
    k_value: int | None = 1,
    icl_threshold: float | None = -5000.0,
    neuron_col: str = "neuron_id",
    center_col: str = "center",
    k_col: str = "k",
) -> pd.DataFrame:
    """One (x, y) centre per neuron from a GMM centre table, filtered by k and ICL score."""
    df = centers_df.copy()
    if k_value is not None and k_col in df.columns:
        df = df[df[k_col] == k_value]
    if icl_threshold is not None and "icl_score" in df.columns:
        df = df[df["icl_score"] <= icl_threshold]
    rows: list[dict] = []
    for _, row in df.iterrows():
        xy = center_to_xy(row.get(center_col))
        if xy is None:
            continue
        rows.append({neuron_col: row[neuron_col], "x": xy[0], "y": xy[1]})

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # order-dependent: a neuron with several surviving rows keeps its first
    out = out.drop_duplicates(subset=[neuron_col], keep="first")
    return out
