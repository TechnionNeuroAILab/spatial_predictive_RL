"""Pickle loading, pivoting, centre parsing and JSON output.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


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


def load_pickle(
    path: str | Path, required_columns: Sequence[str] | None = None
) -> pd.DataFrame:
    """Load a pickled DataFrame and check it has `required_columns`."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    df = pd.read_pickle(p)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame in {p}, got {type(df)}")
    if required_columns:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"File {p} is missing required columns: {missing}")

    return df


def load_behavior_activity(
    data_dir: str | Path, period: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Behaviour, activity and their merge on timestep for one period."""
    data_dir = Path(data_dir)
    behavior = load_pickle(
        data_dir / f"df_behavior_{period}.pkl", required_columns=["timestep", "x", "y"]
    )
    activity = load_pickle(
        data_dir / f"df_activations_{period}.pkl",
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
