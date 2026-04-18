"""
Shared helpers for the New_Code analysis scripts.
This module centralizes data loading, format validation, center parsing, small statistical helpers, and consistent output writing.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


DEFAULT_REWARD = np.array([0.5, 0.5], dtype=float)

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, payload: dict) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_pickle(path: str | Path, required_columns: Sequence[str] | None = None) -> pd.DataFrame:
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


def load_behavior_activity(data_dir: str | Path, period: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    behavior = load_pickle(data_dir / f"df_behavior_{period}.pkl",required_columns=["timestep", "x", "y"])
    activity = load_pickle(data_dir / f"df_activations_{period}.pkl",required_columns=["timestep", "neuron_id", "activity"])
    merged = pd.merge(activity, behavior, on="timestep", how="inner")
    if merged.empty:
        raise ValueError(f"Merged DataFrame is empty for period '{period}'.")

    return behavior, activity, merged


def pivot_activity_matrix(
    activity_df: pd.DataFrame,
    behavior_df: pd.DataFrame | None = None,
    fill_value: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    X_df = activity_df.pivot(index="timestep", columns="neuron_id", values="activity")
    X_df = X_df.sort_index().fillna(fill_value)

    if behavior_df is None:
        return X_df, None
    behavior_aligned = behavior_df.groupby("timestep")[["x", "y"]].first().sort_index()
    common = X_df.index.intersection(behavior_aligned.index)
    if len(common) == 0:
        raise ValueError("No common timesteps between activity and behavior matrices.")

    return X_df.loc[common], behavior_aligned.loc[common]


def center_to_xy(center: object) -> tuple[float, float] | None:
    if center is None:
        return None
    try:
        arr = np.asarray(center, dtype=float)
    except Exception:
        return None

    if arr.ndim == 1 and arr.size == 2:
        return float(arr[0]), float(arr[1])

    if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] >= 1:
        return float(arr[0, 0]), float(arr[0, 1])

    return None


def extract_single_centers(
    centers_df: pd.DataFrame,
    k_value: int | None = 1,
    neuron_col: str = "neuron_id",
    center_col: str = "center",
    k_col: str = "k",
) -> pd.DataFrame:
    df = centers_df.copy()
    if k_value is not None and k_col in df.columns:
        df = df[df[k_col] == k_value]
    rows: list[dict] = []
    for _, row in df.iterrows():
        xy = center_to_xy(row.get(center_col))
        if xy is None:
            continue
        rows.append({neuron_col: row[neuron_col], "x": xy[0], "y": xy[1]})

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.drop_duplicates(subset=[neuron_col], keep="first")
    return out


def compute_distance_to_reward(x: np.ndarray,y: np.ndarray,reward: np.ndarray | Sequence[float] = DEFAULT_REWARD,) -> np.ndarray:
    reward = np.asarray(reward, dtype=float)
    return np.sqrt((x - reward[0]) ** 2 + (y - reward[1]) ** 2)

def bin_positions(
    x: np.ndarray,
    y: np.ndarray,
    grid_size: int,
    clamp: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_bin = np.floor(x * grid_size).astype(int)
    y_bin = np.floor(y * grid_size).astype(int)

    if clamp:
        x_bin = np.clip(x_bin, 0, grid_size - 1)
        y_bin = np.clip(y_bin, 0, grid_size - 1)

    flat = x_bin * grid_size + y_bin
    return x_bin, y_bin, flat


def maybe_subsample_indices(n_rows: int, max_rows: int | None, seed: int = 42) -> np.ndarray:
    idx = np.arange(n_rows)
    if max_rows is None or n_rows <= max_rows:
        return idx

    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(idx, size=max_rows, replace=False))


def bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int = 2000,
    ci: float = 95.0,
    seed: int = 42,
) -> tuple[float, float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    n = vals.size
    for i in range(n_boot):
        sample = rng.choice(vals, size=n, replace=True)
        means[i] = np.mean(sample)

    alpha = (100.0 - ci) / 2.0
    low = np.percentile(means, alpha)
    high = np.percentile(means, 100.0 - alpha)
    return float(np.mean(vals)), float(low), float(high)


def cohen_d_independent(x: Iterable[float], y: Iterable[float]) -> float:
    x = np.asarray(list(x), dtype=float)
    y = np.asarray(list(y), dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return np.nan

    nx, ny = x.size, y.size
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if np.isclose(pooled, 0.0):
        return np.nan

    return float((np.mean(x) - np.mean(y)) / pooled)


def cohen_d_paired(x: Iterable[float], y: Iterable[float]) -> float:
    x = np.asarray(list(x), dtype=float)
    y = np.asarray(list(y), dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    d = x[mask] - y[mask]
    if d.size < 2:
        return np.nan
    sd = np.std(d, ddof=1)
    if np.isclose(sd, 0.0):
        return np.nan

    return float(np.mean(d) / sd)


def fdr_bh(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    adj = np.empty_like(ranked)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        adj[i] = prev
    out = np.empty_like(adj)
    out[order] = np.clip(adj, 0.0, 1.0)
    return out
