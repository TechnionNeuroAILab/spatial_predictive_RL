from __future__ import annotations
import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from scipy.optimize import curve_fit
from scipy.spatial import procrustes
from scipy.stats import gaussian_kde, mannwhitneyu, spearmanr
from tqdm import tqdm

from common import (
    DEFAULT_REWARD,
    cohen_d_independent,
    extract_single_centers,
    load_behavior_activity,
    load_pickle,
    pivot_activity_matrix,
    save_json,
)

@dataclass
class SeriesData:
    step: np.ndarray
    ori: np.ndarray
    procrustes: np.ndarray
    directional_shift: np.ndarray
    success: np.ndarray
    pred_mse: np.ndarray


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_goal_position(data_dir: Path, goal_file: str, reward_xy: np.ndarray) -> tuple[np.ndarray, str]:
    p = data_dir / goal_file
    if p.exists():
        arr = np.asarray(np.load(p), dtype=float).reshape(-1)
        if arr.size >= 2:
            return np.array([float(arr[0]), float(arr[1])], dtype=float), str(p)
    return reward_xy.copy(), "fallback_args"


def parse_numeric_from_name(name: str) -> int | None:
    m = re.search(r"(\d+)", name)
    if m:
        return int(m.group(1))
    return None


def first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def as_float_array(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def to_jsonable_float(value: float) -> float:
    return float(value) if np.isfinite(value) else np.nan


def safe_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else np.nan


def safe_std(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.std(x, ddof=1)) if x.size > 1 else np.nan


def safe_sem(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.std(x, ddof=1) / np.sqrt(x.size)) if x.size > 1 else np.nan


def percentile_ci(x: np.ndarray, low: float = 2.5, high: float = 97.5) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    return float(np.percentile(x, low)), float(np.percentile(x, high))


def aggregate_seed_curves(seed_series: dict[str, SeriesData]) -> dict[str, np.ndarray]:
    if not seed_series:
        empty = np.array([], dtype=float)
        return {
            "step": empty,
            "ori_mean": empty,
            "ori_std": empty,
            "procrustes_mean": empty,
            "procrustes_std": empty,
            "directional_shift_mean": empty,
            "directional_shift_std": empty,
            "success_mean": empty,
            "success_std": empty,
            "pred_mse_mean": empty,
            "pred_mse_std": empty,
            "ori_matrix": np.empty((0, 0), dtype=float),
            "pred_mse_matrix": np.empty((0, 0), dtype=float),
            "seed_names": [],
        }

    all_steps = sorted(set(np.concatenate([s.step for s in seed_series.values()]).astype(float).tolist()))
    step_arr = np.asarray(all_steps, dtype=float)
    n_seed = len(seed_series)
    n_step = step_arr.size

    def fill_matrix(getter: str) -> np.ndarray:
        mat = np.full((n_seed, n_step), np.nan, dtype=float)
        for r, (_, sd) in enumerate(seed_series.items()):
            values = getattr(sd, getter)
            step_map = {float(st): values[i] for i, st in enumerate(sd.step)}
            for c, st in enumerate(step_arr):
                if float(st) in step_map:
                    mat[r, c] = float(step_map[float(st)])
        return mat

    ori_mat = fill_matrix("ori")
    proc_mat = fill_matrix("procrustes")
    dsh_mat = fill_matrix("directional_shift")
    suc_mat = fill_matrix("success")
    mse_mat = fill_matrix("pred_mse")

    def mmean(mat: np.ndarray) -> np.ndarray:
        return np.array([safe_mean(mat[:, i]) for i in range(mat.shape[1])], dtype=float)

    def mstd(mat: np.ndarray) -> np.ndarray:
        return np.array([safe_std(mat[:, i]) for i in range(mat.shape[1])], dtype=float)

    return {
        "step": step_arr,
        "ori_mean": mmean(ori_mat),
        "ori_std": mstd(ori_mat),
        "procrustes_mean": mmean(proc_mat),
        "procrustes_std": mstd(proc_mat),
        "directional_shift_mean": mmean(dsh_mat),
        "directional_shift_std": mstd(dsh_mat),
        "success_mean": mmean(suc_mat),
        "success_std": mstd(suc_mat),
        "pred_mse_mean": mmean(mse_mat),
        "pred_mse_std": mstd(mse_mat),
        "ori_matrix": ori_mat,
        "pred_mse_matrix": mse_mat,
        "seed_names": list(seed_series.keys()),
    }

def load_centers_any(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".pkl":
        df = load_pickle(path)
        if {"neuron_id", "x", "y"}.issubset(df.columns):
            out = df[["neuron_id", "x", "y"]].copy()
            out["neuron_id"] = out["neuron_id"].astype(int)
            return out
        if {"neuron_id", "center"}.issubset(df.columns):
            out = extract_single_centers(df, k_value=1)
            if out.empty:
                return pd.DataFrame(columns=["neuron_id", "x", "y"])
            out["neuron_id"] = out["neuron_id"].astype(int)
            return out[["neuron_id", "x", "y"]]
        return pd.DataFrame(columns=["neuron_id", "x", "y"])

    if suffix == ".csv":
        df = pd.read_csv(path)
        if {"x", "y"}.issubset(df.columns):
            if "neuron_id" not in df.columns:
                df = df.copy()
                df["neuron_id"] = np.arange(df.shape[0], dtype=int)
            return df[["neuron_id", "x", "y"]].copy()
        return pd.DataFrame(columns=["neuron_id", "x", "y"])

    if suffix == ".npy":
        arr = np.asarray(np.load(path), dtype=float)
        if arr.ndim != 2:
            return pd.DataFrame(columns=["neuron_id", "x", "y"])
        if arr.shape[1] >= 3:
            return pd.DataFrame(
                {
                    "neuron_id": arr[:, 0].astype(int),
                    "x": arr[:, 1].astype(float),
                    "y": arr[:, 2].astype(float),
                }
            )
        if arr.shape[1] == 2:
            return pd.DataFrame(
                {
                    "neuron_id": np.arange(arr.shape[0], dtype=int),
                    "x": arr[:, 0].astype(float),
                    "y": arr[:, 1].astype(float),
                }
            )
        return pd.DataFrame(columns=["neuron_id", "x", "y"])

    return pd.DataFrame(columns=["neuron_id", "x", "y"])


def compute_ori_single_goal(centers: pd.DataFrame, goal: np.ndarray, goal_radius: float) -> dict[str, float]:
    if centers.empty:
        return {
            "n_total": 0,
            "n_goal": 0,
            "n_non_goal": 0,
            "ori": np.nan,
        }

    xy = centers[["x", "y"]].to_numpy(dtype=float)
    dist = np.sqrt((xy[:, 0] - goal[0]) ** 2 + (xy[:, 1] - goal[1]) ** 2)
    in_goal = dist <= goal_radius

    n_goal = int(np.sum(in_goal))
    n_non_goal = int(np.sum(~in_goal))
    n_total = int(xy.shape[0])

    f_goal = float(np.clip(np.pi * goal_radius * goal_radius, 1e-9, 1.0 - 1e-9))
    f_non = 1.0 - f_goal

    dens_goal = n_goal / (n_total * f_goal)
    dens_non = (n_non_goal / (n_total * f_non)) if n_non_goal > 0 else np.nan
    ori = dens_goal / dens_non if np.isfinite(dens_non) and dens_non > 0 else np.nan

    return {
        "n_total": n_total,
        "n_goal": n_goal,
        "n_non_goal": n_non_goal,
        "ori": float(ori),
    }


def compute_ori_dual_goal(centers: pd.DataFrame, goal1: np.ndarray, goal2: np.ndarray, goal_radius: float) -> dict[str, float]:
    if centers.empty:
        return {
            "n_total": 0,
            "n_goal_1": 0,
            "n_goal_2": 0,
            "n_goal_total": 0,
            "n_non_goal": 0,
            "ori_combined": np.nan,
            "ori_goal1": np.nan,
            "ori_goal2": np.nan,
        }

    xy = centers[["x", "y"]].to_numpy(dtype=float)
    d1 = np.sqrt((xy[:, 0] - goal1[0]) ** 2 + (xy[:, 1] - goal1[1]) ** 2)
    d2 = np.sqrt((xy[:, 0] - goal2[0]) ** 2 + (xy[:, 1] - goal2[1]) ** 2)

    z1 = d1 <= goal_radius
    z2 = d2 <= goal_radius
    zg = z1 | z2

    n1 = int(np.sum(z1))
    n2 = int(np.sum(z2))
    ng = int(np.sum(zg))
    nn = int(np.sum(~zg))
    nt = int(xy.shape[0])

    f = float(np.clip(np.pi * goal_radius * goal_radius, 1e-9, 1.0 - 1e-9))
    f_total = float(np.clip(2.0 * f, 1e-9, 1.0 - 1e-9))
    f_non_total = 1.0 - f_total

    dens_goal_total = ng / (nt * f_total)
    dens_non_total = (nn / (nt * f_non_total)) if nn > 0 else np.nan
    ori_combined = dens_goal_total / dens_non_total if np.isfinite(dens_non_total) and dens_non_total > 0 else np.nan
    # Per-goal ORI against all bins not in that specific goal zone.
    nn1 = nt - n1
    nn2 = nt - n2
    f_non_single = 1.0 - f

    ori1 = (n1 / (nt * f)) / (nn1 / (nt * f_non_single)) if nn1 > 0 else np.nan
    ori2 = (n2 / (nt * f)) / (nn2 / (nt * f_non_single)) if nn2 > 0 else np.nan

    return {
        "n_total": nt,
        "n_goal_1": n1,
        "n_goal_2": n2,
        "n_goal_total": ng,
        "n_non_goal": nn,
        "ori_combined": float(ori_combined),
        "ori_goal1": float(ori1),
        "ori_goal2": float(ori2),
    }


def permutation_goal_equality_test(
    n_total: int,
    n_goal_total: int,
    goal_radius: float,
    observed_abs_diff: float,
    n_perm: int,
    seed: int,
) -> tuple[float, np.ndarray]:
    if n_total <= 0 or n_goal_total <= 1:
        return np.nan, np.array([], dtype=float)

    rng = np.random.default_rng(seed)
    f = float(np.clip(np.pi * goal_radius * goal_radius, 1e-9, 1.0 - 1e-9))
    f_non_single = 1.0 - f

    null = np.zeros(n_perm, dtype=float)

    for i in tqdm(range(n_perm), desc="Permutation Test (Goal Equality)", leave=False):
        n1 = rng.binomial(n=n_goal_total, p=0.5)
        n2 = n_goal_total - n1

        nn1 = n_total - n1
        nn2 = n_total - n2
        if nn1 <= 0 or nn2 <= 0:
            null[i] = np.nan
            continue

        ori1 = (n1 / (n_total * f)) / (nn1 / (n_total * f_non_single))
        ori2 = (n2 / (n_total * f)) / (nn2 / (n_total * f_non_single))
        null[i] = abs(ori1 - ori2)

    null = null[np.isfinite(null)]
    if null.size == 0:
        return np.nan, null

    p = float(np.mean(null >= observed_abs_diff))
    return p, null


def bootstrap_ori_ci(
    centers: pd.DataFrame,
    goal: np.ndarray,
    goal_radius: float,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    if centers.empty:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    n = centers.shape[0]
    vals = np.zeros(n_boot, dtype=float)

    for i in tqdm(range(n_boot), desc="Bootstrap (ORI CI)", leave=False):
        idx = rng.choice(n, size=n, replace=True)
        c = centers.iloc[idx]
        vals[i] = compute_ori_single_goal(c, goal, goal_radius)["ori"]

    return percentile_ci(vals)


def load_baseline_centers(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    before_path = first_existing(
        [
            data_dir / "df_centers_before.pkl",
            data_dir / "place_field_centres_phase1.npy",
            data_dir / "centers_phase1.csv",
        ]
    )
    after_path = first_existing(
        [
            data_dir / "df_centers_after.pkl",
            data_dir / "place_field_centres_phase2.npy",
            data_dir / "centers_phase2.csv",
        ]
    )

    if before_path is None:
        before = pd.DataFrame(columns=["neuron_id", "x", "y"])
        src_b = "missing"
    else:
        before = load_centers_any(before_path)
        src_b = str(before_path)

    if after_path is None:
        after = pd.DataFrame(columns=["neuron_id", "x", "y"])
        src_a = "missing"
    else:
        after = load_centers_any(after_path)
        src_a = str(after_path)

    return before, after, src_b, src_a


def directional_shift_components(
    base_centers: pd.DataFrame,
    target_centers: pd.DataFrame,
    goal: np.ndarray,
) -> np.ndarray:
    if base_centers.empty or target_centers.empty:
        return np.array([], dtype=float)

    b = base_centers[["neuron_id", "x", "y"]].copy()
    t = target_centers[["neuron_id", "x", "y"]].copy()

    m = pd.merge(b, t, on="neuron_id", suffixes=("_b", "_t"))
    if m.empty:
        return np.array([], dtype=float)

    base_xy = m[["x_b", "y_b"]].to_numpy(dtype=float)
    target_xy = m[["x_t", "y_t"]].to_numpy(dtype=float)

    shift = target_xy - base_xy
    goal_vec = goal[None, :] - base_xy
    gnorm = np.linalg.norm(goal_vec, axis=1)

    valid = gnorm > 1e-9
    if not np.any(valid):
        return np.array([], dtype=float)

    u = goal_vec[valid] / gnorm[valid, None]
    comp = np.sum(shift[valid] * u, axis=1)
    return comp.astype(float)


def directional_shift_stat(
    base_centers: pd.DataFrame,
    target_centers: pd.DataFrame,
    goal: np.ndarray,
    n_perm: int,
    seed: int,
    alternative: str = "greater",
) -> dict[str, float | int]:
    comp = directional_shift_components(base_centers, target_centers, goal)
    if comp.size == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "perm_p": np.nan,
        }

    observed = float(np.mean(comp))

    rng = np.random.default_rng(seed)
    null = np.zeros(n_perm, dtype=float)
    for i in tqdm(range(n_perm), desc="Permutation Test (Shift)", leave=False):
        signs = rng.choice(np.array([-1.0, 1.0]), size=comp.size, replace=True)
        null[i] = float(np.mean(comp * signs))

    if alternative == "greater":
        p = float(np.mean(null >= observed))
    elif alternative == "less":
        p = float(np.mean(null <= observed))
    else:
        p = float(np.mean(np.abs(null) >= abs(observed)))

    return {
        "n": int(comp.size),
        "mean": observed,
        "perm_p": p,
    }


def load_pv_like(data_dir: Path, stem: str) -> tuple[np.ndarray, np.ndarray] | None:
    p = data_dir / stem
    if not p.exists():
        return None

    npz = np.load(p)
    if "positions" not in npz.files or "activations" not in npz.files:
        return None

    pos = np.asarray(npz["positions"], dtype=float)
    act = np.asarray(npz["activations"], dtype=float)

    if pos.ndim != 2 or pos.shape[1] < 2 or act.ndim != 2:
        return None

    n = min(pos.shape[0], act.shape[0])
    return pos[:n, :2], act[:n]


def load_period_matrix_from_pkl(data_dir: Path, period: str) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        beh, act_df, _ = load_behavior_activity(data_dir, period)
        X_df, B_df = pivot_activity_matrix(act_df, beh)
        if B_df is None:
            return None
        pos = B_df[["x", "y"]].to_numpy(dtype=float)
        act = X_df.to_numpy(dtype=float)
        n = min(pos.shape[0], act.shape[0])
        return pos[:n], act[:n]
    except Exception:
        return None


def build_bin_pc_matrix(
    positions: np.ndarray,
    projected: np.ndarray,
    grid_size: int,
    min_visits: int,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray]:
    K = grid_size * grid_size
    mat = np.full((K, n_components), np.nan, dtype=float)
    cnt = np.zeros(K, dtype=int)

    xb = np.clip(np.floor(positions[:, 0] * grid_size).astype(int), 0, grid_size - 1)
    yb = np.clip(np.floor(positions[:, 1] * grid_size).astype(int), 0, grid_size - 1)
    flat = xb * grid_size + yb

    for b in range(K):
        idx = np.where(flat == b)[0]
        cnt[b] = int(idx.size)
        if idx.size < min_visits:
            continue
        mat[b] = np.mean(projected[idx, :n_components], axis=0)

    return mat, cnt


def procrustes_disparity_against_phase1(
    data_dir: Path,
    candidate_npz: Path | None = None,
    grid_size: int = 16,
    min_visits: int = 3,
    n_components: int = 10,
) -> float:
    p1 = load_pv_like(data_dir, "pv_phase1.npz")
    if p1 is None:
        p1 = load_period_matrix_from_pkl(data_dir, "before")
    if p1 is None:
        return np.nan

    if candidate_npz is not None and candidate_npz.exists():
        cnpz = np.load(candidate_npz)
        if "positions" in cnpz.files and "activations" in cnpz.files:
            pos_t = np.asarray(cnpz["positions"], dtype=float)
            act_t = np.asarray(cnpz["activations"], dtype=float)
            n = min(pos_t.shape[0], act_t.shape[0])
            p2 = (pos_t[:n, :2], act_t[:n])
        else:
            p2 = load_pv_like(data_dir, "pv_phase2.npz")
    else:
        p2 = load_pv_like(data_dir, "pv_phase2.npz")
    if p2 is None:
        p2 = load_period_matrix_from_pkl(data_dir, "after")
    if p2 is None:
        return np.nan

    pos1, act1 = p1
    pos2, act2 = p2

    n_feat = min(act1.shape[1], act2.shape[1])
    act1 = act1[:, :n_feat]
    act2 = act2[:, :n_feat]

    act1c = act1 - np.mean(act1, axis=0, keepdims=True)
    act2c = act2 - np.mean(act2, axis=0, keepdims=True)

    from sklearn.decomposition import PCA

    pca = PCA(n_components=min(max(n_components, 20), act1c.shape[0], act1c.shape[1]), svd_solver="full")
    z1 = pca.fit_transform(act1c)
    z2 = pca.transform(act2c)

    nc = min(n_components, z1.shape[1], z2.shape[1])

    b1, c1 = build_bin_pc_matrix(pos1, z1, grid_size, min_visits, nc)
    b2, c2 = build_bin_pc_matrix(pos2, z2, grid_size, min_visits, nc)

    valid = (c1 >= min_visits) & (c2 >= min_visits)
    if np.sum(valid) < 8:
        return np.nan

    m1 = b1[valid]
    m2 = b2[valid]
    _, _, d = procrustes(m1, m2)
    return float(d)


def load_series_from_json(path: Path) -> SeriesData | None:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    # Dict of arrays format.
    if isinstance(payload, dict):
        step = None
        for key in ["epoch", "episode", "step", "epochs", "episodes", "steps"]:
            if key in payload:
                step = as_float_array(payload[key])
                break
        if step is None:
            return None

        def arr(keys: list[str]) -> np.ndarray:
            for k in keys:
                if k in payload:
                    return as_float_array(payload[k])
            return np.full(step.shape[0], np.nan, dtype=float)

        return SeriesData(
            step=step,
            ori=arr(["ori", "ORI"]),
            procrustes=arr(["procrustes", "procrustes_disparity", "disparity"]),
            directional_shift=arr(["directional_shift", "mean_directional_shift", "shift"]),
            success=arr(["success", "success_rate"]),
            pred_mse=arr(["pred_mse", "prediction_mse", "mse"]),
        )
    if isinstance(payload, list) and len(payload) > 0 and isinstance(payload[0], dict):
        df = pd.DataFrame(payload)
        return load_series_from_dataframe(df)

    return None


def load_series_from_dataframe(df: pd.DataFrame) -> SeriesData | None:
    step_col = None
    for c in ["epoch", "episode", "step"]:
        if c in df.columns:
            step_col = c
            break
    if step_col is None:
        return None

    step = df[step_col].to_numpy(dtype=float)

    def col(cands: list[str]) -> np.ndarray:
        for c in cands:
            if c in df.columns:
                return df[c].to_numpy(dtype=float)
        return np.full(step.shape[0], np.nan, dtype=float)

    return SeriesData(
        step=step,
        ori=col(["ori", "ORI"]),
        procrustes=col(["procrustes", "procrustes_disparity", "disparity"]),
        directional_shift=col(["directional_shift", "mean_directional_shift", "shift"]),
        success=col(["success", "success_rate"]),
        pred_mse=col(["pred_mse", "prediction_mse", "mse"]),
    )


def load_series_file(path: Path) -> SeriesData | None:
    if not path.exists():
        return None
    if path.suffix.lower() == ".json":
        return load_series_from_json(path)
    if path.suffix.lower() == ".csv":
        return load_series_from_dataframe(pd.read_csv(path))
    if path.suffix.lower() == ".npz":
        npz = np.load(path)
        if "step" not in npz.files:
            return None
        step = as_float_array(npz["step"])

        def arr(name: str) -> np.ndarray:
            return as_float_array(npz[name]) if name in npz.files else np.full(step.shape[0], np.nan, dtype=float)

        return SeriesData(
            step=step,
            ori=arr("ori"),
            procrustes=arr("procrustes"),
            directional_shift=arr("directional_shift"),
            success=arr("success"),
            pred_mse=arr("pred_mse"),
        )
    return None


def load_series_from_branch(branch_dir: Path) -> SeriesData | None:
    candidates = [
        branch_dir / "timeseries.json",
        branch_dir / "timeseries.csv",
        branch_dir / "trajectory.json",
        branch_dir / "trajectory.csv",
        branch_dir / "metrics_timeseries.json",
        branch_dir / "metrics_timeseries.csv",
        branch_dir / "series.npz",
    ]

    for p in candidates:
        s = load_series_file(p)
        if s is not None:
            return s

    return None


def load_seed_series(branch_root: Path) -> dict[str, SeriesData]:
    seeds: dict[str, SeriesData] = {}
    seed_dirs = sorted([p for p in branch_root.glob("seed_*") if p.is_dir()], key=lambda x: x.name)

    if len(seed_dirs) == 0:
        single = load_series_from_branch(branch_root)
        if single is not None:
            seeds["seed_0"] = single
        return seeds

    for sdir in seed_dirs:
        s = load_series_from_branch(sdir)
        if s is not None:
            seeds[sdir.name] = s

    return seeds


def first_center_file_in_dir(d: Path) -> Path | None:
    cands = [
        d / "df_centers.pkl",
        d / "df_centers_after.pkl",
        d / "centers.csv",
        d / "centers.npy",
        d / "place_field_centres.npy",
    ]
    return first_existing(cands)


def scan_epoch_center_series(branch_dir: Path) -> list[tuple[int, pd.DataFrame]]:
    if not branch_dir.exists() or not branch_dir.is_dir():
        return []
    series: list[tuple[int, pd.DataFrame]] = []
    for d in sorted([p for p in branch_dir.iterdir() if p.is_dir()], key=lambda x: x.name):
        num = parse_numeric_from_name(d.name)
        if num is None:
            continue
        cf = first_center_file_in_dir(d)
        if cf is None:
            continue
        centers = load_centers_any(cf)
        if centers.empty:
            continue
        series.append((num, centers))
    return series


def decay_epoch_within_tolerance(
    step: np.ndarray,
    values: np.ndarray,
    baseline_phase1: float,
    peak_phase2: float,
    tolerance_frac: float = 0.10,
) -> float:
    if step.size == 0:
        return np.nan

    delta = abs(peak_phase2 - baseline_phase1)
    if not np.isfinite(delta) or delta <= 1e-12:
        return np.nan

    tol = tolerance_frac * delta
    for i in range(step.size):
        if np.isfinite(values[i]) and abs(values[i] - baseline_phase1) <= tol:
            return float(step[i])
    return np.nan


def plot_reward_removal_decay(
    out_path: Path,
    agg: dict[str, np.ndarray],
    refs: dict[str, float],
) -> None:
    step = agg["step"]

    fig, ax1 = plt.subplots(figsize=(9.0, 5.2))

    ori_m = agg["ori_mean"]
    ori_s = agg["ori_std"]

    ax1.plot(step, ori_m, color="#2E5FA3", linewidth=2.3, label="ORI")
    ax1.fill_between(step, ori_m - ori_s, ori_m + ori_s, color="#2E5FA3", alpha=0.22)

    # Normalize Procrustes to ORI scale for visual overlay on the left axis.
    proc_m = agg["procrustes_mean"]
    proc_s = agg["procrustes_std"]

    p1 = refs.get("procrustes_phase1", 0.0)
    p2 = refs.get("procrustes_phase2", np.nan)
    o1 = refs.get("ori_phase1", np.nan)
    o2 = refs.get("ori_phase2", np.nan)

    def norm_proc(v: np.ndarray) -> np.ndarray:
        if not np.isfinite(p2) or not np.isfinite(p1) or abs(p2 - p1) < 1e-12:
            return np.full(v.shape, np.nan, dtype=float)
        return o1 + (v - p1) / (p2 - p1) * (o2 - o1)

    proc_mn = norm_proc(proc_m)
    proc_up = norm_proc(proc_m + proc_s)
    proc_dn = norm_proc(proc_m - proc_s)

    ax1.plot(step, proc_mn, color="#E07A2D", linewidth=2.0, label="Procrustes (normalized)")
    ax1.fill_between(step, proc_dn, proc_up, color="#E07A2D", alpha=0.20)

    ax1.axhline(refs.get("ori_phase1", np.nan), color="green", linestyle="--", linewidth=1.2, label="Phase 1 baseline")
    ax1.axhline(refs.get("ori_phase2", np.nan), color="red", linestyle="--", linewidth=1.2, label="Phase 2 peak")

    ax1.set_xlabel("Phase 3 epoch")
    ax1.set_ylabel("ORI / normalized Procrustes")

    ax2 = ax1.twinx()
    d_m = agg["directional_shift_mean"]
    d_s = agg["directional_shift_std"]
    ax2.plot(step, d_m, color="#1B8A5A", linewidth=2.0, label="Directional shift")
    ax2.fill_between(step, d_m - d_s, d_m + d_s, color="#1B8A5A", alpha=0.18)
    ax2.set_ylabel("Mean directional shift")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best", fontsize=9)

    ax1.set_title("Reward removal decay: ORI, Procrustes, directional shift")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_reward_removal_field_maps(
    out_path: Path,
    centers_p1: pd.DataFrame,
    centers_p2: pd.DataFrame,
    centers_p3_end: pd.DataFrame,
    goal: np.ndarray,
    goal_radius: float,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), sharex=True, sharey=True)
    panels = [
        (centers_p1, "Phase 1"),
        (centers_p2, "Phase 2"),
        (centers_p3_end, "Phase 3 end"),
    ]

    for ax, (c, title) in zip(axes, panels):
        if not c.empty:
            ax.scatter(c["x"], c["y"], s=12, alpha=0.45, color="#2E5FA3")
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.add_patch(Circle((goal[0], goal[1]), goal_radius, fill=False, linestyle="--", linewidth=1.6, color="red"))
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")

    axes[0].set_ylabel("y")
    fig.suptitle("Field-center comparison: baseline, peak, reward-removal end")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def analyze_reward_removal(
    data_dir: Path,
    branch_dir: Path,
    output_dir: Path,
    goal: np.ndarray,
    goal_radius: float,
    n_perm: int,
    seed: int,
) -> dict[str, Any]:
    d_out = ensure_dir(output_dir)
    fig_dir = ensure_dir(d_out / "figures")

    centers_p1, centers_p2, src_p1, src_p2 = load_baseline_centers(data_dir)

    ori_p1 = compute_ori_single_goal(centers_p1, goal, goal_radius)["ori"]
    ori_p2 = compute_ori_single_goal(centers_p2, goal, goal_radius)["ori"]

    proc_p1 = 0.0
    proc_p2 = procrustes_disparity_against_phase1(data_dir)

    dshift_p1 = 0.0
    dshift_p2 = directional_shift_stat(centers_p1, centers_p2, goal, n_perm=n_perm, seed=seed, alternative="greater")["mean"]

    seed_series = load_seed_series(branch_dir)

    # If no timeseries file is available, infer ORI/dshift from epoch center folders.
    if len(seed_series) == 0:
        epoch_centers = scan_epoch_center_series(branch_dir)
        if len(epoch_centers) > 0:
            step = np.array([e for e, _ in epoch_centers], dtype=float)
            ori = np.array([compute_ori_single_goal(c, goal, goal_radius)["ori"] for _, c in epoch_centers], dtype=float)
            dshift = np.array(
                [
                    directional_shift_stat(centers_p1, c, goal, n_perm=max(200, n_perm // 10), seed=seed + i, alternative="greater")["mean"]
                    for i, (_, c) in enumerate(epoch_centers)
                ],
                dtype=float,
            )
            seed_series["seed_0"] = SeriesData(
                step=step,
                ori=ori,
                procrustes=np.full(step.shape[0], np.nan, dtype=float),
                directional_shift=dshift,
                success=np.full(step.shape[0], np.nan, dtype=float),
                pred_mse=np.full(step.shape[0], np.nan, dtype=float),
            )

    agg = aggregate_seed_curves(seed_series)

    refs = {
        "ori_phase1": ori_p1,
        "ori_phase2": ori_p2,
        "procrustes_phase1": proc_p1,
        "procrustes_phase2": proc_p2,
        "directional_phase1": dshift_p1,
        "directional_phase2": dshift_p2,
    }

    decay_ori = decay_epoch_within_tolerance(agg["step"], agg["ori_mean"], ori_p1, ori_p2, tolerance_frac=0.10)
    decay_proc = decay_epoch_within_tolerance(agg["step"], agg["procrustes_mean"], proc_p1, proc_p2, tolerance_frac=0.10)
    decay_shift = decay_epoch_within_tolerance(
        agg["step"], agg["directional_shift_mean"], dshift_p1, dshift_p2, tolerance_frac=0.10
    )

    plot_reward_removal_decay(fig_dir / "reward_removal_decay_curves.png", agg=agg, refs=refs)
    end_centers = pd.DataFrame(columns=["neuron_id", "x", "y"])
    epoch_centers = scan_epoch_center_series(branch_dir)
    if len(epoch_centers) > 0:
        end_centers = epoch_centers[-1][1]
    plot_reward_removal_field_maps(
        fig_dir / "reward_removal_field_maps_comparison.png",
        centers_p1=centers_p1,
        centers_p2=centers_p2,
        centers_p3_end=end_centers,
        goal=goal,
        goal_radius=goal_radius,
    )

    results = {
        "script": "reward_manipulation.py",
        "analysis": "D1_reward_removal",
        "sources": {
            "baseline_phase1_centers": src_p1,
            "baseline_phase2_centers": src_p2,
            "branch_dir": str(branch_dir),
        },
        "references": refs,
        "trajectory": {
            "epoch": agg["step"].tolist(),
            "ori_mean": agg["ori_mean"].tolist(),
            "ori_sd": agg["ori_std"].tolist(),
            "procrustes_mean": agg["procrustes_mean"].tolist(),
            "procrustes_sd": agg["procrustes_std"].tolist(),
            "directional_shift_mean": agg["directional_shift_mean"].tolist(),
            "directional_shift_sd": agg["directional_shift_std"].tolist(),
        },
        "decay_to_within_10pct_of_phase1": {
            "ori_epoch": decay_ori,
            "procrustes_epoch": decay_proc,
            "directional_shift_epoch": decay_shift,
        },
    }

    save_json(d_out / "reward_removal_results.json", results)
    return results

def load_multi_reward_centers(branch_dir: Path) -> tuple[pd.DataFrame, str]:
    cands = [
        branch_dir / "df_centers_after.pkl",
        branch_dir / "df_centers.pkl",
        branch_dir / "field_centers.csv",
        branch_dir / "centers.npy",
    ]
    p = first_existing(cands)
    if p is None:
        return pd.DataFrame(columns=["neuron_id", "x", "y"]), "missing"
    return load_centers_any(p), str(p)


def multi_reward_rsa(branch_dir: Path, goal1: np.ndarray, goal2: np.ndarray, grid_size: int, min_visits: int) -> float:
    pv = first_existing([branch_dir / "pv_phase2.npz", branch_dir / "pv_multi_reward.npz"]) 
    if pv is None:
        return np.nan

    npz = np.load(pv)
    if "positions" not in npz.files or "activations" not in npz.files:
        return np.nan

    pos = np.asarray(npz["positions"], dtype=float)
    act = np.asarray(npz["activations"], dtype=float)
    n = min(pos.shape[0], act.shape[0])
    pos = pos[:n, :2]
    act = act[:n]

    xb = np.clip(np.floor(pos[:, 0] * grid_size).astype(int), 0, grid_size - 1)
    yb = np.clip(np.floor(pos[:, 1] * grid_size).astype(int), 0, grid_size - 1)
    flat = xb * grid_size + yb

    K = grid_size * grid_size
    N = act.shape[1]
    R = np.full((K, N), np.nan, dtype=float)
    valid = np.zeros(K, dtype=bool)

    for b in range(K):
        idx = np.where(flat == b)[0]
        if idx.size < min_visits:
            continue
        R[b] = np.mean(act[idx], axis=0)
        valid[b] = True

    idx = np.where(valid)[0]
    if idx.size < 8:
        return np.nan

    Rv = R[idx]
    from sklearn.metrics import pairwise_distances

    rdm = pairwise_distances(Rv, metric="correlation")

    centers = []
    for b in idx:
        i = b // grid_size
        j = b % grid_size
        centers.append([(i + 0.5) / grid_size, (j + 0.5) / grid_size])
    C = np.asarray(centers, dtype=float)

    md = np.minimum(
        np.sqrt((C[:, 0] - goal1[0]) ** 2 + (C[:, 1] - goal1[1]) ** 2),
        np.sqrt((C[:, 0] - goal2[0]) ** 2 + (C[:, 1] - goal2[1]) ** 2),
    )
    model = np.abs(md[:, None] - md[None, :])

    tri = np.triu_indices(rdm.shape[0], k=1)
    rho, _ = spearmanr(rdm[tri], model[tri])
    return float(rho)


def plot_multi_reward_density(
    out_path: Path,
    centers: pd.DataFrame,
    goal1: np.ndarray,
    goal2: np.ndarray,
    goal_radius: float,
) -> None:
    plt.figure(figsize=(6.2, 5.8))
    ax = plt.gca()

    if centers.shape[0] >= 5:
        xy = centers[["x", "y"]].to_numpy(dtype=float).T
        kde = gaussian_kde(xy)
        gx = np.linspace(0, 1, 32)
        gy = np.linspace(0, 1, 32)
        XX, YY = np.meshgrid(gx, gy, indexing="xy")
        ZZ = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
        im = ax.imshow(ZZ.T, origin="lower", extent=[0, 1, 0, 1], cmap="magma", aspect="equal")
        plt.colorbar(im, shrink=0.83, label="Field density (KDE)")
    else:
        ax.text(0.5, 0.5, "Insufficient centres for KDE", ha="center", va="center")

    ax.scatter(centers.get("x", pd.Series([], dtype=float)), centers.get("y", pd.Series([], dtype=float)), s=8, color="white", alpha=0.3)
    ax.add_patch(Circle((goal1[0], goal1[1]), goal_radius, fill=False, linestyle="--", linewidth=1.8, color="cyan"))
    ax.add_patch(Circle((goal2[0], goal2[1]), goal_radius, fill=False, linestyle="--", linewidth=1.8, color="lime"))
    ax.set_title("Two-reward place-field density")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_multi_reward_ori_bar(
    out_path: Path,
    ori1: float,
    ori2: float,
    ori_single: float,
    ci1: tuple[float, float],
    ci2: tuple[float, float],
    ci_single: tuple[float, float],
) -> None:
    labels = ["Goal 1", "Goal 2", "Single-reward"]
    vals = [ori1, ori2, ori_single]

    lows = [ori1 - ci1[0], ori2 - ci2[0], ori_single - ci_single[0]]
    highs = [ci1[1] - ori1, ci2[1] - ori2, ci_single[1] - ori_single]

    plt.figure(figsize=(6.8, 4.8))
    plt.bar(labels, vals, yerr=[lows, highs], capsize=5, color=["#2E5FA3", "#E07A2D", "#6D8F2A"], alpha=0.9)
    plt.axhline(1.0, linestyle="--", color="black", linewidth=1.2)
    plt.ylabel("ORI")
    plt.title("Multi-reward ORI comparison")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def analyze_multi_reward(
    data_dir: Path,
    branch_dir: Path,
    output_dir: Path,
    goal1: np.ndarray,
    goal2: np.ndarray,
    single_goal: np.ndarray,
    goal_radius: float,
    n_perm: int,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    d_out = ensure_dir(output_dir)
    fig_dir = ensure_dir(d_out / "figures")

    centers_multi, src_multi = load_multi_reward_centers(branch_dir)
    centers_p1, centers_p2, src_p1, src_p2 = load_baseline_centers(data_dir)

    single_ori_payload = compute_ori_single_goal(centers_p2, single_goal, goal_radius)
    single_ori = float(single_ori_payload["ori"])

    dual_payload = compute_ori_dual_goal(centers_multi, goal1, goal2, goal_radius)

    obs_diff = abs(dual_payload["ori_goal1"] - dual_payload["ori_goal2"]) if np.isfinite(dual_payload["ori_goal1"]) and np.isfinite(dual_payload["ori_goal2"]) else np.nan
    p_eq, null_eq = permutation_goal_equality_test(
        n_total=int(dual_payload["n_total"]),
        n_goal_total=int(dual_payload["n_goal_total"]),
        goal_radius=goal_radius,
        observed_abs_diff=obs_diff,
        n_perm=n_perm,
        seed=seed,
    )

    rsa_rho = multi_reward_rsa(branch_dir, goal1=goal1, goal2=goal2, grid_size=16, min_visits=3)

    ci_single = bootstrap_ori_ci(centers_p2, single_goal, goal_radius, n_boot=n_boot, seed=seed)
    ci_g1 = bootstrap_ori_ci(centers_multi, goal1, goal_radius, n_boot=n_boot, seed=seed + 1)
    ci_g2 = bootstrap_ori_ci(centers_multi, goal2, goal_radius, n_boot=n_boot, seed=seed + 2)

    plot_multi_reward_density(fig_dir / "multi_reward_field_density.png", centers_multi, goal1, goal2, goal_radius)
    plot_multi_reward_ori_bar(
        fig_dir / "multi_reward_ori_bar.png",
        ori1=dual_payload["ori_goal1"],
        ori2=dual_payload["ori_goal2"],
        ori_single=single_ori,
        ci1=ci_g1,
        ci2=ci_g2,
        ci_single=ci_single,
    )

    results = {
        "script": "reward_manipulation.py",
        "analysis": "D2_multi_reward",
        "sources": {
            "multi_reward_centers": src_multi,
            "single_reward_centers_phase2": src_p2,
            "single_reward_centers_phase1": src_p1,
            "branch_dir": str(branch_dir),
        },
        "goal_positions": {
            "goal1": goal1.tolist(),
            "goal2": goal2.tolist(),
            "single_reward_goal": single_goal.tolist(),
            "goal_radius": goal_radius,
        },
        "metrics": {
            "ORI_combined": dual_payload["ori_combined"],
            "ORI_goal1": dual_payload["ori_goal1"],
            "ORI_goal2": dual_payload["ori_goal2"],
            "ORI_single_reward": single_ori,
            "ORI_goal1_ci_95": [ci_g1[0], ci_g1[1]],
            "ORI_goal2_ci_95": [ci_g2[0], ci_g2[1]],
            "ORI_single_reward_ci_95": [ci_single[0], ci_single[1]],
            "goal1_vs_goal2_abs_diff": obs_diff,
            "goal1_vs_goal2_permutation_p": p_eq,
            "goal1_vs_goal2_null_mean": safe_mean(null_eq),
            "RSA_multi_reward_proximity_rho": rsa_rho,
            "n_goal_1": int(dual_payload["n_goal_1"]),
            "n_goal_2": int(dual_payload["n_goal_2"]),
            "n_non_goal": int(dual_payload["n_non_goal"]),
        },
    }

    save_json(d_out / "multi_reward_results.json", results)
    return results


def scan_checkpoint_center_files(branch_dir: Path) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    for p in branch_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".pkl", ".csv", ".npy"}:
            continue
        low = p.name.lower()
        if "center" not in low and "centre" not in low:
            continue
        ep = parse_numeric_from_name(str(p.parent.name))
        if ep is None:
            ep = parse_numeric_from_name(p.name)
        if ep is None:
            continue
        out.append((ep, p))

    out = sorted(out, key=lambda x: (x[0], 0 if x[1].suffix.lower() == ".pkl" else (1 if x[1].suffix.lower() == ".csv" else 2)))
    uniq: dict[int, Path] = {}
    for ep, p in out:
        if ep not in uniq:
            uniq[ep] = p
    return sorted([(ep, p) for ep, p in uniq.items()], key=lambda x: x[0])


def crossing_episode(step: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    if step.size == 0:
        return np.nan
    diff = b - a
    for i in range(diff.size):
        if np.isfinite(diff[i]) and diff[i] >= 0:
            return float(step[i])
    return np.nan


def plot_relocation_crossing(
    out_path: Path,
    step: np.ndarray,
    ori_old: np.ndarray,
    ori_new: np.ndarray,
    success: np.ndarray,
) -> None:
    fig, ax1 = plt.subplots(figsize=(8.8, 5.0))

    ax1.plot(step, ori_old, color="#2E5FA3", linewidth=2.2, label="ORI old goal")
    ax1.plot(step, ori_new, color="#E07A2D", linewidth=2.2, label="ORI new goal")
    ax1.set_xlabel("Relocation episode")
    ax1.set_ylabel("ORI")

    if np.any(np.isfinite(success)):
        ax2 = ax1.twinx()
        ax2.plot(step, success, color="#1B8A5A", linewidth=1.9, alpha=0.9, label="Success")
        ax2.set_ylabel("Success rate")

    h, l = ax1.get_legend_handles_labels()
    if "ax2" in locals():
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h + h2, l + l2, loc="best")
    else:
        ax1.legend(loc="best")

    ax1.set_title("Reward relocation: ORI old vs new goal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_relocation_vector_field(
    out_path: Path,
    base: pd.DataFrame,
    end: pd.DataFrame,
    goal_new: np.ndarray,
    goal_old: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.0))

    m = pd.merge(base[["neuron_id", "x", "y"]], end[["neuron_id", "x", "y"]], on="neuron_id", suffixes=("_b", "_e"))
    if m.empty:
        ax.text(0.5, 0.5, "No matched neurons", ha="center", va="center")
        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return

    xb = m["x_b"].to_numpy(dtype=float)
    yb = m["y_b"].to_numpy(dtype=float)
    u = m["x_e"].to_numpy(dtype=float) - xb
    v = m["y_e"].to_numpy(dtype=float) - yb
    # Color by alignment with new-goal direction.
    gvec = np.column_stack([goal_new[0] - xb, goal_new[1] - yb])
    gnorm = np.linalg.norm(gvec, axis=1)
    snorm = np.sqrt(u * u + v * v)
    dot = u * gvec[:, 0] + v * gvec[:, 1]
    align = np.divide(dot, np.maximum(gnorm * snorm, 1e-9), out=np.zeros_like(dot), where=(gnorm > 1e-9) & (snorm > 1e-9))

    q = ax.quiver(xb, yb, u, v, align, angles="xy", scale_units="xy", scale=1.0, cmap="coolwarm", width=0.004)
    plt.colorbar(q, shrink=0.8, label="Alignment to new-goal direction")

    ax.scatter([goal_old[0]], [goal_old[1]], marker="x", s=90, color="black", label="Old goal")
    ax.scatter([goal_new[0]], [goal_new[1]], marker="*", s=120, color="gold", edgecolors="black", label="New goal")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Relocation shift vector field")
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_relocation_field_comparison(
    out_path: Path,
    c_start: pd.DataFrame,
    c_mid: pd.DataFrame,
    c_end: pd.DataFrame,
    goal_old: np.ndarray,
    goal_new: np.ndarray,
    goal_radius: float,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), sharex=True, sharey=True)
    items = [
        (c_start, "After Phase 2"),
        (c_mid, "Mid relocation"),
        (c_end, "End relocation"),
    ]

    for ax, (c, title) in zip(axes, items):
        if not c.empty:
            ax.scatter(c["x"], c["y"], s=10, alpha=0.45, color="#2E5FA3")
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")

        ax.add_patch(Circle((goal_old[0], goal_old[1]), goal_radius, fill=False, linestyle="--", linewidth=1.4, color="black"))
        ax.add_patch(Circle((goal_new[0], goal_new[1]), goal_radius, fill=False, linestyle="--", linewidth=1.4, color="red"))
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")

    axes[0].set_ylabel("y")
    fig.suptitle("Field distributions across reward relocation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def analyze_reward_relocation(
    data_dir: Path,
    branch_dir: Path,
    output_dir: Path,
    old_goal: np.ndarray,
    new_goal: np.ndarray,
    goal_radius: float,
    n_perm: int,
    seed: int,
) -> dict[str, Any]:
    d_out = ensure_dir(output_dir)
    fig_dir = ensure_dir(d_out / "figures")

    _, centers_phase2, _, src_phase2 = load_baseline_centers(data_dir)

    ckpt_files = scan_checkpoint_center_files(branch_dir)

    step = []
    ori_old = []
    ori_new = []
    center_series: list[pd.DataFrame] = []

    for ep, fp in ckpt_files:
        c = load_centers_any(fp)
        if c.empty:
            continue
        step.append(float(ep))
        ori_old.append(compute_ori_single_goal(c, old_goal, goal_radius)["ori"])
        ori_new.append(compute_ori_single_goal(c, new_goal, goal_radius)["ori"])
        center_series.append(c)

    if len(step) == 0:
        ts = load_series_from_branch(branch_dir)
        if ts is not None:
            step = ts.step.tolist()
            ori_old = [np.nan] * len(step)
            ori_new = [np.nan] * len(step)

    step_arr = np.asarray(step, dtype=float)
    ori_old_arr = np.asarray(ori_old, dtype=float)
    ori_new_arr = np.asarray(ori_new, dtype=float)

    cross_ep = crossing_episode(step_arr, ori_old_arr, ori_new_arr)

    # Success trajectory if available.
    ts = load_series_from_branch(branch_dir)
    if ts is not None:
        success = np.asarray(ts.success, dtype=float)
        if success.size != step_arr.size:
            success = np.full(step_arr.shape, np.nan, dtype=float)
    else:
        success = np.full(step_arr.shape, np.nan, dtype=float)

    # Directional shift tests at end checkpoint.
    end_centers = center_series[-1] if len(center_series) > 0 else pd.DataFrame(columns=["neuron_id", "x", "y"])

    stat_new = directional_shift_stat(
        centers_phase2,
        end_centers,
        new_goal,
        n_perm=n_perm,
        seed=seed,
        alternative="greater",
    )
    stat_old = directional_shift_stat(
        centers_phase2,
        end_centers,
        old_goal,
        n_perm=n_perm,
        seed=seed + 1,
        alternative="less",
    )

    plot_relocation_crossing(fig_dir / "relocation_ori_crossing.png", step_arr, ori_old_arr, ori_new_arr, success)
    plot_relocation_vector_field(
        fig_dir / "relocation_vector_field.png",
        base=centers_phase2,
        end=end_centers,
        goal_new=new_goal,
        goal_old=old_goal,
    )

    if len(center_series) > 0:
        mid_centers = center_series[len(center_series) // 2]
    else:
        mid_centers = pd.DataFrame(columns=["neuron_id", "x", "y"])

    plot_relocation_field_comparison(
        fig_dir / "relocation_field_distribution_comparison.png",
        c_start=centers_phase2,
        c_mid=mid_centers,
        c_end=end_centers,
        goal_old=old_goal,
        goal_new=new_goal,
        goal_radius=goal_radius,
    )

    results = {
        "script": "reward_manipulation.py",
        "analysis": "D3_reward_relocation",
        "sources": {
            "phase2_centers": src_phase2,
            "branch_dir": str(branch_dir),
            "n_checkpoint_files": int(len(ckpt_files)),
        },
        "goal_positions": {
            "old_goal": old_goal.tolist(),
            "new_goal": new_goal.tolist(),
            "goal_radius": goal_radius,
        },
        "trajectory": {
            "episode": step_arr.tolist(),
            "ORI_old_goal": ori_old_arr.tolist(),
            "ORI_new_goal": ori_new_arr.tolist(),
            "success": success.tolist(),
            "crossing_episode": cross_ep,
        },
        "directional_shift": {
            "toward_new_goal_mean": stat_new["mean"],
            "toward_new_goal_perm_p": stat_new["perm_p"],
            "toward_old_goal_mean": stat_old["mean"],
            "toward_old_goal_perm_p": stat_old["perm_p"],
        },
        "behavior": {
            "success_start": float(success[0]) if success.size and np.isfinite(success[0]) else np.nan,
            "success_end": float(success[-1]) if success.size and np.isfinite(success[-1]) else np.nan,
        },
    }

    save_json(d_out / "reward_relocation_results.json", results)
    return results

def exp_decay(t: np.ndarray, tau: float, y0: float, yb: float) -> np.ndarray:
    return (y0 - yb) * np.exp(-t / np.maximum(tau, 1e-6)) + yb


def fit_tau(step: np.ndarray, ori: np.ndarray, baseline: float) -> float:
    mask = np.isfinite(step) & np.isfinite(ori)
    if np.sum(mask) < 3 or not np.isfinite(baseline):
        return np.nan

    t = np.asarray(step[mask], dtype=float)
    y = np.asarray(ori[mask], dtype=float)

    y0 = float(y[0])
    t0 = float(np.max([1.0, (t[-1] - t[0]) / 3.0]))

    try:
        popt, _ = curve_fit(
            lambda tt, tau: exp_decay(tt - t[0], tau=tau, y0=y0, yb=baseline),
            t,
            y,
            p0=[t0],
            bounds=(1e-3, 1e6),
            maxfev=10000,
        )
        return float(popt[0])
    except Exception:
        return np.nan


def bootstrap_tau_diff(tau_a: np.ndarray, tau_b: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    a = tau_a[np.isfinite(tau_a)]
    b = tau_b[np.isfinite(tau_b)]
    if a.size < 1 or b.size < 1:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    d = np.zeros(n_boot, dtype=float)
    for i in tqdm(range(n_boot), desc="Bootstrap (Tau Diff)", leave=False):
        ab = rng.choice(a, size=a.size, replace=True)
        bb = rng.choice(b, size=b.size, replace=True)
        d[i] = float(np.mean(bb) - np.mean(ab))

    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def select_obs_corruption_root(branch_root: Path, target_noise: float) -> tuple[Path, str]:
    cands = [
        branch_root / f"noise_{target_noise}",
        branch_root / f"noise_{target_noise:.1f}",
        branch_root / f"noise_{target_noise:.2f}",
    ]
    for c in cands:
        if c.exists() and c.is_dir():
            return c, c.name

    return branch_root, "root"


def plot_disruption_curves(
    out_path: Path,
    step_a: np.ndarray,
    ori_a_mean: np.ndarray,
    ori_a_sd: np.ndarray,
    mse_a_mean: np.ndarray,
    mse_a_sd: np.ndarray,
    step_b: np.ndarray,
    ori_b_mean: np.ndarray,
    ori_b_sd: np.ndarray,
    mse_b_mean: np.ndarray,
    mse_b_sd: np.ndarray,
    refs: dict[str, float],
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 7.2), sharex=False)

    ax = axes[0]
    ax.plot(step_a, ori_a_mean, color="#2E5FA3", linewidth=2.2, label="Condition A: reward removal")
    ax.fill_between(step_a, ori_a_mean - ori_a_sd, ori_a_mean + ori_a_sd, color="#2E5FA3", alpha=0.22)

    ax.plot(step_b, ori_b_mean, color="#E07A2D", linewidth=2.2, label="Condition B: obs corruption")
    ax.fill_between(step_b, ori_b_mean - ori_b_sd, ori_b_mean + ori_b_sd, color="#E07A2D", alpha=0.20)

    ax.axhline(refs.get("ori_phase1", np.nan), color="green", linestyle="--", linewidth=1.2, label="Phase 1 baseline")
    ax.axhline(refs.get("ori_phase2", np.nan), color="red", linestyle="--", linewidth=1.2, label="Phase 2 peak")

    ax.set_xlabel("Disruption epoch")
    ax.set_ylabel("ORI")
    ax.set_title("ORI decay under dissociation conditions")
    ax.legend(loc="best", fontsize=9)

    ax2 = axes[1]
    ax2.plot(step_a, mse_a_mean, color="#2E5FA3", linewidth=2.0, label="Condition A pred MSE")
    ax2.fill_between(step_a, mse_a_mean - mse_a_sd, mse_a_mean + mse_a_sd, color="#2E5FA3", alpha=0.22)

    ax2.plot(step_b, mse_b_mean, color="#E07A2D", linewidth=2.0, label="Condition B pred MSE")
    ax2.fill_between(step_b, mse_b_mean - mse_b_sd, mse_b_mean + mse_b_sd, color="#E07A2D", alpha=0.20)

    ax2.set_xlabel("Disruption epoch")
    ax2.set_ylabel("Prediction MSE")
    ax2.set_title("Manipulation check: prediction loss trajectories")
    ax2.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_decay_constants(out_path: Path, tau_a: np.ndarray, tau_b: np.ndarray) -> None:
    ta = tau_a[np.isfinite(tau_a)]
    tb = tau_b[np.isfinite(tau_b)]

    ma = float(np.mean(ta)) if ta.size else np.nan
    mb = float(np.mean(tb)) if tb.size else np.nan

    sa = float(np.std(ta, ddof=1)) if ta.size > 1 else np.nan
    sb = float(np.std(tb, ddof=1)) if tb.size > 1 else np.nan

    plt.figure(figsize=(6.8, 4.6))
    plt.bar(["Reward removal", "Obs corruption"], [ma, mb], yerr=[[sa, sb], [sa, sb]], capsize=5, color=["#2E5FA3", "#E07A2D"], alpha=0.9)
    plt.ylabel("Decay constant tau")
    plt.title("ORI decay time constants")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def analyze_dissociation(
    reward_removal_results: dict[str, Any],
    branch_reward_removal: Path,
    branch_obs_corruption: Path,
    output_dir: Path,
    target_noise: float,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    d_out = ensure_dir(output_dir)
    fig_dir = ensure_dir(d_out / "figures")
    seed_a = load_seed_series(branch_reward_removal)
    agg_a = aggregate_seed_curves(seed_a)
    b_root, b_source = select_obs_corruption_root(branch_obs_corruption, target_noise=target_noise)
    seed_b = load_seed_series(b_root)
    agg_b = aggregate_seed_curves(seed_b)
    ori_phase1 = reward_removal_results.get("references", {}).get("ori_phase1", np.nan)
    ori_phase2 = reward_removal_results.get("references", {}).get("ori_phase2", np.nan)
    def seed_final_ori(seed_dict: dict[str, SeriesData]) -> np.ndarray:
        vals = []
        for sd in seed_dict.values():
            if sd.ori.size == 0:
                vals.append(np.nan)
            else:
                vals.append(float(sd.ori[-1]))
        return np.asarray(vals, dtype=float)

    final_a = seed_final_ori(seed_a)
    final_b = seed_final_ori(seed_b)

    if np.sum(np.isfinite(final_a)) > 0 and np.sum(np.isfinite(final_b)) > 0:
        mw = mannwhitneyu(final_a[np.isfinite(final_a)], final_b[np.isfinite(final_b)], alternative="less")
        mw_stat = float(mw.statistic)
        mw_p = float(mw.pvalue)
    else:
        mw_stat, mw_p = np.nan, np.nan

    d = cohen_d_independent(final_b[np.isfinite(final_b)], final_a[np.isfinite(final_a)])
    tau_a = np.array([fit_tau(sd.step, sd.ori, baseline=ori_phase1) for sd in seed_a.values()], dtype=float)
    tau_b = np.array([fit_tau(sd.step, sd.ori, baseline=ori_phase1) for sd in seed_b.values()], dtype=float)
    tau_ci = bootstrap_tau_diff(tau_a, tau_b, n_boot=n_boot, seed=seed)
    mse_b = agg_b["pred_mse_mean"]
    manipulation_ratio = float(mse_b[1] / mse_b[0]) if mse_b.size >= 2 and np.isfinite(mse_b[0]) and abs(mse_b[0]) > 1e-12 else np.nan

    plot_disruption_curves(
        fig_dir / "disruption_ori_decay_comparison.png",
        step_a=agg_a["step"],
        ori_a_mean=agg_a["ori_mean"],
        ori_a_sd=agg_a["ori_std"],
        mse_a_mean=agg_a["pred_mse_mean"],
        mse_a_sd=agg_a["pred_mse_std"],
        step_b=agg_b["step"],
        ori_b_mean=agg_b["ori_mean"],
        ori_b_sd=agg_b["ori_std"],
        mse_b_mean=agg_b["pred_mse_mean"],
        mse_b_sd=agg_b["pred_mse_std"],
        refs={"ori_phase1": ori_phase1, "ori_phase2": ori_phase2},
    )

    plot_decay_constants(fig_dir / "disruption_decay_constants_bar.png", tau_a=tau_a, tau_b=tau_b)

    results = {
        "script": "reward_manipulation.py",
        "analysis": "D4_dissociation",
        "sources": {
            "condition_A_reward_removal_branch": str(branch_reward_removal),
            "condition_B_obs_corruption_branch": str(b_root),
            "condition_B_noise_source": b_source,
        },
        "trajectories": {
            "condition_A": {
                "step": agg_a["step"].tolist(),
                "ori_mean": agg_a["ori_mean"].tolist(),
                "ori_sd": agg_a["ori_std"].tolist(),
                "procrustes_mean": agg_a["procrustes_mean"].tolist(),
                "procrustes_sd": agg_a["procrustes_std"].tolist(),
                "pred_mse_mean": agg_a["pred_mse_mean"].tolist(),
                "pred_mse_sd": agg_a["pred_mse_std"].tolist(),
            },
            "condition_B": {
                "step": agg_b["step"].tolist(),
                "ori_mean": agg_b["ori_mean"].tolist(),
                "ori_sd": agg_b["ori_std"].tolist(),
                "procrustes_mean": agg_b["procrustes_mean"].tolist(),
                "procrustes_sd": agg_b["procrustes_std"].tolist(),
                "pred_mse_mean": agg_b["pred_mse_mean"].tolist(),
                "pred_mse_sd": agg_b["pred_mse_std"].tolist(),
            },
        },
        "end_point_stats": {
            "ORI_end_condition_A": safe_mean(final_a),
            "ORI_end_condition_B": safe_mean(final_b),
            "mann_whitney_u_stat": mw_stat,
            "mann_whitney_u_p": mw_p,
            "cohens_d": d,
        },
        "decay_constants": {
            "tau_A_per_seed": tau_a.tolist(),
            "tau_B_per_seed": tau_b.tolist(),
            "tau_A_mean": safe_mean(tau_a),
            "tau_B_mean": safe_mean(tau_b),
            "tau_difference_ci_95": [tau_ci[0], tau_ci[1]],
        },
        "prediction_mse_manipulation_check_ratio": manipulation_ratio,
    }

    save_json(d_out / "disruption_dissociation_results.json", results)
    return results


def compose_figure4(
    output_dir: Path,
    d1_fig: Path,
    d2_fig: Path,
    d3_fig: Path,
    d4_fig: Path,
) -> Path:
    out = ensure_dir(output_dir)
    target = out / "figure4_model_predictions.png"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    items = [
        (d1_fig, "Panel A: Reward removal"),
        (d2_fig, "Panel B: Multiple rewards"),
        (d3_fig, "Panel C: Reward relocation"),
        (d4_fig, "Panel D: Dissociation"),
    ]

    for ax, (path, title) in zip(axes.ravel(), items):
        ax.set_title(title)
        if path.exists():
            img = plt.imread(path)
            ax.imshow(img)
            ax.axis("off")
        else:
            ax.text(0.5, 0.5, f"Missing: {path.name}", ha="center", va="center")
            ax.set_axis_off()

    fig.suptitle("Figure 4: Model predictions for reward manipulations", fontsize=14)
    fig.tight_layout()
    fig.savefig(target, dpi=300)
    plt.close(fig)
    return target


def build_summary_json(
    out_path: Path,
    d1: dict[str, Any],
    d2: dict[str, Any],
    d3: dict[str, Any],
    d4: dict[str, Any],
) -> dict[str, Any]:
    d1r = d1.get("references", {})
    d1t = d1.get("trajectory", {})
    d1_ori = np.asarray(d1t.get("ori_mean", []), dtype=float)

    d2m = d2.get("metrics", {})

    d3t = d3.get("trajectory", {})
    d3d = d3.get("directional_shift", {})

    d4e = d4.get("end_point_stats", {})
    d4tau = d4.get("decay_constants", {})

    summary = {
        "D1_reward_removal": {
            "ORI_phase2": d1r.get("ori_phase2", np.nan),
            "ORI_phase3_end": float(d1_ori[-1]) if d1_ori.size else np.nan,
            "ORI_phase1_baseline": d1r.get("ori_phase1", np.nan),
            "decay_time_constant_tau": d4tau.get("tau_A_mean", np.nan),
            "tau_bootstrap_ci": d4tau.get("tau_difference_ci_95", [np.nan, np.nan]),
            "Procrustes_disparity_phase2": d1r.get("procrustes_phase2", np.nan),
            "Procrustes_disparity_phase3_end": (
                float(np.asarray(d1t.get("procrustes_mean", []), dtype=float)[-1])
                if len(d1t.get("procrustes_mean", [])) > 0
                else np.nan
            ),
        },
        "D2_multi_reward": {
            "ORI_goal1": d2m.get("ORI_goal1", np.nan),
            "ORI_goal2": d2m.get("ORI_goal2", np.nan),
            "ORI_goal1_vs_goal2_pval": d2m.get("goal1_vs_goal2_permutation_p", np.nan),
            "ORI_combined": d2m.get("ORI_combined", np.nan),
        },
        "D3_reward_relocation": {
            "ORI_old_goal_start": float(np.asarray(d3t.get("ORI_old_goal", []), dtype=float)[0]) if len(d3t.get("ORI_old_goal", [])) > 0 else np.nan,
            "ORI_new_goal_start": float(np.asarray(d3t.get("ORI_new_goal", []), dtype=float)[0]) if len(d3t.get("ORI_new_goal", [])) > 0 else np.nan,
            "crossing_episode": d3t.get("crossing_episode", np.nan),
            "ORI_old_goal_end": float(np.asarray(d3t.get("ORI_old_goal", []), dtype=float)[-1]) if len(d3t.get("ORI_old_goal", [])) > 0 else np.nan,
            "ORI_new_goal_end": float(np.asarray(d3t.get("ORI_new_goal", []), dtype=float)[-1]) if len(d3t.get("ORI_new_goal", [])) > 0 else np.nan,
            "directional_shift_new_goal_pval": d3d.get("toward_new_goal_perm_p", np.nan),
        },
        "D4_dissociation": {
            "ORI_end_reward_removal": d4e.get("ORI_end_condition_A", np.nan),
            "ORI_end_obs_corruption": d4e.get("ORI_end_condition_B", np.nan),
            "mann_whitney_pval": d4e.get("mann_whitney_u_p", np.nan),
            "cohens_d": d4e.get("cohens_d", np.nan),
            "tau_reward_removal": d4tau.get("tau_A_mean", np.nan),
            "tau_obs_corruption": d4tau.get("tau_B_mean", np.nan),
            "tau_difference_ci": d4tau.get("tau_difference_ci_95", [np.nan, np.nan]),
        },
    }

    save_json(out_path, summary)
    return summary

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Section D reward-manipulation analysis aggregator.")

    parser.add_argument("--data-dir", "--data_dir", dest="data_dir", type=str, default="data")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=str, default="../New_Results/reward_manipulation")

    parser.add_argument("--goal-file", type=str, default="goal_position.npy")
    parser.add_argument("--reward-x", type=float, default=float(DEFAULT_REWARD[0]))
    parser.add_argument("--reward-y", type=float, default=float(DEFAULT_REWARD[1]))
    parser.add_argument("--goal-area-fraction", type=float, default=0.20)

    parser.add_argument("--old-goal-x", type=float, default=0.75)
    parser.add_argument("--old-goal-y", type=float, default=0.75)
    parser.add_argument("--new-goal-x", type=float, default=0.25)
    parser.add_argument("--new-goal-y", type=float, default=0.25)
    parser.add_argument("--goal1-x", type=float, default=0.25)
    parser.add_argument("--goal1-y", type=float, default=0.75)
    parser.add_argument("--goal2-x", type=float, default=0.75)
    parser.add_argument("--goal2-y", type=float, default=0.25)

    parser.add_argument("--reward-removal-dir", type=str, default="reward_removal")
    parser.add_argument("--multi-reward-dir", type=str, default="multi_reward")
    parser.add_argument("--reward-relocation-dir", type=str, default="reward_relocation")
    parser.add_argument("--obs-corruption-dir", type=str, default="obs_corruption")
    parser.add_argument("--obs-noise-level", type=float, default=0.5)

    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_all_section_d(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = Path(args.data_dir)
    out_dir = ensure_dir(args.output_dir if args.output_dir else (data_dir / "reward_manipulation"))

    reward_xy = np.array([args.reward_x, args.reward_y], dtype=float)
    goal_main, goal_source = load_goal_position(data_dir, args.goal_file, reward_xy)
    goal_radius = float(np.sqrt(args.goal_area_fraction / np.pi))

    rr_dir = Path(args.reward_removal_dir)
    if not rr_dir.is_absolute():
        rr_dir = out_dir / rr_dir

    mr_dir = Path(args.multi_reward_dir)
    if not mr_dir.is_absolute():
        mr_dir = out_dir / mr_dir

    reloc_dir = Path(args.reward_relocation_dir)
    if not reloc_dir.is_absolute():
        reloc_dir = out_dir / reloc_dir

    obs_dir = Path(args.obs_corruption_dir)
    if not obs_dir.is_absolute():
        obs_dir = out_dir / obs_dir

    d1_out = ensure_dir(out_dir / "D1_reward_removal")
    d2_out = ensure_dir(out_dir / "D2_multi_reward")
    d3_out = ensure_dir(out_dir / "D3_reward_relocation")
    d4_out = ensure_dir(out_dir / "D4_dissociation")

    print("\n" + "=" * 50)
    print("Executing Section D Manipulations...")
    print("=" * 50)

    print("\n[1/4] Processing D1: Reward Removal...")
    d1 = analyze_reward_removal(
        data_dir=data_dir,
        branch_dir=rr_dir,
        output_dir=d1_out,
        goal=goal_main,
        goal_radius=goal_radius,
        n_perm=args.n_permutations,
        seed=args.seed,
    )

    print("\n[2/4] Processing D2: Multi-Reward...")
    d2 = analyze_multi_reward(
        data_dir=data_dir,
        branch_dir=mr_dir,
        output_dir=d2_out,
        goal1=np.array([args.goal1_x, args.goal1_y], dtype=float),
        goal2=np.array([args.goal2_x, args.goal2_y], dtype=float),
        single_goal=goal_main,
        goal_radius=goal_radius,
        n_perm=args.n_permutations,
        n_boot=args.n_bootstrap,
        seed=args.seed,
    )

    print("\n[3/4] Processing D3: Reward Relocation...")
    d3 = analyze_reward_relocation(
        data_dir=data_dir,
        branch_dir=reloc_dir,
        output_dir=d3_out,
        old_goal=np.array([args.old_goal_x, args.old_goal_y], dtype=float),
        new_goal=np.array([args.new_goal_x, args.new_goal_y], dtype=float),
        goal_radius=goal_radius,
        n_perm=args.n_permutations,
        seed=args.seed,
    )

    print("\n[4/4] Processing D4: Dissociation (Obs Corruption)...")
    d4 = analyze_dissociation(
        reward_removal_results=d1,
        branch_reward_removal=rr_dir,
        branch_obs_corruption=obs_dir,
        output_dir=d4_out,
        target_noise=args.obs_noise_level,
        n_boot=args.n_bootstrap,
        seed=args.seed,
    )

    print("\n[Done] Generating Figure 4 summary...")
    fig4 = compose_figure4(
        output_dir=out_dir / "figures",
        d1_fig=d1_out / "figures" / "reward_removal_decay_curves.png",
        d2_fig=d2_out / "figures" / "multi_reward_field_density.png",
        d3_fig=d3_out / "figures" / "relocation_ori_crossing.png",
        d4_fig=d4_out / "figures" / "disruption_ori_decay_comparison.png",
    )

    summary = build_summary_json(out_dir / "reward_manipulation_summary.json", d1=d1, d2=d2, d3=d3, d4=d4)

    payload = {
        "goal_position": goal_main.tolist(),
        "goal_position_source": goal_source,
        "goal_radius": goal_radius,
        "figure4": str(fig4),
        "summary": summary,
        "paths": {
            "D1": str(d1_out),
            "D2": str(d2_out),
            "D3": str(d3_out),
            "D4": str(d4_out),
        },
    }
    save_json(out_dir / "reward_manipulation_run_manifest.json", payload)
    return payload


def main() -> None:
    args = parse_args()
    payload = run_all_section_d(args)
    print("Saved Section D outputs:")
    print(json.dumps(payload["paths"], indent=2))


if __name__ == "__main__":
    main()
