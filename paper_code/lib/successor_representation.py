"""
Successor-representation analyses: transition matrices, the SR, and SR-vs-RNN comparisons.

States are the interior grid: `wall_buffer` cells are dropped from each side of
the m x m arena, leaving (m - 2*wb)^2 reachable cells flattened row-major.

Two layers live here. The library surface -- the index helpers, the
transition-matrix builders, compute_sr_matrix and the shift-vector functions --
is imported by lib/sr_common.py and figures/fig5/sr_direction_analysis.py.
Everything else exists only for main(), a standalone CLI writing three
sr_*_results.json files that nothing in the pipeline reads; it is not wired into
regenerate_all.py or smoke_test.py, and its --reward-x/y default of 0.75
matches no condition in the current runs.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from scipy.spatial.distance import pdist, squareform
from scipy.stats import circmean, pearsonr, spearmanr
from dataframe_io import (
    ensure_dir,
    extract_single_centers,
    load_behavior_activity,
    load_pickle,
    pivot_activity_matrix,
    save_json,
)

ACTION_UPDATES = np.array(
    [
        [1, 0],
        [1, 1],
        [0, 1],
        [-1, 1],
        [-1, 0],
        [-1, -1],
        [0, -1],
        [1, -1],
    ],
    dtype=int,
)

BASE_ACTION_PROBS = np.array([0.5, 0.15, 0.05, 0.04, 0.02, 0.04, 0.05, 0.15], dtype=float)


def parse_args() -> argparse.Namespace:
    """CLI arguments for the standalone SR analysis (main() only)."""
    parser = argparse.ArgumentParser(description="Section E: Successor Representation analyses.")
    parser.add_argument("--data-dir", "--data_dir", dest="data_dir", type=str, default="data")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=str, default="../New_Results/successor_representation")
    parser.add_argument("--m", type=int, default=32, help="Original grid size used by the simulator.")
    parser.add_argument("--wall-buffer", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--sr-method", type=str, default="auto", choices=["auto", "inverse", "pinv", "neumann"])
    parser.add_argument("--neumann-k", type=int, default=500)
    parser.add_argument("--condition-threshold", type=float, default=1e10)
    parser.add_argument("--goal-file", type=str, default="goal_position.npy")
    parser.add_argument("--reward-x", type=float, default=0.75)
    parser.add_argument("--reward-y", type=float, default=0.75)
    parser.add_argument("--goal-area-fraction", type=float, default=0.20)
    parser.add_argument(
        "--policy-source",
        type=str,
        default="auto",
        choices=["auto", "empirical", "goal_biased", "behavior_after"],
        help="How to construct T_policy.",
    )
    parser.add_argument("--policy-transition-file", type=str, default="")
    parser.add_argument("--policy-behavior-file", type=str, default="")
    parser.add_argument("--policy-beta", type=float, default=8.0, help="Goal-bias strength when policy_source uses goal_biased.")
    parser.add_argument("--state-min-visits", type=int, default=1)
    parser.add_argument("--bin-min-visits", type=int, default=3)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--n-shuffle", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def interior_size(m: int, wall_buffer: int) -> int:
    """Side length of the interior grid: m minus `wall_buffer` on each side."""
    return m - 2 * wall_buffer


def n_interior_states(m: int, wall_buffer: int) -> int:
    """Number of interior states, i.e. interior_size squared."""
    s = interior_size(m, wall_buffer)
    return s * s


def interior_index(row: int, col: int, m: int, wall_buffer: int) -> int:
    """Flat state index for an (row, col) arena coordinate."""
    s = interior_size(m, wall_buffer)
    return (row - wall_buffer) * s + (col - wall_buffer)


def interior_coord(idx: int, m: int, wall_buffer: int) -> tuple[int, int]:
    """Inverse of `interior_index`: flat state index back to (row, col)."""
    s = interior_size(m, wall_buffer)
    r = idx // s + wall_buffer
    c = idx % s + wall_buffer
    return int(r), int(c)


def state_centers(m: int, wall_buffer: int) -> np.ndarray:
    """(n_states, 2) centre of every interior state, in arena coords [0, 1]."""
    n = n_interior_states(m, wall_buffer)
    out = np.zeros((n, 2), dtype=float)
    for idx in range(n):
        r, c = interior_coord(idx, m, wall_buffer)
        out[idx, 0] = r / m
        out[idx, 1] = c / m
    return out


def pos_to_interior_idx(x: np.ndarray, y: np.ndarray, m: int, wall_buffer: int) -> np.ndarray:
    """Map continuous (x, y) positions to the interior state index containing them."""
    s = interior_size(m, wall_buffer)
    rr = np.clip((x * s).astype(int), 0, s - 1) + wall_buffer
    cc = np.clip((y * s).astype(int), 0, s - 1) + wall_buffer
    idx = (rr - wall_buffer) * s + (cc - wall_buffer)
    return idx.astype(int)


def action_probs_like_get_next_move(row: int, col: int, hd_idx: int, m: int, wall_buffer: int) -> np.ndarray:
    """Action distribution for one state and heading, mirroring the simulator.

    Reproduces the movement policy of `model/ratinabox_v11.py` so that the SR
    baseline describes the same random walk the network actually experienced.
    `BASE_ACTION_PROBS` is rolled by the heading index (forward is most likely,
    reversals least), then any action that would cross a wall is zeroed and the
    remainder renormalized.

    Parameters
    ----------
    row, col : int
        State coordinates in arena grid units.
    hd_idx : int
        Heading index 0..7; rolls the base distribution so it is relative to
        the current facing.
    m, wall_buffer : int
        Arena size and the interior margin.

    Returns
    -------
    ndarray (8,), float
        Probability per action, summing to 1. A state with every action blocked
        falls back to uniform.
    """
    p = np.roll(BASE_ACTION_PROBS, hd_idx).astype(float).copy()
    border_flags = [
        row >= (m - wall_buffer),
        col >= (m - wall_buffer),
        row <= wall_buffer,
        col <= wall_buffer,
    ]

    if np.any(border_flags):
        if border_flags[0]:
            p[[0, 1, 7]] = 0.0
        if border_flags[1]:
            p[[2, 1, 3]] = 0.0
        if border_flags[2]:
            p[[4, 3, 5]] = 0.0
        if border_flags[3]:
            p[[6, 5, 7]] = 0.0

        s = float(np.sum(p))
        if s > 0:
            p /= s
        else:
            p[:] = 1.0 / p.size

    return p


def clipped_next_coord(row: int, col: int, action_idx: int, m: int, wall_buffer: int) -> tuple[int, int]:
    """Apply action `action_idx` to (row, col), clipped to stay inside the interior."""
    dr, dc = ACTION_UPDATES[action_idx]
    r2 = int(np.clip(row + dr, wall_buffer, m - wall_buffer - 1))
    c2 = int(np.clip(col + dc, wall_buffer, m - wall_buffer - 1))
    return r2, c2


def build_T_random(m: int, wall_buffer: int) -> np.ndarray:
    """Transition matrix for the unbiased random walk over interior states.

    Marginalizes `action_probs_like_get_next_move` over all 8 headings -- the
    walk has no preferred direction, so each heading is equally likely -- and
    accumulates the resulting probability into the clipped destination state.

    Parameters
    ----------
    m, wall_buffer : int
        Arena size and the interior margin.

    Returns
    -------
    ndarray (n_states, n_states), float
        Row-stochastic. Rows are renormalized if floating-point drift leaves
        them off 1.0.
    """
    n = n_interior_states(m, wall_buffer)
    T = np.zeros((n, n), dtype=float)

    for i in range(n):
        row, col = interior_coord(i, m, wall_buffer)
        probs = np.zeros(8, dtype=float)

        for hd in range(8):
            probs += action_probs_like_get_next_move(row, col, hd, m, wall_buffer)
        probs /= 8.0

        for a in range(8):
            r2, c2 = clipped_next_coord(row, col, a, m, wall_buffer)
            j = interior_index(r2, c2, m, wall_buffer)
            T[i, j] += probs[a]

    row_sum = T.sum(axis=1)
    bad = np.where(~np.isclose(row_sum, 1.0, atol=1e-8))[0]
    if bad.size > 0:
        # Renormalize just in case of tiny numerical drift.
        T = np.divide(T, row_sum[:, None], out=np.zeros_like(T), where=row_sum[:, None] > 0)

    return T


def uniform_neighbor_row(row: int, col: int, m: int, wall_buffer: int) -> np.ndarray:
    """Transition row placing equal mass on each of one state's 8 clipped neighbours."""
    n = n_interior_states(m, wall_buffer)
    r = np.zeros(n, dtype=float)

    for a in range(8):
        r2, c2 = clipped_next_coord(row, col, a, m, wall_buffer)
        j = interior_index(r2, c2, m, wall_buffer)
        r[j] += 1.0

    s = float(np.sum(r))
    if s > 0:
        r /= s
    return r


def normalize_transition_rows(T: np.ndarray, m: int, wall_buffer: int) -> np.ndarray:
    """Row-normalize T; a state never left falls back to `uniform_neighbor_row`."""
    n = T.shape[0]
    row_sum = T.sum(axis=1)
    out = np.zeros_like(T)

    for i in range(n):
        if row_sum[i] > 0:
            out[i] = T[i] / row_sum[i]
        else:
            row, col = interior_coord(i, m, wall_buffer)
            out[i] = uniform_neighbor_row(row, col, m, wall_buffer)

    return out


def transition_from_behavior_positions(pos: np.ndarray, m: int, wall_buffer: int) -> tuple[np.ndarray, float]:
    """Empirical transition matrix counted from a trajectory, plus the fraction of states it visited."""
    n = n_interior_states(m, wall_buffer)
    T_counts = np.zeros((n, n), dtype=float)

    idx = pos_to_interior_idx(pos[:, 0], pos[:, 1], m, wall_buffer)
    if idx.size >= 2:
        for t in range(idx.size - 1):
            T_counts[idx[t], idx[t + 1]] += 1.0

    visited_fraction = float(np.mean(T_counts.sum(axis=1) > 0))
    T = normalize_transition_rows(T_counts, m, wall_buffer)
    return T, visited_fraction


def transition_goal_biased(m: int, wall_buffer: int, goal_xy: np.ndarray, beta: float) -> np.ndarray:
    """Synthetic policy biased toward `goal_xy`: actions are reweighted by exp(beta * distance reduced)."""
    n = n_interior_states(m, wall_buffer)
    centers = state_centers(m, wall_buffer)
    T = np.zeros((n, n), dtype=float)

    for i in range(n):
        row, col = interior_coord(i, m, wall_buffer)
        curr_xy = centers[i]
        curr_dist = float(np.linalg.norm(curr_xy - goal_xy))

        per_action = np.zeros(8, dtype=float)
        for hd in range(8):
            p = action_probs_like_get_next_move(row, col, hd, m, wall_buffer)
            per_action += p
        per_action /= 8.0

        w = np.zeros(8, dtype=float)
        for a in range(8):
            r2, c2 = clipped_next_coord(row, col, a, m, wall_buffer)
            j = interior_index(r2, c2, m, wall_buffer)
            next_dist = float(np.linalg.norm(centers[j] - goal_xy))
            progress = curr_dist - next_dist
            w[a] = per_action[a] * np.exp(beta * progress)

        if np.sum(w) <= 0:
            w[:] = 1.0 / 8.0
        else:
            w /= np.sum(w)

        for a in range(8):
            r2, c2 = clipped_next_coord(row, col, a, m, wall_buffer)
            j = interior_index(r2, c2, m, wall_buffer)
            T[i, j] += w[a]

    return normalize_transition_rows(T, m, wall_buffer)


def load_transition_from_file(path: Path, m: int, wall_buffer: int) -> np.ndarray | None:
    """Load a precomputed transition matrix, or None if the file is missing or the wrong shape."""
    n = n_interior_states(m, wall_buffer)
    if not path.exists():
        return None

    if path.suffix.lower() == ".npy":
        arr = np.asarray(np.load(path), dtype=float)
        if arr.shape == (n, n):
            return normalize_transition_rows(arr, m, wall_buffer)
        return None

    if path.suffix.lower() == ".npz":
        npz = np.load(path)
        for key in ["T", "transition", "counts"]:
            if key in npz.files:
                arr = np.asarray(npz[key], dtype=float)
                if arr.shape == (n, n):
                    return normalize_transition_rows(arr, m, wall_buffer)
        return None

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if {"state_i", "state_j", "count"}.issubset(df.columns):
            M = np.zeros((n, n), dtype=float)
            for _, r in df.iterrows():
                i = int(r["state_i"])
                j = int(r["state_j"])
                if 0 <= i < n and 0 <= j < n:
                    M[i, j] += float(r["count"])
            return normalize_transition_rows(M, m, wall_buffer)

        if {"state_i", "state_j", "prob"}.issubset(df.columns):
            M = np.zeros((n, n), dtype=float)
            for _, r in df.iterrows():
                i = int(r["state_i"])
                j = int(r["state_j"])
                if 0 <= i < n and 0 <= j < n:
                    M[i, j] += float(r["prob"])
            return normalize_transition_rows(M, m, wall_buffer)

        return None

    return None


def load_behavior_positions(path: Path) -> np.ndarray | None:
    """Load an (n, 2) position array from a behaviour file, or None if absent."""
    if not path.exists():
        return None

    if path.suffix.lower() == ".pkl":
        df = pd.read_pickle(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() == ".npz":
        npz = np.load(path)
        if "positions" in npz.files:
            arr = np.asarray(npz["positions"], dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return arr[:, :2]
        return None
    else:
        return None

    if not isinstance(df, pd.DataFrame):
        return None
    if not {"x", "y"}.issubset(df.columns):
        return None
    return df[["x", "y"]].to_numpy(dtype=float)


def build_T_policy(
    data_dir: Path,
    m: int,
    wall_buffer: int,
    goal_xy: np.ndarray,
    policy_source: str,
    policy_transition_file: str,
    policy_behavior_file: str,
    beta: float,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    """Post-learning transition matrix, from whichever `policy_source` is configured."""
    info: dict[str, Any] = {}
    if policy_source in {"auto", "empirical"} and policy_transition_file.strip() != "":
        p = Path(policy_transition_file)
        if not p.is_absolute():
            p = data_dir / p
        T = load_transition_from_file(p, m, wall_buffer)
        if T is not None:
            info["visited_fraction"] = np.nan
            return T, f"transition_file:{p}", info
    if policy_source in {"auto", "empirical"} and policy_behavior_file.strip() != "":
        p = Path(policy_behavior_file)
        if not p.is_absolute():
            p = data_dir / p
        pos = load_behavior_positions(p)
        if pos is not None and pos.shape[0] >= 2:
            T, vf = transition_from_behavior_positions(pos, m, wall_buffer)
            info["visited_fraction"] = vf
            return T, f"behavior_file:{p}", info

    if policy_source in {"auto", "empirical"}:
        for name in ["df_behavior_policy.pkl", "policy_behavior.pkl", "policy_rollout_behavior.pkl"]:
            p = data_dir / name
            pos = load_behavior_positions(p)
            if pos is not None and pos.shape[0] >= 2:
                T, vf = transition_from_behavior_positions(pos, m, wall_buffer)
                info["visited_fraction"] = vf
                return T, f"behavior_file:{p}", info

    if policy_source == "behavior_after":
        p = data_dir / "df_behavior_after.pkl"
        pos = load_behavior_positions(p)
        if pos is not None and pos.shape[0] >= 2:
            T, vf = transition_from_behavior_positions(pos, m, wall_buffer)
            info["visited_fraction"] = vf
            return T, f"behavior_after:{p}", info
    T = transition_goal_biased(m, wall_buffer, goal_xy=goal_xy, beta=beta)
    info["visited_fraction"] = np.nan
    return T, "goal_biased_analytic", info


def compute_sr_matrix(
    T: np.ndarray,
    gamma: float,
    method: str,
    neumann_k: int,
    cond_threshold: float,
) -> tuple[np.ndarray, float, str]:
    """Successor representation M = (I - gamma*T)^-1, with the solver chosen by conditioning.

    M[i, j] is the discounted expected number of visits to state j when
    starting from state i, so column j is the "successor field" of j.

    Parameters
    ----------
    T : ndarray (n, n)
        Row-stochastic transition matrix.
    gamma : float
        Discount; 0.99 throughout this project.
    method : {"auto", "inverse", "pinv", "neumann"}
        "auto" picks `inverse` when cond(I - gamma*T) <= `cond_threshold` and
        `pinv` otherwise. "neumann" sums the truncated series instead.
    neumann_k : int
        Terms in the Neumann series, used only by that method.
    cond_threshold : float
        Conditioning cutoff for "auto".

    Returns
    -------
    M : ndarray (n, n)
        The SR.
    cond : float
        Condition number of (I - gamma*T), worth recording -- it is what
        decided the solver.
    used : str
        Which method actually ran, including "pinv_fallback" when the direct
        solve raised.
    """
    n = T.shape[0]
    I = np.eye(n, dtype=float)
    A = I - gamma * T
    cond = float(np.linalg.cond(A))

    used = method
    if method == "auto":
        used = "inverse" if cond <= cond_threshold else "pinv"

    if used == "inverse":
        try:
            M = np.linalg.solve(A, I)
        except np.linalg.LinAlgError:
            M = np.linalg.pinv(A)
            used = "pinv_fallback"
        return M, cond, used

    if used == "pinv":
        M = np.linalg.pinv(A)
        return M, cond, used

    if used == "neumann":
        M = I.copy()
        Tk = I.copy()
        coeff = 1.0
        for _ in range(1, int(neumann_k) + 1):
            Tk = Tk @ T
            coeff *= gamma
            M += coeff * Tk
        return M, cond, used

    raise ValueError(f"Unknown SR method: {method}")


def sr_participation_ratio(M: np.ndarray) -> tuple[float, np.ndarray]:
    """Participation ratio of the symmetrized SR's eigenspectrum, with the spectrum itself."""
    Ms = 0.5 * (M + M.T)
    eig = np.linalg.eigvalsh(Ms)
    eig = np.sort(np.maximum(eig, 0.0))[::-1]

    valid = eig[eig > 1e-12]
    if valid.size == 0:
        return np.nan, eig

    pr = float((np.sum(valid) ** 2) / np.sum(valid ** 2))
    return pr, eig


def vector_to_grid(vec: np.ndarray, m: int, wall_buffer: int) -> np.ndarray:
    """Reshape a per-state vector back into the (s, s) interior grid, for plotting."""
    s = interior_size(m, wall_buffer)
    return np.asarray(vec, dtype=float).reshape(s, s)


def plot_sr_place_fields(M: np.ndarray, out_path: Path, title: str, m: int, wall_buffer: int, seed: int) -> None:
    """Grid of randomly chosen SR columns rendered as place fields."""
    n = M.shape[0]
    rng = np.random.default_rng(seed)
    cols = rng.choice(n, size=min(100, n), replace=False)

    fig, axes = plt.subplots(10, 10, figsize=(12, 12))
    axes = axes.ravel()

    for k, j in enumerate(cols):
        ax = axes[k]
        grid = vector_to_grid(M[:, j], m, wall_buffer)
        ax.imshow(grid, origin="lower", cmap="viridis")
        ax.set_xticks([])
        ax.set_yticks([])

    for k in range(len(cols), 100):
        axes[k].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def state_distances_to_goal(m: int, wall_buffer: int, goal_xy: np.ndarray) -> np.ndarray:
    """Euclidean distance from every state centre to the goal."""
    coords = state_centers(m, wall_buffer)
    d = np.sqrt((coords[:, 0] - goal_xy[0]) ** 2 + (coords[:, 1] - goal_xy[1]) ** 2)
    return d.astype(float)


def sr_ori_from_diag(diag_vals: np.ndarray, dists: np.ndarray, goal_radius: float) -> tuple[float, dict[str, int]]:
    """Over-representation index: how much more often above-mean SR diagonal values fall inside the goal radius than outside."""
    vals = np.asarray(diag_vals, dtype=float)
    thr = float(np.nanmean(vals))
    above = vals > thr
    goal_mask = dists <= goal_radius
    non_mask = ~goal_mask
    n_goal = int(np.sum(goal_mask))
    n_non = int(np.sum(non_mask))
    n_goal_above = int(np.sum(above & goal_mask))
    n_non_above = int(np.sum(above & non_mask))

    frac_goal = (n_goal_above / n_goal) if n_goal > 0 else np.nan
    frac_non = (n_non_above / n_non) if n_non > 0 else np.nan
    ori = frac_goal / frac_non if np.isfinite(frac_goal) and np.isfinite(frac_non) and frac_non > 0 else np.nan
    return float(ori), {
        "n_goal": n_goal,
        "n_non_goal": n_non,
        "n_goal_above": n_goal_above,
        "n_non_goal_above": n_non_above,
    }


def bootstrap_sr_ori(diag_vals: np.ndarray, dists: np.ndarray, goal_radius: float, n_boot: int, seed: int) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for `sr_ori_from_diag`."""
    vals = np.asarray(diag_vals, dtype=float)
    n = vals.size
    rng = np.random.default_rng(seed)
    out = np.zeros(n_boot, dtype=float)

    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        v = vals[idx]
        d = dists[idx]
        out[i], _ = sr_ori_from_diag(v, d, goal_radius)

    out = out[np.isfinite(out)]
    if out.size == 0:
        return np.nan, np.nan
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def load_goal_position(data_dir: Path, goal_file: str, reward_x: float, reward_y: float) -> tuple[np.ndarray, str]:
    """Goal position from `goal_file` if it exists, else the --reward-x/y fallback, with its source."""
    p = data_dir / goal_file
    if p.exists():
        arr = np.asarray(np.load(p), dtype=float).reshape(-1)
        if arr.size >= 2:
            return np.array([float(arr[0]), float(arr[1])], dtype=float), str(p)
    return np.array([reward_x, reward_y], dtype=float), "fallback_args"


def load_centers(data_dir: Path, which: str) -> tuple[pd.DataFrame, str]:
    """Field centres for 'before' or 'after' from whichever centre file exists, with the path used."""
    if which == "before":
        cands = [data_dir / "df_centers_before.pkl", data_dir / "place_field_centres_phase1.npy"]
    else:
        cands = [data_dir / "df_centers_after.pkl", data_dir / "place_field_centres_phase2.npy"]

    for p in cands:
        if not p.exists():
            continue
        if p.suffix.lower() == ".pkl":
            df = load_pickle(p)
            if {"neuron_id", "x", "y"}.issubset(df.columns):
                out = df[["neuron_id", "x", "y"]].copy()
                out["neuron_id"] = out["neuron_id"].astype(int)
                return out, str(p)
            if {"neuron_id", "center"}.issubset(df.columns):
                out = extract_single_centers(df, k_value=1)
                if not out.empty:
                    out["neuron_id"] = out["neuron_id"].astype(int)
                    return out[["neuron_id", "x", "y"]], str(p)
        elif p.suffix.lower() == ".npy":
            arr = np.asarray(np.load(p), dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 3:
                out = pd.DataFrame({"neuron_id": arr[:, 0].astype(int), "x": arr[:, 1], "y": arr[:, 2]})
                return out, str(p)

    return pd.DataFrame(columns=["neuron_id", "x", "y"]), "missing"


def rnn_ori_from_centers(centers: pd.DataFrame, goal_xy: np.ndarray, goal_radius: float) -> float:
    """The same over-representation index as `sr_ori_from_diag`, for RNN field centres."""
    if centers.empty:
        return np.nan
    xy = centers[["x", "y"]].to_numpy(dtype=float)
    d = np.sqrt((xy[:, 0] - goal_xy[0]) ** 2 + (xy[:, 1] - goal_xy[1]) ** 2)
    in_goal = d <= goal_radius
    n_goal = int(np.sum(in_goal))
    n_non = int(np.sum(~in_goal))
    n_total = int(xy.shape[0])
    if n_total == 0 or n_goal == 0 or n_non == 0:
        return np.nan

    f_goal = float(np.clip(np.pi * goal_radius * goal_radius, 1e-9, 1.0 - 1e-9))
    f_non = 1.0 - f_goal

    dens_goal = n_goal / (n_total * f_goal)
    dens_non = n_non / (n_total * f_non)
    return float(dens_goal / dens_non) if dens_non > 0 else np.nan


def bootstrap_rnn_ori(centers: pd.DataFrame, goal_xy: np.ndarray, goal_radius: float, n_boot: int, seed: int) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for `rnn_ori_from_centers`."""
    if centers.empty:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    n = centers.shape[0]
    vals = np.zeros(n_boot, dtype=float)

    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        vals[i] = rnn_ori_from_centers(centers.iloc[idx], goal_xy, goal_radius)

    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def plot_ori_bar(sr_ori: float, sr_ci: tuple[float, float], rnn_ori: float, rnn_ci: tuple[float, float], out_path: Path, title: str) -> None:
    """Two-bar SR-vs-RNN over-representation figure with bootstrap CIs."""
    labels = ["SR ORI", "RNN ORI"]
    vals = [sr_ori, rnn_ori]

    low = [sr_ori - sr_ci[0], rnn_ori - rnn_ci[0]]
    high = [sr_ci[1] - sr_ori, rnn_ci[1] - rnn_ori]

    plt.figure(figsize=(6.4, 4.6))
    plt.bar(labels, vals, yerr=[low, high], capsize=5, color=["#2E5FA3", "#E07A2D"], alpha=0.9)
    plt.axhline(1.0, linestyle="--", color="black", linewidth=1.2)
    plt.ylabel("ORI")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def load_period_matrix(data_dir: Path, period: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One period as (activity (t, n_cells), positions (t, 2), neuron_ids)."""
    beh, act_df, _ = load_behavior_activity(data_dir, period)
    X_df, B_df = pivot_activity_matrix(act_df, beh)
    if B_df is None:
        raise RuntimeError(f"Could not align behavior/activity for period={period}.")
    X = X_df.to_numpy(dtype=float)
    neuron_ids = X_df.columns.to_numpy(dtype=int)
    pos = B_df[["x", "y"]].to_numpy(dtype=float)

    n = min(X.shape[0], pos.shape[0])
    return X[:n], pos[:n], neuron_ids


def build_rnn_state_maps(X: np.ndarray, pos: np.ndarray, m: int, wall_buffer: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell mean activity per interior state, plus each state's visit count. Unvisited states are NaN."""
    S = n_interior_states(m, wall_buffer)
    N = X.shape[1]

    idx = pos_to_interior_idx(pos[:, 0], pos[:, 1], m, wall_buffer)
    maps = np.full((N, S), np.nan, dtype=float)
    counts = np.zeros(S, dtype=int)

    for s in range(S):
        t = np.where(idx == s)[0]
        counts[s] = int(t.size)
        if t.size == 0:
            continue
        maps[:, s] = np.mean(X[t], axis=0)

    return maps, counts


def build_population_response_16(X: np.ndarray, pos: np.ndarray, min_visits: int, m: int, wall_buffer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Population response on a coarser 16x16 grid, with a validity mask and each bin's interior state index."""
    N = X.shape[1]
    K = 16 * 16
    xb = np.clip(np.floor(pos[:, 0] * 16).astype(int), 0, 15)
    yb = np.clip(np.floor(pos[:, 1] * 16).astype(int), 0, 15)
    flat = xb * 16 + yb
    R = np.full((K, N), np.nan, dtype=float)
    valid = np.zeros(K, dtype=bool)

    for b in range(K):
        t = np.where(flat == b)[0]
        if t.size < min_visits:
            continue
        R[b] = np.mean(X[t], axis=0)
        valid[b] = True
    sidx = np.zeros(K, dtype=int)
    for b in range(K):
        i = b // 16
        j = b % 16
        x = (i + 0.5) / 16.0
        y = (j + 0.5) / 16.0
        sidx[b] = int(pos_to_interior_idx(np.array([x]), np.array([y]), m, wall_buffer)[0])

    return R, valid, sidx


def fill_nan_rowwise(A: np.ndarray) -> np.ndarray:
    """Replace NaNs in each row with that row's mean, so correlations can be taken."""
    B = np.asarray(A, dtype=float).copy()
    for i in range(B.shape[0]):
        row = B[i]
        mask = np.isfinite(row)
        if np.any(mask):
            mean = float(np.mean(row[mask]))
            row[~mask] = mean
        else:
            row[:] = 0.0
        B[i] = row
    return B


def normalize_rows_for_corr(X: np.ndarray) -> np.ndarray:
    """Centre and unit-norm each row, so a dot product is a Pearson correlation."""
    Xc = X - np.mean(X, axis=1, keepdims=True)
    norm = np.linalg.norm(Xc, axis=1, keepdims=True)
    out = np.divide(Xc, norm, out=np.zeros_like(Xc), where=norm > 1e-12)
    return out


def normalize_cols_for_corr(Y: np.ndarray) -> np.ndarray:
    """Centre and unit-norm each column, the column-wise counterpart of `normalize_rows_for_corr`."""
    Yc = Y - np.mean(Y, axis=0, keepdims=True)
    norm = np.linalg.norm(Yc, axis=0, keepdims=True)
    out = np.divide(Yc, norm, out=np.zeros_like(Yc), where=norm > 1e-12)
    return out


def max_corr_vs_sr(rnn_maps: np.ndarray, SR: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Best-matching SR column for every RNN cell: max correlation, its index, and the full matrix."""
    X = fill_nan_rowwise(rnn_maps)
    Xn = normalize_rows_for_corr(X)
    Yn = normalize_cols_for_corr(SR)

    C = Xn @ Yn
    max_corr = np.max(C, axis=1)
    best_idx = np.argmax(C, axis=1)
    return max_corr.astype(float), best_idx.astype(int), C.astype(float), Xn.astype(float), Yn.astype(float)


def shuffle_max_corr_baseline(Xn: np.ndarray, Yn: np.ndarray, n_shuffle: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Null for `max_corr_vs_sr` from permuting SR states: the first draw per cell and the per-shuffle means."""
    rng = np.random.default_rng(seed)
    N, S = Xn.shape
    if n_shuffle <= 0:
        return np.full(N, np.nan, dtype=float), np.array([], dtype=float)
    mean_max = np.zeros(n_shuffle, dtype=float)
    baseline_first = np.full(N, np.nan, dtype=float)
    for k in range(n_shuffle):
        perm = rng.permutation(S)
        Yp = Yn[perm, :]
        C = Xn @ Yp
        mc = np.max(C, axis=1)
        if k == 0:
            baseline_first = mc.copy()
        mean_max[k] = float(np.mean(mc))

    return baseline_first, mean_max


def spearman_upper(A: np.ndarray, B: np.ndarray) -> tuple[float, float]:
    """Spearman correlation between two RDMs, over the upper triangle only."""
    if A.shape != B.shape or A.shape[0] < 2:
        return np.nan, np.nan

    tri = np.triu_indices(A.shape[0], k=1)
    x = A[tri]
    y = B[tri]
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 5:
        return np.nan, np.nan

    rho, p = spearmanr(x[mask], y[mask])
    return float(rho), float(p)


def corr_distance_rdm(X: np.ndarray) -> np.ndarray:
    """Representational dissimilarity matrix using correlation distance."""
    if X.shape[0] < 2:
        return np.full((X.shape[0], X.shape[0]), np.nan)
    d = pdist(X, metric="correlation")
    return squareform(d)


def bootstrap_interaction(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the interaction (mean d - mean c) - (mean b - mean a)."""
    n = min(a.size, b.size, c.size, d.size)
    if n < 5:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    vals = np.zeros(n_boot, dtype=float)

    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        vals[i] = float((np.mean(d[idx]) - np.mean(c[idx])) - (np.mean(b[idx]) - np.mean(a[idx])))

    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def sr_predicted_shift_vectors(M_random: np.ndarray, M_policy: np.ndarray, state_xy: np.ndarray) -> np.ndarray:
    """Field-centre displacement each state's successor field is predicted to undergo.

    For state j, the change in its successor field is M_policy[:, j] -
    M_random[:, j]. Treating the positive part of that change as a weight over
    state centres gives a centroid, and the predicted shift is that centroid
    minus the state's own position -- i.e. the direction the field is pulled by
    moving from a random walk to the goal-directed policy.

    Parameters
    ----------
    M_random, M_policy : ndarray (n, n)
        SR matrices for the two policies.
    state_xy : ndarray (n, 2)
        State centres in arena coords, from `state_centers`.

    Returns
    -------
    ndarray (n, 2), float
        Predicted (dx, dy) per state. States whose field does not change get a
        zero vector; if the positive part vanishes the absolute change is used
        instead, so a purely negative change still yields a direction.
    """
    S = M_random.shape[0]
    out = np.zeros((S, 2), dtype=float)
    for j in range(S):
        delta = M_policy[:, j] - M_random[:, j]
        w = np.maximum(delta, 0.0)
        if np.sum(w) <= 1e-12:
            w = np.abs(delta)
        if np.sum(w) <= 1e-12:
            out[j] = 0.0
            continue

        centroid = np.sum(state_xy * w[:, None], axis=0) / np.sum(w)
        out[j] = centroid - state_xy[j]

    return out


def nearest_state_for_points(xy: np.ndarray, state_xy: np.ndarray) -> np.ndarray:
    """Index of the nearest interior state to each (x, y) point."""
    d2 = (
        (xy[:, None, 0] - state_xy[None, :, 0]) ** 2
        + (xy[:, None, 1] - state_xy[None, :, 1]) ** 2
    )
    return np.argmin(d2, axis=1).astype(int)


def angle_between_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Unsigned angle in [0, pi] between paired row vectors; NaN where either is degenerate."""
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    valid = (na > 1e-12) & (nb > 1e-12)
    out = np.full(a.shape[0], np.nan, dtype=float)
    if np.any(valid):
        av = a[valid] / na[valid, None]
        bv = b[valid] / nb[valid, None]
        dot = np.clip(np.sum(av * bv, axis=1), -1.0, 1.0)
        out[valid] = np.arccos(dot)

    return out


def rayleigh_from_angles(angles: np.ndarray) -> tuple[float, float]:
    """Rayleigh test on a set of angles: resultant length R and its p-value."""
    ang = np.asarray(angles, dtype=float)
    ang = ang[np.isfinite(ang)]
    n = ang.size
    if n == 0:
        return np.nan, np.nan

    R = float(np.abs(np.mean(np.exp(1j * ang))))
    p = float(np.exp(-n * (R ** 2)))
    return R, p

def plot_corr_hist(
    out_path: Path,
    max_p1: np.ndarray,
    max_p2: np.ndarray,
    shuffle_baseline: np.ndarray,
) -> None:
    """Histogram of per-cell best SR correlation for both phases against the shuffled baseline."""
    plt.figure(figsize=(7.2, 4.8))
    plt.hist(shuffle_baseline[np.isfinite(shuffle_baseline)], bins=30, alpha=0.5, color="gray", label="Shuffle baseline")
    plt.hist(max_p1[np.isfinite(max_p1)], bins=30, alpha=0.55, color="#2E5FA3", label="Phase 1")
    plt.hist(max_p2[np.isfinite(max_p2)], bins=30, alpha=0.55, color="#E07A2D", label="Phase 2")
    plt.xlabel("Max SR-RNN correlation per neuron")
    plt.ylabel("Count")
    plt.title("SR-RNN max-correlation distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_2x2_corr_bar(
    out_path: Path,
    p1_rand: float,
    p1_pol: float,
    p2_rand: float,
    p2_pol: float,
) -> None:
    """2x2 bar chart of mean best correlation across phase x SR variant."""
    x = np.arange(2)
    w = 0.35

    plt.figure(figsize=(7.0, 4.8))
    plt.bar(x - w / 2, [p1_rand, p2_rand], width=w, color="#2E5FA3", alpha=0.9, label="vs SR_random")
    plt.bar(x + w / 2, [p1_pol, p2_pol], width=w, color="#E07A2D", alpha=0.9, label="vs SR_policy")
    plt.xticks(x, ["Phase 1", "Phase 2"])
    plt.ylabel("Mean max correlation")
    plt.title("SR-RNN correspondence (unit-level)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_rsa_bar(out_path: Path, vals: list[float], labels: list[str]) -> None:
    """Bar chart of RSA similarity values."""
    plt.figure(figsize=(8.0, 4.8))
    plt.bar(labels, vals, color=["#2E5FA3", "#E07A2D", "#2E5FA3", "#E07A2D"], alpha=0.9)
    plt.axhline(0.0, linestyle="--", color="black", linewidth=1.2)
    plt.ylabel("Spearman rho")
    plt.title("RSA: RNN geometry vs SR geometry")
    plt.xticks(rotation=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_sr_eigenspectrum(out_path: Path, eig: np.ndarray, rnn_pr: float) -> None:
    """SR eigenspectrum with the RNN participation ratio marked."""
    r = np.arange(1, eig.size + 1)

    plt.figure(figsize=(7.2, 5.0))
    plt.loglog(r, np.maximum(eig, 1e-12), color="#2E5FA3", linewidth=2.0)
    if np.isfinite(rnn_pr):
        plt.axvline(rnn_pr, linestyle="--", color="red", linewidth=1.4, label="RNN PR")
        plt.legend()
    plt.xlabel("Eigenvalue rank")
    plt.ylabel("Eigenvalue")
    plt.title("SR eigenvalue spectrum")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_residual_map(out_path: Path, residual_map: np.ndarray) -> None:
    """Per-state residual drawn as an arena heatmap."""
    plt.figure(figsize=(6.2, 5.4))
    im = plt.imshow(residual_map.T, origin="lower", cmap="magma")
    plt.colorbar(im, shrink=0.85, label="Residual magnitude")
    plt.title("SR-RNN residual map (16x16)")
    plt.xlabel("x bin")
    plt.ylabel("y bin")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_predicted_vs_actual_shift(out_path: Path, pred_mag: np.ndarray, act_mag: np.ndarray, dist_reward: np.ndarray) -> None:
    """SR-predicted against actual shift magnitude, coloured by distance to reward."""
    plt.figure(figsize=(6.8, 5.6))
    sc = plt.scatter(pred_mag, act_mag, c=dist_reward, cmap="viridis", s=22, alpha=0.7)
    lim = np.nanmax([np.nanmax(pred_mag), np.nanmax(act_mag), 1e-6])
    plt.plot([0, lim], [0, lim], linestyle="--", color="black", linewidth=1.2)
    plt.colorbar(sc, shrink=0.85, label="Distance to reward")
    plt.xlabel("SR-predicted shift magnitude")
    plt.ylabel("Actual RNN shift magnitude")
    plt.title("SR-predicted vs actual shift magnitudes")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_rose(out_path: Path, angles: np.ndarray, R: float, p: float) -> None:
    """Rose plot of shift angles annotated with the Rayleigh R and p."""
    a = angles[np.isfinite(angles)]
    bins = np.linspace(0, np.pi, 19)
    hist, edges = np.histogram(a, bins=bins)
    theta = 0.5 * (edges[:-1] + edges[1:])
    width = np.diff(edges)

    fig = plt.figure(figsize=(6.2, 6.0))
    ax = fig.add_subplot(111, projection="polar")
    ax.bar(theta, hist, width=width, alpha=0.75, color="#2E5FA3", edgecolor="black")
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_title(f"Angle difference rose plot\nRayleigh R={R:.3f}, p={p:.2e}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_residual_vs_distance(out_path: Path, residual: np.ndarray, dist: np.ndarray, rho: float, p: float) -> None:
    """Residual against distance to reward, annotated with Spearman rho and p."""
    plt.figure(figsize=(6.8, 5.4))
    plt.scatter(dist, residual, s=20, alpha=0.65, color="#2E5FA3")

    mask = np.isfinite(dist) & np.isfinite(residual)
    if np.sum(mask) >= 2:
        x = dist[mask]
        y = residual[mask]
        b1, b0 = np.polyfit(x, y, deg=1)
        xx = np.linspace(np.min(x), np.max(x), 100)
        yy = b1 * xx + b0
        plt.plot(xx, yy, color="red", linewidth=1.5)

    plt.xlabel("Distance to reward")
    plt.ylabel("Residual shift magnitude")
    plt.title(f"Residual shift vs distance (rho={rho:.3f}, p={p:.3g})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def main() -> None:
    """Run the standalone SR analysis end to end and write its JSONs and figures."""
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    data_dir = Path(args.data_dir)
    out_dir = ensure_dir(args.output_dir if args.output_dir else (data_dir / "successor_representation"))
    out_data = ensure_dir(out_dir / "data")
    out_fig = ensure_dir(out_dir / "figures")
    goal_xy, goal_source = load_goal_position(data_dir, args.goal_file, args.reward_x, args.reward_y)
    goal_radius = float(np.sqrt(args.goal_area_fraction / np.pi))
    S = n_interior_states(args.m, args.wall_buffer)
    state_xy = state_centers(args.m, args.wall_buffer)
    T_random = build_T_random(args.m, args.wall_buffer)
    T_policy, policy_source, policy_info = build_T_policy(
        data_dir=data_dir,
        m=args.m,
        wall_buffer=args.wall_buffer,
        goal_xy=goal_xy,
        policy_source=args.policy_source,
        policy_transition_file=args.policy_transition_file,
        policy_behavior_file=args.policy_behavior_file,
        beta=args.policy_beta,
    )
    row_ok_random = bool(np.allclose(T_random.sum(axis=1), 1.0, atol=1e-6))
    row_ok_policy = bool(np.allclose(T_policy.sum(axis=1), 1.0, atol=1e-6))
    SR_random, cond_random, sr_method_random = compute_sr_matrix(
        T_random,
        gamma=args.gamma,
        method=args.sr_method,
        neumann_k=args.neumann_k,
        cond_threshold=args.condition_threshold,
    )
    SR_policy, cond_policy, sr_method_policy = compute_sr_matrix(
        T_policy,
        gamma=args.gamma,
        method=args.sr_method,
        neumann_k=args.neumann_k,
        cond_threshold=args.condition_threshold,
    )

    np.save(out_data / "T_random.npy", T_random)
    np.save(out_data / "T_policy.npy", T_policy)
    np.save(out_data / "SR_random.npy", SR_random)
    np.save(out_data / "SR_policy.npy", SR_policy)

    plot_sr_place_fields(
        SR_random,
        out_fig / "sr_place_fields_random.png",
        title="SR place fields under random-walk policy",
        m=args.m,
        wall_buffer=args.wall_buffer,
        seed=args.seed,
    )
    plot_sr_place_fields(
        SR_policy,
        out_fig / "sr_place_fields_policy.png",
        title="SR place fields under goal-directed policy",
        m=args.m,
        wall_buffer=args.wall_buffer,
        seed=args.seed + 1,
    )
    sr_diag = np.diag(SR_policy)
    dist_state_goal = state_distances_to_goal(args.m, args.wall_buffer, goal_xy)
    sr_ori, sr_ori_counts = sr_ori_from_diag(sr_diag, dist_state_goal, goal_radius)
    sr_ori_ci = bootstrap_sr_ori(sr_diag, dist_state_goal, goal_radius, n_boot=args.n_bootstrap, seed=args.seed)
    centers_before, centers_before_src = load_centers(data_dir, "before")
    centers_after, centers_after_src = load_centers(data_dir, "after")
    rnn_ori = rnn_ori_from_centers(centers_after, goal_xy, goal_radius)
    rnn_ori_ci = bootstrap_rnn_ori(centers_after, goal_xy, goal_radius, n_boot=args.n_bootstrap, seed=args.seed + 5)

    plot_ori_bar(
        sr_ori=sr_ori,
        sr_ci=sr_ori_ci,
        rnn_ori=rnn_ori,
        rnn_ci=rnn_ori_ci,
        out_path=out_fig / "sr_ori_vs_rnn_ori.png",
        title="SR ORI vs RNN ORI",
    )
    X_before, pos_before, neuron_ids_before = load_period_matrix(data_dir, "before")
    X_after, pos_after, neuron_ids_after = load_period_matrix(data_dir, "after")

    n_shared = min(X_before.shape[1], X_after.shape[1])
    X_before = X_before[:, :n_shared]
    X_after = X_after[:, :n_shared]
    maps_before, counts_before = build_rnn_state_maps(X_before, pos_before, args.m, args.wall_buffer)
    maps_after, counts_after = build_rnn_state_maps(X_after, pos_after, args.m, args.wall_buffer)
    max_b_rand, best_b_rand, _, Xn_b, Yn_rand = max_corr_vs_sr(maps_before, SR_random)
    max_b_pol, best_b_pol, _, _, Yn_pol = max_corr_vs_sr(maps_before, SR_policy)
    max_a_rand, best_a_rand, _, Xn_a, _ = max_corr_vs_sr(maps_after, SR_random)
    max_a_pol, best_a_pol, _, _, _ = max_corr_vs_sr(maps_after, SR_policy)
    shuffle_baseline_per_neuron, shuffle_mean_dist = shuffle_max_corr_baseline(Xn_a, Yn_pol, n_shuffle=args.n_shuffle, seed=args.seed)
    mean_b_rand = float(np.nanmean(max_b_rand))
    mean_b_pol = float(np.nanmean(max_b_pol))
    mean_a_rand = float(np.nanmean(max_a_rand))
    mean_a_pol = float(np.nanmean(max_a_pol))
    interaction = (mean_a_pol - mean_a_rand) - (mean_b_pol - mean_b_rand)
    ci_inter = bootstrap_interaction(max_b_rand, max_b_pol, max_a_rand, max_a_pol, n_boot=args.n_bootstrap, seed=args.seed)
    plot_corr_hist(
        out_fig / "sr_rnn_max_correlation_histogram.png",
        max_p1=max_b_rand,
        max_p2=max_a_pol,
        shuffle_baseline=shuffle_baseline_per_neuron,
    )
    plot_2x2_corr_bar(
        out_fig / "sr_rnn_correlation_2x2_bar.png",
        p1_rand=mean_b_rand,
        p1_pol=mean_b_pol,
        p2_rand=mean_a_rand,
        p2_pol=mean_a_pol,
    )
    resp_b, valid_b, state16_b = build_population_response_16(
        X_before,
        pos_before,
        min_visits=args.bin_min_visits,
        m=args.m,
        wall_buffer=args.wall_buffer,
    )
    resp_a, valid_a, state16_a = build_population_response_16(
        X_after,
        pos_after,
        min_visits=args.bin_min_visits,
        m=args.m,
        wall_buffer=args.wall_buffer,
    )

    common = valid_b & valid_a
    if np.sum(common) >= 8:
        R_b = resp_b[common]
        R_a = resp_a[common]

        rdm_b = corr_distance_rdm(R_b)
        rdm_a = corr_distance_rdm(R_a)

        states_common = state16_b[common]

        SR_cols_rand = SR_random[:, states_common].T  # (K,S)
        SR_cols_pol = SR_policy[:, states_common].T
        rdm_sr_rand = corr_distance_rdm(SR_cols_rand)
        rdm_sr_pol = corr_distance_rdm(SR_cols_pol)
        rsa_b_rand, rsa_b_rand_p = spearman_upper(rdm_b, rdm_sr_rand)
        rsa_b_pol, rsa_b_pol_p = spearman_upper(rdm_b, rdm_sr_pol)
        rsa_a_rand, rsa_a_rand_p = spearman_upper(rdm_a, rdm_sr_rand)
        rsa_a_pol, rsa_a_pol_p = spearman_upper(rdm_a, rdm_sr_pol)
    else:
        rsa_b_rand = rsa_b_rand_p = np.nan
        rsa_b_pol = rsa_b_pol_p = np.nan
        rsa_a_rand = rsa_a_rand_p = np.nan
        rsa_a_pol = rsa_a_pol_p = np.nan

    plot_rsa_bar(
        out_fig / "rsa_sr_rnn_comparison.png",
        vals=[rsa_b_rand, rsa_b_pol, rsa_a_rand, rsa_a_pol],
        labels=["P1-SRr", "P1-SRp", "P2-SRr", "P2-SRp"],
    )

    sr_pr, sr_eig = sr_participation_ratio(SR_random)
    Xc = X_before - np.mean(X_before, axis=0, keepdims=True)
    cov = np.cov(Xc, rowvar=False)
    eig_rnn = np.linalg.eigvalsh(cov)
    eig_rnn = eig_rnn[eig_rnn > 1e-12]
    if eig_rnn.size > 0:
        rnn_pr = float((np.sum(eig_rnn) ** 2) / np.sum(eig_rnn ** 2))
    else:
        rnn_pr = np.nan
    plot_sr_eigenspectrum(out_fig / "sr_eigenvalue_spectrum.png", sr_eig, rnn_pr=rnn_pr)
    residual_map = np.full((16, 16), np.nan, dtype=float)
    if np.sum(valid_a) > 0:
        bins = np.where(valid_a)[0]
        actual = resp_a[bins]  # (Kb, N)
        pred = np.full_like(actual, np.nan)
        for k, b in enumerate(bins):
            sidx = state16_a[b]
            pred[k] = SR_policy[sidx, best_a_pol]
        for n in range(actual.shape[1]):
            av = actual[:, n]
            pv = pred[:, n]
            ma = np.nanmean(av)
            sa = np.nanstd(av, ddof=1)
            mp = np.nanmean(pv)
            sp = np.nanstd(pv, ddof=1)
            if np.isfinite(sa) and sa > 1e-12:
                actual[:, n] = (av - ma) / sa
            else:
                actual[:, n] = np.nan
            if np.isfinite(sp) and sp > 1e-12:
                pred[:, n] = (pv - mp) / sp
            else:
                pred[:, n] = np.nan
        diff = actual - pred
        res = np.sqrt(np.nanmean(diff ** 2, axis=1))
        for k, b in enumerate(bins):
            i = b // 16
            j = b % 16
            residual_map[i, j] = res[k]

    plot_residual_map(out_fig / "sr_rnn_residual_map.png", residual_map)

    sr_rnn_results = {
        "mean_corr_phase1_vs_SR_random": mean_b_rand,
        "mean_corr_phase1_vs_SR_policy": mean_b_pol,
        "mean_corr_phase2_vs_SR_random": mean_a_rand,
        "mean_corr_phase2_vs_SR_policy": mean_a_pol,
        "rsa_rnn_phase1_vs_sr_random": {"rho": rsa_b_rand, "p": rsa_b_rand_p},
        "rsa_rnn_phase2_vs_sr_random": {"rho": rsa_a_rand, "p": rsa_a_rand_p},
        "rsa_rnn_phase1_vs_sr_policy": {"rho": rsa_b_pol, "p": rsa_b_pol_p},
        "rsa_rnn_phase2_vs_sr_policy": {"rho": rsa_a_pol, "p": rsa_a_pol_p},
        "interaction_contrast": interaction,
        "interaction_bootstrap_ci_low": ci_inter[0],
        "interaction_bootstrap_ci_high": ci_inter[1],
        "SR_participation_ratio": sr_pr,
        "RNN_participation_ratio": rnn_pr,
        "shuffle_baseline": {
            "mean_of_mean_max_corr": float(np.nanmean(shuffle_mean_dist)) if shuffle_mean_dist.size else np.nan,
            "std_of_mean_max_corr": float(np.nanstd(shuffle_mean_dist, ddof=1)) if shuffle_mean_dist.size > 1 else np.nan,
            "n_shuffle": int(args.n_shuffle),
        },
        "n_neurons_used": int(n_shared),
        "n_common_bins_16": int(np.sum(common)),
    }
    save_json(out_dir / "sr_rnn_comparison_results.json", sr_rnn_results)
    pred_state_vec = sr_predicted_shift_vectors(SR_random, SR_policy, state_xy)
    mdf = pd.merge(
        centers_before[["neuron_id", "x", "y"]],
        centers_after[["neuron_id", "x", "y"]],
        on="neuron_id",
        suffixes=("_b", "_a"),
    )

    if mdf.shape[0] > 0:
        xy_b = mdf[["x_b", "y_b"]].to_numpy(dtype=float)
        xy_a = mdf[["x_a", "y_a"]].to_numpy(dtype=float)
        actual_shift = xy_a - xy_b
        state_match = nearest_state_for_points(xy_b, state_xy)
        pred_shift = pred_state_vec[state_match]
        pred_mag = np.linalg.norm(pred_shift, axis=1)
        actual_mag = np.linalg.norm(actual_shift, axis=1)
        angles = angle_between_vectors(actual_shift, pred_shift)
        pred_norm = np.linalg.norm(pred_shift, axis=1)
        u = np.divide(pred_shift, pred_norm[:, None], out=np.zeros_like(pred_shift), where=pred_norm[:, None] > 1e-12)
        proj_scalar = np.sum(actual_shift * u, axis=1)
        proj_vec = proj_scalar[:, None] * u
        residual_vec = actual_shift - proj_vec
        residual_mag = np.linalg.norm(residual_vec, axis=1)
        dist_reward = np.sqrt((xy_b[:, 0] - goal_xy[0]) ** 2 + (xy_b[:, 1] - goal_xy[1]) ** 2)
        mask = np.isfinite(residual_mag) & np.isfinite(dist_reward)
        if np.sum(mask) >= 5:
            rho_resid, p_resid = spearmanr(residual_mag[mask], dist_reward[mask])
            rho_resid = float(rho_resid)
            p_resid = float(p_resid)
        else:
            rho_resid, p_resid = np.nan, np.nan

        R_ray, p_ray = rayleigh_from_angles(angles)
        mean_ang = float(circmean(angles[np.isfinite(angles)], high=np.pi, low=0)) if np.any(np.isfinite(angles)) else np.nan
        frac_consistent = float(np.mean(angles[np.isfinite(angles)] < (np.pi / 4))) if np.any(np.isfinite(angles)) else np.nan
        frac_inconsistent = float(np.mean(angles[np.isfinite(angles)] > (3 * np.pi / 4))) if np.any(np.isfinite(angles)) else np.nan
        plot_predicted_vs_actual_shift(
            out_fig / "sr_predicted_vs_actual_shifts.png",
            pred_mag=pred_mag,
            act_mag=actual_mag,
            dist_reward=dist_reward,
        )
        plot_rose(out_fig / "shift_angle_difference_rose.png", angles=angles, R=R_ray, p=p_ray)
        plot_residual_vs_distance(
            out_fig / "residual_shift_vs_distance.png",
            residual=residual_mag,
            dist=dist_reward,
            rho=rho_resid,
            p=p_resid,
        )
    else:
        pred_mag = np.array([], dtype=float)
        actual_mag = np.array([], dtype=float)
        angles = np.array([], dtype=float)
        residual_mag = np.array([], dtype=float)
        dist_reward = np.array([], dtype=float)
        rho_resid = p_resid = np.nan
        R_ray = p_ray = np.nan
        mean_ang = np.nan
        frac_consistent = np.nan
        frac_inconsistent = np.nan
        for name in [
            "sr_predicted_vs_actual_shifts.png",
            "shift_angle_difference_rose.png",
            "residual_shift_vs_distance.png",
        ]:
            plt.figure(figsize=(5, 4))
            plt.text(0.5, 0.5, "Insufficient center data", ha="center", va="center")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(out_fig / name, dpi=300)
            plt.close()

    plot_ori_bar(
        sr_ori=sr_ori,
        sr_ci=sr_ori_ci,
        rnn_ori=rnn_ori,
        rnn_ci=rnn_ori_ci,
        out_path=out_fig / "ori_rnn_vs_sr_bar.png",
        title="RNN ORI vs SR ORI",
    )

    sr_reshaping_results = {
        "SR_ORI": sr_ori,
        "SR_ORI_ci_95": [sr_ori_ci[0], sr_ori_ci[1]],
        "RNN_ORI": rnn_ori,
        "RNN_ORI_ci_95": [rnn_ori_ci[0], rnn_ori_ci[1]],
        "ori_difference_rnn_minus_sr": (rnn_ori - sr_ori) if np.isfinite(rnn_ori) and np.isfinite(sr_ori) else np.nan,
        "rayleigh_R": R_ray,
        "rayleigh_p": p_ray,
        "mean_angle_difference": mean_ang,
        "spearman_residual_shift_vs_distance": {
            "rho": rho_resid,
            "p": p_resid,
        },
        "fraction_angle_lt_pi_over_4": frac_consistent,
        "fraction_angle_gt_3pi_over_4": frac_inconsistent,
        "n_neurons_with_shifts": int(pred_mag.size),
    }
    save_json(out_dir / "sr_reshaping_comparison_results.json", sr_reshaping_results)
    sr_comp = {
        "gamma": float(args.gamma),
        "m": int(args.m),
        "wall_buffer": int(args.wall_buffer),
        "n_states": int(S),
        "goal_position": goal_xy.tolist(),
        "goal_position_source": goal_source,
        "goal_zone_radius": goal_radius,
        "transition_sources": {
            "T_random": "analytic_get_next_move_marginalized_heading",
            "T_policy": policy_source,
            "T_policy_info": policy_info,
        },
        "row_stochastic_checks": {
            "T_random": row_ok_random,
            "T_policy": row_ok_policy,
        },
        "condition_numbers": {
            "I_minus_gamma_T_random": cond_random,
            "I_minus_gamma_T_policy": cond_policy,
            "threshold": float(args.condition_threshold),
        },
        "sr_solver": {
            "requested_method": args.sr_method,
            "used_random": sr_method_random,
            "used_policy": sr_method_policy,
            "neumann_K": int(args.neumann_k),
        },
        "SR_ORI": sr_ori,
        "SR_ORI_ci_95": [sr_ori_ci[0], sr_ori_ci[1]],
        "RNN_ORI_from_B1": rnn_ori,
        "RNN_ORI_ci_95": [rnn_ori_ci[0], rnn_ori_ci[1]],
        "sr_ori_counts": sr_ori_counts,
        "qualitative_match_assessment": bool(
            np.isfinite(sr_ori) and np.isfinite(rnn_ori) and (abs(sr_ori - rnn_ori) <= 0.75)
        ),
        "center_sources": {
            "before": centers_before_src,
            "after": centers_after_src,
        },
    }
    save_json(out_dir / "sr_computation_results.json", sr_comp)

    print(f"Saved Section E outputs to: {out_dir}")

if __name__ == "__main__":
    main()
