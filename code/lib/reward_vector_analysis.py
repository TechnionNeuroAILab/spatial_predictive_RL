"""Reward-distance ramping (Gauthier & Tank 2018) and egocentric goal-bearing tuning (Sarel et al. 2017).

Per-cell scores with cyclic-shift nulls, FDR-BH plus effect floors, an iso-distance ring control
and the relocation rotation prediction. Entry point: analyze_period().
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from analysis_helpers import load_period_activity  # noqa: E402
from stats_helpers import fdr_bh  # noqa: E402

M_GRID = 32
WALL_BUFFER_CELLS = 1 # simulator's hard movement restriction
EDGE_MARGIN_CELLS = 1 # additional soft margin against bearing/distance border effects

# Relocation periods whose active reward is B; B is not in config.json, it is the CLI default (0.7, 0.7).
RELOC_PHASE_B_PERIODS = {
    "after", "cycle_2", "cycle_2_goal", "goal_conditioned", "replay_reloc_postB",
}
DEFAULT_REWARD_B = (0.7, 0.7)

def _load_config(run_dir) -> dict:
    with open(Path(run_dir) / "config.json") as f:
        return json.load(f)


def resolve_period_reward_pos(run_dir, period: str, reward_b=DEFAULT_REWARD_B) -> np.ndarray:
    """Active reward for `period`: config["reward_pos"], or B for relocation phase-B periods."""
    config = _load_config(run_dir)
    reward_a = np.asarray(config["reward_pos"], dtype=float)
    if config.get("n_cycles", 1) <= 1:
        return reward_a
    # extinction runs have n_cycles > 1 but the reward never moves
    if config.get("extinguish_at_cycle", -1) >= 0:
        return reward_a
    return np.asarray(reward_b, dtype=float) if period in RELOC_PHASE_B_PERIODS else reward_a



def wall_buffer_mask(x: np.ndarray, y: np.ndarray, m: int = M_GRID,
                      wall_buffer: int = WALL_BUFFER_CELLS,
                      margin_cells: int = EDGE_MARGIN_CELLS) -> np.ndarray:
    """Timesteps kept after dropping `margin_cells` grid cells inside the movement boundary."""
    lo = (wall_buffer + margin_cells) / m
    hi = (m - 1 - wall_buffer - margin_cells) / m      # agent reaches k/m for k = 1..m-2: same margin both sides
    return (x >= lo) & (x <= hi) & (y >= lo) & (y <= hi)



def geometry(x: np.ndarray, y: np.ndarray, theta: np.ndarray, reward_pos):
    """Distance to reward, egocentric bearing (0 = straight ahead) and allocentric bearing, per step."""
    reward_pos = np.asarray(reward_pos, dtype=float)
    dist = np.sqrt((x - reward_pos[0]) ** 2 + (y - reward_pos[1]) ** 2)
    target_angle = np.arctan2(reward_pos[1] - y, reward_pos[0] - x)
    ego = np.arctan2(np.sin(target_angle - theta), np.cos(target_angle - theta))
    return dist, ego, target_angle


def circular_diff(a, b):
    """Wrap-safe a-b in [-pi, pi]."""
    return np.angle(np.exp(1j * (np.asarray(a) - np.asarray(b))))



def _col_corr(X: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Pearson correlation of each column of X (T,N) with vector v (T,)."""
    v = v - v.mean()
    vn = np.sqrt((v * v).sum())
    Xc = X - X.mean(axis=0, keepdims=True)
    Xn = np.sqrt((Xc * Xc).sum(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (Xc * v[:, None]).sum(axis=0) / (Xn * vn)
    return np.where(np.isfinite(r), r, 0.0)


def distance_modulation(X: np.ndarray, dist: np.ndarray, nbins: int = 12):
    """Ramping index corr(activity, -distance) (positive = more active near reward) and tuning curves."""
    ramping = _col_corr(X, -dist)  # (N,)
    edges = np.linspace(dist.min(), dist.max(), nbins + 1)
    which = np.clip(np.digitize(dist, edges) - 1, 0, nbins - 1)
    onehot = np.zeros((len(dist), nbins), dtype=np.float32)
    onehot[np.arange(len(dist)), which] = 1.0
    counts = onehot.sum(axis=0)
    curve = (onehot.T @ X) / np.maximum(counts[:, None], 1)  # (nbins, N)
    return ramping, curve.T, 0.5 * (edges[:-1] + edges[1:])  # curve (N, nbins)


def ring_angular_uniformity(X: np.ndarray, dist: np.ndarray, allo_angle: np.ndarray,
                             n_dist_bins: int = 6, n_angle_bins: int = 8,
                             min_bin_count: int = 20):
    """Occupancy-weighted MRL of angular tuning within iso-distance rings; low = true distance cell."""
    edges = np.linspace(dist.min(), dist.max(), n_dist_bins + 1)
    ring_id = np.clip(np.digitize(dist, edges) - 1, 0, n_dist_bins - 1)
    N = X.shape[1]
    ring_mrls = []
    ring_weights = []
    angle_edges = np.linspace(-np.pi, np.pi, n_angle_bins + 1)
    for r in range(n_dist_bins):
        in_ring = ring_id == r
        # complete rings only: a wall-cut ring's empty sectors would read as silence and fake angle tuning
        sector = np.clip(np.digitize(allo_angle[in_ring], angle_edges) - 1, 0, n_angle_bins - 1)
        if np.bincount(sector, minlength=n_angle_bins).min() < min_bin_count:
            continue
        mrl_r, _, _, _ = circular_tuning(X[in_ring], allo_angle[in_ring], nbins=n_angle_bins)
        ring_mrls.append(mrl_r)
        ring_weights.append(in_ring.sum())
    if not ring_mrls:
        return np.zeros(N, dtype=float)
    ring_mrls = np.stack(ring_mrls, axis=0)          # (n_valid_rings, N)
    w = np.asarray(ring_weights, dtype=float)
    w = w / w.sum()
    return (ring_mrls * w[:, None]).sum(axis=0)      # (N,), weighted mean across rings


def ring_angular_uniformity_null(X: np.ndarray, dist: np.ndarray, allo_angle: np.ndarray,
                                  n_dist_bins: int = 6, n_angle_bins: int = 8,
                                  min_bin_count: int = 20, n_null: int = 100, seed: int = 0,
                                  return_samples: bool = False):
    """Cyclic-shift null for ring_angular_uniformity (allo_angle rolled against activity)."""
    rng = np.random.default_rng(seed)
    T = len(dist)
    null = np.empty((n_null, X.shape[1]), dtype=np.float32)
    for s in range(n_null):
        shift = int(rng.integers(1, T))
        null[s] = ring_angular_uniformity(X, dist, np.roll(allo_angle, shift),
                                           n_dist_bins=n_dist_bins, n_angle_bins=n_angle_bins,
                                           min_bin_count=min_bin_count)
    return null if return_samples else np.percentile(null, 95, axis=0)


def distance_modulation_null(X: np.ndarray, dist: np.ndarray, nbins: int = 12,
                              n_null: int = 100, seed: int = 0,
                              return_samples: bool = False):
    """Cyclic-shift null of |ramping index|: 95th percentile per neuron, or all samples."""
    rng = np.random.default_rng(seed)
    T = len(dist)
    null = np.empty((n_null, X.shape[1]), dtype=np.float32)
    for s in range(n_null):
        shift = int(rng.integers(1, T))
        null[s] = np.abs(_col_corr(X, -np.roll(dist, shift)))
    return null if return_samples else np.percentile(null, 95, axis=0)



def circular_tuning(X: np.ndarray, angle: np.ndarray, nbins: int = 16):
    """Binned circular tuning: (mrl, preferred angle, curve, bin centres) per neuron."""
    edges = np.linspace(-np.pi, np.pi, nbins + 1)
    which = np.clip(np.digitize(angle, edges) - 1, 0, nbins - 1)
    onehot = np.zeros((len(angle), nbins), dtype=np.float32)
    onehot[np.arange(len(angle)), which] = 1.0
    counts = onehot.sum(axis=0)
    curve = ((onehot.T @ X) / np.maximum(counts[:, None], 1)).T  # (N, nbins)
    phi = 0.5 * (edges[:-1] + edges[1:])
    vec = (curve * np.exp(1j * phi)[None, :]).sum(axis=1)
    tot = curve.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mrl = np.where(tot > 0, np.abs(vec) / tot, 0.0)
    return mrl, np.angle(vec), curve, phi


def circular_tuning_exact(X: np.ndarray, values_discrete: np.ndarray):
    """circular_tuning grouped by exact value, for the 8 discrete headings."""
    vals = np.unique(values_discrete)
    onehot = (values_discrete[:, None] == vals[None, :]).astype(np.float32)
    counts = onehot.sum(axis=0)
    curve = ((onehot.T @ X) / np.maximum(counts[:, None], 1)).T  # (N, n_vals)
    vec = (curve * np.exp(1j * vals)[None, :]).sum(axis=1)
    tot = curve.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mrl = np.where(tot > 0, np.abs(vec) / tot, 0.0)
    return mrl, np.angle(vec), curve, vals


def circular_tuning_exact_null(X: np.ndarray, values_discrete: np.ndarray,
                                n_null: int = 100, seed: int = 0,
                                return_samples: bool = False):
    """Cyclic-shift null for circular_tuning_exact."""
    rng = np.random.default_rng(seed)
    T = len(values_discrete)
    vals = np.unique(values_discrete)
    null = np.empty((n_null, X.shape[1]), dtype=np.float32)
    for s in range(n_null):
        shift = int(rng.integers(1, T))
        v = np.roll(values_discrete, shift)
        onehot = (v[:, None] == vals[None, :]).astype(np.float32)
        counts = onehot.sum(axis=0)
        curve = ((onehot.T @ X) / np.maximum(counts[:, None], 1)).T
        vec = (curve * np.exp(1j * vals)[None, :]).sum(axis=1)
        tot = curve.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            null[s] = np.where(tot > 0, np.abs(vec) / tot, 0.0)
    return null if return_samples else np.percentile(null, 95, axis=0)


def bearing_tuning(X: np.ndarray, ego: np.ndarray, nbins: int = 16):
    return circular_tuning(X, ego, nbins=nbins)


def bearing_tuning_null(X: np.ndarray, ego: np.ndarray, nbins: int = 16,
                         n_null: int = 100, seed: int = 0,
                         return_samples: bool = False):
    """Cyclic-shift null of the bearing MRL: 95th percentile per neuron, or all samples."""
    rng = np.random.default_rng(seed)
    T = len(ego)
    edges = np.linspace(-np.pi, np.pi, nbins + 1)
    phi = 0.5 * (edges[:-1] + edges[1:])
    null = np.empty((n_null, X.shape[1]), dtype=np.float32)
    for s in range(n_null):
        shift = int(rng.integers(1, T))
        e = np.roll(ego, shift)
        which = np.clip(np.digitize(e, edges) - 1, 0, nbins - 1)
        onehot = np.zeros((T, nbins), dtype=np.float32)
        onehot[np.arange(T), which] = 1.0
        counts = onehot.sum(axis=0)
        curve = ((onehot.T @ X) / np.maximum(counts[:, None], 1)).T
        vec = (curve * np.exp(1j * phi)[None, :]).sum(axis=1)
        tot = curve.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            null[s] = np.where(tot > 0, np.abs(vec) / tot, 0.0)
    return null if return_samples else np.percentile(null, 95, axis=0)



def _empirical_p_from_null(observed: np.ndarray, null_thr_95: np.ndarray,
                            null_samples: np.ndarray | None = None) -> np.ndarray:
    """Empirical p = (#null >= observed + 1) / (n_null + 1); binary 0.01 / 1 if only the threshold is given."""
    if null_samples is not None:
        ge = (null_samples >= observed[None, :]).sum(axis=0)
        return (ge + 1) / (null_samples.shape[0] + 1)
    return np.where(observed > null_thr_95, 0.01, 1.0)


def classify_cells(ramp: np.ndarray, ramp_p: np.ndarray,
                    mrl_ego: np.ndarray, mrl_ego_p: np.ndarray,
                    mrl_hd: np.ndarray, mrl_hd_p: np.ndarray,
                    ring_mrl: np.ndarray | None = None,
                    effect_floor_ramp: float = 0.1, effect_floor_mrl: float = 0.1,
                    effect_floor_ring: float = 0.15,
                    alpha: float = 0.05) -> dict:
    """Boolean cell classes from FDR-BH q <= alpha plus effect floors; distance cells also need low ring_mrl."""
    ramp_q = fdr_bh(ramp_p)
    mrl_ego_q = fdr_bh(mrl_ego_p)
    mrl_hd_q = fdr_bh(mrl_hd_p)

    ramping_sig = (ramp_q <= alpha) & (np.abs(ramp) >= effect_floor_ramp)
    if ring_mrl is not None:
        # with one fixed reward any place field ramps, so also require uniform firing around rings
        distance_cell = ramping_sig & (ring_mrl < effect_floor_ring)
    else:
        distance_cell = ramping_sig
    goal_direction_cell = (mrl_ego_q <= alpha) & (mrl_ego >= effect_floor_mrl)
    head_direction_cell = (mrl_hd_q <= alpha) & (mrl_hd >= effect_floor_mrl)
    goal_vector_cell = distance_cell & goal_direction_cell
    goal_specific_cell = goal_direction_cell & ~head_direction_cell

    return {
        "distance_cell": distance_cell,
        "ramping_only_cell": ramping_sig,   # before the ring control
        "goal_direction_cell": goal_direction_cell,
        "goal_vector_cell": goal_vector_cell,
        "head_direction_cell": head_direction_cell,
        "goal_specific_cell": goal_specific_cell,
    }



def geometric_rotation(run_dir, period_a: str = "cycle_1_goal",
                        period_b: str = "cycle_2_goal",
                        reward_b=DEFAULT_REWARD_B) -> float:
    """Delta_geom: circular mean change of the allocentric bearing A -> B at phase-A positions."""
    data_dir = Path(run_dir) / "data"
    _, x_a, y_a, _ = load_period_activity(str(data_dir), period_a)
    reward_a = resolve_period_reward_pos(run_dir, period_a, reward_b)
    reward_b_pos = resolve_period_reward_pos(run_dir, period_b, reward_b)
    ta_a = np.arctan2(reward_a[1] - y_a, reward_a[0] - x_a)
    ta_b = np.arctan2(reward_b_pos[1] - y_a, reward_b_pos[0] - x_a)  # same positions, hypothetical B
    return float(np.angle(np.mean(np.exp(1j * (ta_b - ta_a)))))


def bootstrap_circular_mean_ci(angles: np.ndarray, n_boot: int = 2000,
                                seed: int = 0, ci: float = 0.95):
    """Circular mean of `angles` and its bootstrap CI."""
    if len(angles) == 0:
        return float("nan"), (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = len(angles)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = np.angle(np.mean(np.exp(1j * angles[idx])))
    circ_mean = float(np.angle(np.mean(np.exp(1j * angles))))
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(means, [100 * alpha, 100 * (1 - alpha)])
    return circ_mean, (float(lo), float(hi))



def analyze_period(run_dir, period: str, hidden_dim: int = 512, m: int = M_GRID,
                    wall_buffer: int = WALL_BUFFER_CELLS, margin_cells: int = EDGE_MARGIN_CELLS,
                    nbins_dist: int = 12, nbins_bearing: int = 16,
                    n_null: int = 100, seed: int = 0, reward_b=DEFAULT_REWARD_B) -> dict | None:
    """All per-cell scores and null p-values for one period; None if missing or < 100 steps kept."""
    data_dir = Path(run_dir) / "data"
    try:
        X, x, y, theta = load_period_activity(str(data_dir), period, hidden_dim=hidden_dim)
    except Exception:
        return None

    mask = wall_buffer_mask(x, y, m=m, wall_buffer=wall_buffer, margin_cells=margin_cells)
    if mask.sum() < 100:
        return None
    X, x, y, theta = X[mask], x[mask], y[mask], theta[mask]

    reward_pos = resolve_period_reward_pos(run_dir, period, reward_b)
    dist, ego, target_angle = geometry(x, y, theta, reward_pos)

    ramp, dist_curve, dbins = distance_modulation(X, dist, nbins=nbins_dist)
    ramp_null = distance_modulation_null(X, dist, nbins=nbins_dist, n_null=n_null,
                                          seed=seed, return_samples=True)
    ramp_p = _empirical_p_from_null(np.abs(ramp), None, null_samples=ramp_null)

    mrl_ego, pref_ego, curve_ego, phi_ego = bearing_tuning(X, ego, nbins=nbins_bearing)
    mrl_ego_null = bearing_tuning_null(X, ego, nbins=nbins_bearing, n_null=n_null,
                                        seed=seed + 1, return_samples=True)
    mrl_ego_p = _empirical_p_from_null(mrl_ego, None, null_samples=mrl_ego_null)

    # head-direction control on the 8 exact headings
    mrl_hd, pref_hd, curve_hd, hd_vals = circular_tuning_exact(X, theta)
    mrl_hd_null = circular_tuning_exact_null(X, theta, n_null=n_null, seed=seed + 2,
                                              return_samples=True)
    mrl_hd_p = _empirical_p_from_null(mrl_hd, None, null_samples=mrl_hd_null)

    ring_mrl = ring_angular_uniformity(X, dist, target_angle)

    return {
        "n": int(mask.sum()),
        "reward_pos": reward_pos,
        "ramp": ramp, "ramp_p": ramp_p, "dist_curve": dist_curve, "dbins": dbins,
        "ring_mrl": ring_mrl,
        "mrl_ego": mrl_ego, "pref_ego": pref_ego, "mrl_ego_p": mrl_ego_p,
        "curve_ego": curve_ego, "phi_ego": phi_ego,
        "mrl_hd": mrl_hd, "pref_hd": pref_hd, "mrl_hd_p": mrl_hd_p,
        "curve_hd": curve_hd, "hd_vals": hd_vals,
    }
