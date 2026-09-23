"""
reward_vector_analysis.py — Gauthier & Tank 2018 (reward-distance ramping) and
Sarel et al. 2017 (egocentric goal-vector tuning) analyses.

Pure numpy over the raw per-period arrays written by a run: given a run
directory and a period, resolve the reward position active in that period,
compute per-timestep distance and egocentric bearing to it, and score every
neuron for distance-ramping and goal-direction tuning. Main entry point:
analyze_period().

On top of the two reference analyses, this module adds what a per-period,
per-cell classification needs to be defensible: per-neuron cyclic-shift nulls
for every statistic (significance is empirical, not asymptotic -- at n = 50,000
timesteps a trivial effect clears any nominal threshold), an iso-distance
ring-uniformity test that separates genuine distance cells from ordinary place
cells, wall-buffer exclusion, FDR-BH correction with an effect-size floor, and
a purely geometric rotation prediction for the reward-relocation condition.
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
WALL_BUFFER_CELLS = 1        # simulator's hard movement restriction (successor_representation.py convention)
EDGE_MARGIN_CELLS = 1        # additional soft margin against bearing/distance border effects (default)

# Periods whose active reward is B in a relocation run (n_cycles > 1);
# everything else defaults to A = config["reward_pos"]. B itself is never
# persisted in config.json (relocate_x/y/at_cycle are all None there) --
# it's the CLI default (0.7, 0.7), reproduced identically in
# paper_code/runs/select_best_seed_fig4.py:42-43 and paper_code/figures/fig4/utils.py.
RELOC_PHASE_B_PERIODS = {
    "after", "cycle_2", "cycle_2_goal", "goal_conditioned", "replay_reloc_postB",
}
DEFAULT_REWARD_B = (0.7, 0.7)


# ---------------------------------------------------------------------------
# Reward-position resolution
# ---------------------------------------------------------------------------

def _load_config(run_dir) -> dict:
    with open(Path(run_dir) / "config.json") as f:
        return json.load(f)


def resolve_period_reward_pos(run_dir, period: str, reward_b=DEFAULT_REWARD_B) -> np.ndarray:
    """Reward position active during `period`, read from run_dir/config.json
    with the relocation A/B special-case. Never hardcodes a reward position
    for the single-reward (nohead/head) conditions -- config["reward_pos"] is
    authoritative there. For relocation runs (config["n_cycles"] > 1),
    `period` must be one of RELOC_PHASE_B_PERIODS to get B; anything else
    (before/cycle_1/cycle_1_goal/replay_reloc_pred/replay_reloc_postA) gets
    A = config["reward_pos"]."""
    config = _load_config(run_dir)
    reward_a = np.asarray(config["reward_pos"], dtype=float)
    if config.get("n_cycles", 1) <= 1:
        return reward_a
    # Extinction runs (paper_code/model/ratinabox_extinction.py) also have n_cycles > 1 but
    # the reward never moves -- without this guard their post-cycle-1 periods
    # (after/cycle_2/cycle_2_goal/goal_conditioned) would silently be assigned
    # reward B. Extinction is identified by its own config flag.
    if config.get("extinguish_at_cycle", -1) >= 0:
        return reward_a
    return np.asarray(reward_b, dtype=float) if period in RELOC_PHASE_B_PERIODS else reward_a


# ---------------------------------------------------------------------------
# Wall-buffer filtering
# ---------------------------------------------------------------------------

def wall_buffer_mask(x: np.ndarray, y: np.ndarray, m: int = M_GRID,
                      wall_buffer: int = WALL_BUFFER_CELLS,
                      margin_cells: int = EDGE_MARGIN_CELLS) -> np.ndarray:
    """Boolean mask excluding timesteps within `margin_cells` extra grid
    cells of the simulator's already-enforced [wall_buffer, m-1-wall_buffer]
    movement boundary, where bearing/distance-to-goal sampling is
    systematically restricted (fewer viable headings/approach angles),
    which could otherwise masquerade as tuning."""
    lo = (wall_buffer + margin_cells) / m
    hi = 1.0 - (wall_buffer + margin_cells) / m
    return (x >= lo) & (x <= hi) & (y >= lo) & (y <= hi)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def geometry(x: np.ndarray, y: np.ndarray, theta: np.ndarray, reward_pos):
    """Per-timestep distance to reward and egocentric bearing to reward.

    Takes raw arrays (from load_period_activity) rather than a behavior
    DataFrame.

    ego_bearing in [-pi, pi], 0 = reward straight ahead of current heading.
    Also returns target_angle (allocentric bearing to reward), needed for
    the relocation geometric-rotation prediction.
    """
    reward_pos = np.asarray(reward_pos, dtype=float)
    dist = np.sqrt((x - reward_pos[0]) ** 2 + (y - reward_pos[1]) ** 2)
    target_angle = np.arctan2(reward_pos[1] - y, reward_pos[0] - x)
    ego = np.arctan2(np.sin(target_angle - theta), np.cos(target_angle - theta))
    return dist, ego, target_angle


def circular_diff(a, b):
    """Wrap-safe a-b in [-pi, pi]."""
    return np.angle(np.exp(1j * (np.asarray(a) - np.asarray(b))))


# ---------------------------------------------------------------------------
# Gauthier & Tank 2018 — reward-distance-modulated activity (ramping)
# ---------------------------------------------------------------------------

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
    """Per-neuron ramping index = corr(activity, -distance) (positive = more
    active nearer reward) and per-neuron activity-vs-distance tuning curve."""
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
    """Per-neuron angular tuning within iso-distance rings, occupancy-weighted
    across rings (N,). Low = fires alike all around a ring (a true distance
    cell); high = concentrated at one angle (place cell). Gates distance_cell in classify_cells,
    whose docstring explains why the gate is needed."""
    edges = np.linspace(dist.min(), dist.max(), n_dist_bins + 1)
    ring_id = np.clip(np.digitize(dist, edges) - 1, 0, n_dist_bins - 1)
    N = X.shape[1]
    ring_mrls = []
    ring_weights = []
    for r in range(n_dist_bins):
        in_ring = ring_id == r
        if in_ring.sum() < min_bin_count * n_angle_bins:
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
    """Cyclic-shift null for ring_angular_uniformity (shifts allo_angle,
    which like ego and theta is a deterministic function of the
    trajectory, so shifting it relative to activity preserves each series'
    own autocorrelation while breaking the true activity-geometry pairing)."""
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
    """Cyclic-shift null for the ramping index, mirroring bearing_tuning_null's
    logic but for -distance. Needed because cells are classified within a
    single period rather than by a paired pre/post delta, so each neuron needs
    its own null. By default returns the per-neuron
    95th percentile |ramping index| threshold (N,); pass return_samples=True
    to get the full (n_null, N) null matrix instead, needed for a proper
    continuous empirical p-value (and hence a meaningful FDR-BH step) rather
    than a crude binary above/below-threshold test."""
    rng = np.random.default_rng(seed)
    T = len(dist)
    null = np.empty((n_null, X.shape[1]), dtype=np.float32)
    for s in range(n_null):
        shift = int(rng.integers(1, T))
        null[s] = np.abs(_col_corr(X, -np.roll(dist, shift)))
    return null if return_samples else np.percentile(null, 95, axis=0)


# ---------------------------------------------------------------------------
# Sarel et al. 2017 — goal-vector (egocentric goal-direction) tuning
# ---------------------------------------------------------------------------

def circular_tuning(X: np.ndarray, angle: np.ndarray, nbins: int = 16):
    """Activity-weighted circular tuning curve of each neuron to `angle`
    (radians in [-pi, pi]) and its mean resultant length (MRL). Returns
    (mrl (N,), pref (N,), curve (N,nbins), bin_centres (nbins,)).

    Used for both egocentric goal-direction (angle = bearing to reward) and,
    as a control, allocentric head direction (angle = heading theta) --
    though for the 8-way discretized theta, prefer circular_tuning_exact
    below to avoid bin-aliasing.
    """
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
    """Like circular_tuning() but for an already-discretized angle (e.g. the
    8-way theta head-direction control) -- groups by exact value instead of
    digitizing into nbins edges, avoiding aliasing when nbins doesn't evenly
    divide (or mis-centers on) the true discrete states. Returns
    (mrl (N,), pref (N,), curve (N,n_vals), distinct_values (n_vals,))."""
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
    """Cyclic-shift null for circular_tuning_exact (the 8-way discretized
    head-direction control), mirroring bearing_tuning_null's logic but with
    exact-value grouping instead of bin digitization."""
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
    """Cyclic-shift null distribution of the bearing MRL, preserving the
    temporal autocorrelation of the trajectory. By default returns the
    per-neuron 95th percentile MRL threshold (N,); pass return_samples=True
    to get the full (n_null, N) null matrix for a proper continuous empirical
    p-value."""
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


# ---------------------------------------------------------------------------
# Cell classification (FDR-BH + effect-size floor)
# ---------------------------------------------------------------------------

def _empirical_p_from_null(observed: np.ndarray, null_thr_95: np.ndarray,
                            null_samples: np.ndarray | None = None) -> np.ndarray:
    """Convert a per-neuron 95th-percentile null threshold comparison into an
    approximate empirical p-value. If the full null distribution isn't kept
    (only its 95th percentile), we can only threshold at p=0.05; for a
    genuine FDR-BH step across many cells we want finer-grained p-values, so
    callers should prefer `null_samples` (n_null, N) when available."""
    if null_samples is not None:
        # p = (# null >= observed + 1) / (n_null + 1), per neuron
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
    """Classify cells from FDR-BH q-values plus a minimum effect size (at
    n=50,000 timesteps even r=0.02 is nominally significant, so q alone is
    not enough).

    A significant ramping index alone does not mean "distance cell": with one
    static reward, any localized field correlates with distance by geometry,
    and the raw test flags 50-80% of all cells as many in the reward-naive
    random walk as in the goal-directed one. A low `ring_mrl` is therefore
    required on top; omit it and distance_cell degrades to that raw test.

    Returns boolean (N,) masks: distance_cell, ramping_only_cell (unvetoed,
    for comparison), goal_direction_cell, head_direction_cell,
    goal_vector_cell (Sarel's strict conjunction of distance + direction),
    goal_specific_cell (goal_direction and not head_direction).
    """
    ramp_q = fdr_bh(ramp_p)
    mrl_ego_q = fdr_bh(mrl_ego_p)
    mrl_hd_q = fdr_bh(mrl_hd_p)

    ramping_sig = (ramp_q <= alpha) & (np.abs(ramp) >= effect_floor_ramp)
    if ring_mrl is not None:
        # genuine distance cell: significant across-ring modulation and
        # roughly uniform firing within each ring (low ring_mrl)
        distance_cell = ramping_sig & (ring_mrl < effect_floor_ring)
    else:
        distance_cell = ramping_sig
    goal_direction_cell = (mrl_ego_q <= alpha) & (mrl_ego >= effect_floor_mrl)
    head_direction_cell = (mrl_hd_q <= alpha) & (mrl_hd >= effect_floor_mrl)
    goal_vector_cell = distance_cell & goal_direction_cell
    goal_specific_cell = goal_direction_cell & ~head_direction_cell

    return {
        "distance_cell": distance_cell,
        "ramping_only_cell": ramping_sig,   # pre-ring-control population, for comparison/transparency
        "goal_direction_cell": goal_direction_cell,
        "goal_vector_cell": goal_vector_cell,
        "head_direction_cell": head_direction_cell,
        "goal_specific_cell": goal_specific_cell,
    }


# ---------------------------------------------------------------------------
# Relocation rotation test
# ---------------------------------------------------------------------------

def geometric_rotation(run_dir, period_a: str = "cycle_1_goal",
                        period_b: str = "cycle_2_goal",
                        reward_b=DEFAULT_REWARD_B) -> float:
    """Delta_geom: the population-level (behavior-independent) rotation of
    target_angle induced purely by relocating the reward A->B, evaluated at
    phase-A's own occupied positions (so it isolates the geometric effect of
    reward relocation from any behavioral-sampling difference between the
    two goal-directed phases). Returns a scalar (circular mean of
    target_angle_B(x_A,y_A) - target_angle_A(x_A,y_A))."""
    data_dir = Path(run_dir) / "data"
    _, x_a, y_a, _ = load_period_activity(str(data_dir), period_a)
    reward_a = resolve_period_reward_pos(run_dir, period_a, reward_b)
    reward_b_pos = resolve_period_reward_pos(run_dir, period_b, reward_b)
    ta_a = np.arctan2(reward_a[1] - y_a, reward_a[0] - x_a)
    ta_b = np.arctan2(reward_b_pos[1] - y_a, reward_b_pos[0] - x_a)  # same positions, hypothetical B
    return float(np.angle(np.mean(np.exp(1j * (ta_b - ta_a)))))


def bootstrap_circular_mean_ci(angles: np.ndarray, n_boot: int = 2000,
                                seed: int = 0, ci: float = 0.95):
    """Bootstrap CI on the circular mean of `angles` (resample with
    replacement). Returns (circular_mean, (lo, hi))."""
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


# ---------------------------------------------------------------------------
# Driver: one-stop per-(seed, period) analysis
# ---------------------------------------------------------------------------

def analyze_period(run_dir, period: str, hidden_dim: int = 512, m: int = M_GRID,
                    wall_buffer: int = WALL_BUFFER_CELLS, margin_cells: int = EDGE_MARGIN_CELLS,
                    nbins_dist: int = 12, nbins_bearing: int = 16,
                    n_null: int = 100, seed: int = 0, reward_b=DEFAULT_REWARD_B) -> dict | None:
    """One-stop per-(seed, period) analysis: loads activity+behavior, resolves
    reward_pos, applies wall-buffer filtering, computes ramping index +
    ego-bearing MRL/pref + hd MRL/pref + their cyclic-shift null p-values.
    Returns a dict of (hidden_dim,)-shaped arrays plus scalars, ready for
    classify_cells() and cross-seed aggregation. Returns None if the period
    is missing or too few timesteps survive wall-buffer filtering."""
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

    # hd control: 8-way discretized theta, exact-value grouping (avoids
    # bin-aliasing that circular_tuning(nbins=16) would introduce).
    mrl_hd, pref_hd, curve_hd, hd_vals = circular_tuning_exact(X, theta)
    mrl_hd_null = circular_tuning_exact_null(X, theta, n_null=n_null, seed=seed + 2,
                                              return_samples=True)
    mrl_hd_p = _empirical_p_from_null(mrl_hd, None, null_samples=mrl_hd_null)

    # Ring-based angular-uniformity control (required for a valid Gauthier-
    # style distance_cell classification -- see ring_angular_uniformity's
    # docstring for why raw ramping significance alone is insufficient with
    # a single static reward).
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
