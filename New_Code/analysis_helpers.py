from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from plot_goal_directed_maps import (
    build_rate_maps,
    compute_skaggs,
    smooth_and_norm,
    field_center_of_mass,
)
from plot_replay_v11 import field_peak

def period_paths(data_dir: str, period: str) -> tuple[str, str]:
    beh = os.path.join(data_dir, f"df_behavior_{period}.pkl")
    act = os.path.join(data_dir, f"df_activations_{period}.pkl")
    return beh, act

def has_period(data_dir: str, period: str) -> bool:
    beh, act = period_paths(data_dir, period)
    return os.path.exists(beh) and os.path.exists(act)


def load_period_centers(data_dir: str, period: str, m: int = 32, hidden_dim: int = 512):
    beh_path, act_path = period_paths(data_dir, period)
    if not (os.path.exists(beh_path) and os.path.exists(act_path)):
        raise FileNotFoundError(f"Missing period '{period}' in {data_dir}")
    beh = pd.read_pickle(beh_path)
    act = pd.read_pickle(act_path)
    rm, occ = build_rate_maps(beh, act, m, hidden_dim)
    sk = compute_skaggs(rm, occ)
    com = np.array(
        [field_center_of_mass(smooth_and_norm(rm[n])) for n in range(hidden_dim)],
        dtype=float,
    )
    peak = np.array(
        [field_peak(smooth_and_norm(rm[n])) for n in range(hidden_dim)],
        dtype=float,
    )
    del rm
    return sk, com, peak


def load_period_activity(data_dir: str, period: str, hidden_dim: int = 512):
    beh_path, act_path = period_paths(data_dir, period)
    if not (os.path.exists(beh_path) and os.path.exists(act_path)):
        raise FileNotFoundError(f"Missing period '{period}' in {data_dir}")
    beh = pd.read_pickle(beh_path)
    act = pd.read_pickle(act_path)
    n_steps = len(beh)
    if len(act) != n_steps * hidden_dim:
        raise ValueError(
            f"Activation rows ({len(act)}) != n_steps*hidden_dim "
            f"({n_steps}*{hidden_dim}); cannot reshape safely for '{period}'."
        )
    X = act["activity"].to_numpy(dtype=float).reshape(n_steps, hidden_dim)
    x = beh["x"].to_numpy(dtype=float)
    y = beh["y"].to_numpy(dtype=float)
    theta = beh["theta"].to_numpy(dtype=float) if "theta" in beh.columns else np.zeros(n_steps)
    return X, x, y, theta

def dist_to_reward(centers: np.ndarray, reward) -> np.ndarray:
    reward = np.asarray(reward, dtype=float)
    return np.linalg.norm(centers - reward[None, :], axis=1)


# ---------------------------------------------------------------------------
# Frozen-replay diagnostic (shared by figures 2, 3, 5, and select_best_seed_fig4)
#
# The frozen-replay diagnostic compares field centers on the SAME trajectory
# using PRE-RL weights vs POST-RL weights. Because the trajectory is
# identical, any field-center shift can only come from weight changes -- a
# genuine H2 (RPE-driven, real-remapping) signature, uncontaminated by
# occupancy/exploration differences.
# ---------------------------------------------------------------------------

def frozen_replay_gate(sk_pre, sk_post, skaggs_threshold: float, min_cells: int = 10):
    """Boolean mask of cells that are place cells (Skaggs >= threshold) in
    BOTH periods, plus the count of cells passing. Returns (gate, n)."""
    gate = (np.asarray(sk_pre) >= skaggs_threshold) & (np.asarray(sk_post) >= skaggs_threshold)
    n = int(gate.sum())
    return gate, n


def frozen_replay_delta_from_centers(sk_pre, centers_pre, sk_post, centers_post,
                                      reward_pos, skaggs_threshold: float = 0.3,
                                      min_cells: int = 10):
    """
    Generic frozen-replay diagnostic on already-loaded field-center arrays
    (works for either COM or peak centers -- caller picks which to pass).

    Gates on cells that are place cells in BOTH the pre- and post-weight
    periods, then returns each gated cell's distance to reward_pos in each
    period. Sign/direction of any "delta" is left to the caller, since
    different call sites use opposite conventions.

    Returns {"n", "gate", "d_pre", "d_post"} or None if fewer than
    min_cells cells pass the gate.
    """
    gate, n = frozen_replay_gate(sk_pre, sk_post, skaggs_threshold, min_cells)
    if n < min_cells:
        return None
    reward_pos = np.asarray(reward_pos, dtype=float)
    d_pre = dist_to_reward(np.asarray(centers_pre)[gate], reward_pos)
    d_post = dist_to_reward(np.asarray(centers_post)[gate], reward_pos)
    return {"n": n, "gate": gate, "d_pre": d_pre, "d_post": d_post}


def frozen_replay_delta(data_dir, period_pre: str, period_post: str, reward_pos,
                         skaggs_threshold: float = 0.3, min_cells: int = 10):
    """
    Path-based convenience wrapper: loads period_pre/period_post PEAK centers
    from disk, then runs the frozen-replay diagnostic on them.

    Returns {"n", "gate", "d_pre", "d_post", "mean_delta", "frac_toward"}
    (positive mean_delta/frac_toward = moved TOWARD reward_pos), or None if
    data is missing or too few cells pass the gate.
    """
    try:
        sk_pre, _, pk_pre = load_period_centers(str(data_dir), period_pre)
        sk_post, _, pk_post = load_period_centers(str(data_dir), period_post)
    except Exception:
        return None
    res = frozen_replay_delta_from_centers(
        sk_pre, pk_pre, sk_post, pk_post, reward_pos, skaggs_threshold, min_cells
    )
    if res is None:
        return None
    delta = res["d_pre"] - res["d_post"]   # positive = toward reward
    return {
        **res,
        "mean_delta": float(delta.mean()),
        "frac_toward": float((delta > 0).mean()),
    }


# ---------------------------------------------------------------------------
# Entropy collapse gate (shared seed-quality check)
# ---------------------------------------------------------------------------

MIN_ENTROPY = 5.5
SKAGGS_THRESHOLD = 0.3
M_GRID = 32


def peak_entropy(data_dir, period: str, skaggs_threshold: float = SKAGGS_THRESHOLD):
    """Shannon entropy (bits) of the field-peak distribution over an
    M_GRID x M_GRID grid. Returns None if the period is missing or no cells
    pass the Skaggs gate."""
    if not has_period(str(data_dir), period):
        return None
    sk, _, peak = load_period_centers(str(data_dir), period)
    place = np.where(sk >= skaggs_threshold)[0]
    if len(place) == 0:
        return None
    pk = peak[place]
    binned = (np.clip(pk, 0, 0.9999) * M_GRID).astype(int)
    counts = np.zeros((M_GRID, M_GRID), dtype=int)
    for bx, by in binned:
        counts[bx, by] += 1
    nz = counts.flatten()
    nz = nz[nz > 0]
    p = nz / nz.sum()
    return float(-np.sum(p * np.log2(p)))


def seed_is_clean(data_dir, periods, min_entropy: float = MIN_ENTROPY) -> bool:
    """A seed is 'clean' if every listed period's peak-entropy is above the
    collapse gate (i.e. field peaks haven't collapsed into a tiny region)."""
    ents = [peak_entropy(data_dir, p) for p in periods]
    if any(e is None for e in ents):
        return False
    return min(ents) >= min_entropy

# ---------------------------------------------------------------------------
# Read-out dissociation and relocation asymmetry
# ---------------------------------------------------------------------------

def _mean_delta_to(data_dir, period_pre: str, period_post: str, reward,
                   skaggs_threshold: float = SKAGGS_THRESHOLD, min_cells: int = 10):
    """Mean change in distance to `reward` between two periods.

    NEGATIVE = fields moved TOWARD the reward. (Note this is the opposite sign
    convention to frozen_replay_delta() above, which returns positive = toward;
    that function is kept as-is for backwards compatibility with its existing
    callers. Prefer this one for new code and state the convention explicitly.)
    """
    try:
        sk_a, _, pk_a = load_period_centers(str(data_dir), period_pre)
        sk_b, _, pk_b = load_period_centers(str(data_dir), period_post)
    except Exception:
        return None
    gate = (sk_a >= skaggs_threshold) & (sk_b >= skaggs_threshold)
    if gate.sum() < min_cells:
        return None
    reward = np.asarray(reward, dtype=float)
    d = (np.linalg.norm(pk_b[gate] - reward, axis=1)
         - np.linalg.norm(pk_a[gate] - reward, axis=1))
    return {"n": int(gate.sum()), "mean": float(d.mean()),
            "frac_toward": float((d < 0).mean()), "per_cell": d}


def readout_dissociation(nohead_dir, head_dir, reward,
                         behavioural=("cycle_1", "cycle_1_goal"),
                         frozen=("replay_c1_pred", "replay_c1_postrl"),
                         skaggs_threshold: float = SKAGGS_THRESHOLD):
    """Per-seed behavioural vs frozen-replay shift for a matched no-head/head pair.

    Tests whether the two read-outs of "reward remapping" dissociate: the
    behavioural (occupancy-confounded) contrast and the frozen-replay
    (weight-only) contrast are computed on the SAME cells with the SAME gate,
    so the condition x read-out interaction is interpretable.

    Returns None unless all four measurements succeed. Caller aggregates across
    seeds and tests the interaction (head_beh - head_frz) - (nh_beh - nh_frz).
    """
    out = {}
    for cond, d in (("nh", nohead_dir), ("hd", head_dir)):
        for tag, (pre, post) in (("beh", behavioural), ("frz", frozen)):
            r = _mean_delta_to(d, pre, post, reward, skaggs_threshold)
            if r is None:
                return None
            out[f"{cond}_{tag}"] = r["mean"]
            out[f"{cond}_{tag}_frac"] = r["frac_toward"]
            out[f"{cond}_{tag}_n"] = r["n"]
    out["interaction"] = (out["hd_beh"] - out["hd_frz"]) - (out["nh_beh"] - out["nh_frz"])
    return out


def relocation_asymmetry(data_dir, reward_a, reward_b,
                         periods=("replay_reloc_pred", "replay_reloc_postA",
                                  "replay_reloc_postB"),
                         skaggs_threshold: float = 0.15):
    """Phase-A and phase-B frozen-replay shifts for one relocation seed.

    Phase A: pre -> postA, distance to reward A. Phase B: postA -> postB,
    distance to reward B. Both negative = toward the respective reward.
    Returns the two means plus |dA| - |dB|, so a paired test across seeds can
    establish ASYMMETRY as a tested claim rather than two separately-reported
    numbers.
    """
    pre, post_a, post_b = periods
    ra = _mean_delta_to(data_dir, pre, post_a, reward_a, skaggs_threshold)
    rb = _mean_delta_to(data_dir, post_a, post_b, reward_b, skaggs_threshold)
    if ra is None or rb is None:
        return None
    return {
        "n_A": ra["n"], "n_B": rb["n"],
        "delta_A": ra["mean"], "delta_B": rb["mean"],
        "frac_toward_A": ra["frac_toward"], "frac_toward_B": rb["frac_toward"],
        "abs_gap": abs(ra["mean"]) - abs(rb["mean"]),
    }
