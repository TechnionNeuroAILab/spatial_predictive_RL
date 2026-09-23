"""
Period-level loading and frozen-replay diagnostics for the paper_code pipeline.

Everything here is keyed to a run directory plus a period tag, reading the
df_behavior_<period>.pkl / df_activations_<period>.pkl pair that model/ writes.
This module adds the per-period loaders, the place-cell and entropy-collapse gates, and the
contrasts built on top of them. Imported by most figure generators, by
aggregate/ and by runs/.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from ratemaps import (
    build_rate_maps,
    compute_skaggs,
    smooth_and_norm,
    field_center_of_mass,
    field_peak,
)

def period_paths(data_dir: str, period: str) -> tuple[str, str]:
    beh = os.path.join(data_dir, f"df_behavior_{period}.pkl")
    act = os.path.join(data_dir, f"df_activations_{period}.pkl")
    return beh, act

def has_period(data_dir: str, period: str) -> bool:
    beh, act = period_paths(data_dir, period)
    return os.path.exists(beh) and os.path.exists(act)


def load_period_centers(data_dir: str, period: str, m: int = 32, hidden_dim: int = 512):
    """Load one period's pickles and reduce them to (skaggs, com, peak) per unit."""
    beh_path, act_path = period_paths(data_dir, period)
    if not (os.path.exists(beh_path) and os.path.exists(act_path)):
        raise FileNotFoundError(f"Missing period '{period}' in {data_dir}")
    beh = pd.read_pickle(beh_path)
    act = pd.read_pickle(act_path)
    rm, occ = build_rate_maps(beh, act, m, hidden_dim)
    sk = compute_skaggs(rm, occ)
    # com and peak are two different centre estimates; callers pick one.
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
    """Load one period's raw activity and trajectory as (X, x, y, theta)."""
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
    # A missing theta column gives a constant heading.
    theta = beh["theta"].to_numpy(dtype=float) if "theta" in beh.columns else np.zeros(n_steps)
    return X, x, y, theta

def dist_to_reward(centers: np.ndarray, reward) -> np.ndarray:
    reward = np.asarray(reward, dtype=float)
    return np.linalg.norm(centers - reward[None, :], axis=1)


# ---------------------------------------------------------------------------
# Frozen-replay diagnostic (shared by figures 2, 3, 5, and select_best_seed_fig4)
# ---------------------------------------------------------------------------

def frozen_replay_gate(sk_pre, sk_post, skaggs_threshold: float):
    """Cells passing the Skaggs place-cell gate in both periods.

    Parameters
    ----------
    sk_pre, sk_post : array-like (hidden_dim,)
    skaggs_threshold : float

    Returns
    -------
    gate : ndarray (hidden_dim,), bool
        True where the unit passes the gate in both periods.
    n : int
        Number of units passing.
    """
    gate = (np.asarray(sk_pre) >= skaggs_threshold) & (np.asarray(sk_post) >= skaggs_threshold)
    n = int(gate.sum())
    return gate, n


def frozen_replay_delta_from_centers(sk_pre, centers_pre, sk_post, centers_post,
                                      reward_pos, skaggs_threshold: float = 0.3,
                                      min_cells: int = 10):
    """Per-cell distance to reward in both periods of a frozen-replay pair, place cells only."""
    gate, n = frozen_replay_gate(sk_pre, sk_post, skaggs_threshold)
    if n < min_cells:
        return None
    reward_pos = np.asarray(reward_pos, dtype=float)
    d_pre = dist_to_reward(np.asarray(centers_pre)[gate], reward_pos)
    d_post = dist_to_reward(np.asarray(centers_post)[gate], reward_pos)
    # Distances only -- the caller forms the delta, since call sites disagree
    # on which sign means "toward reward".
    return {"n": n, "gate": gate, "d_pre": d_pre, "d_post": d_post}


def frozen_replay_delta(data_dir, period_pre: str, period_post: str, reward_pos,
                         skaggs_threshold: float = 0.3, min_cells: int = 10):
    """Load peak centres for two periods and add the mean/fraction moved toward reward."""
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
    # positive = toward reward -- the opposite convention to _mean_delta_to below.
    delta = res["d_pre"] - res["d_post"]
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
    """Shannon entropy (bits) of where place-cell field peaks sit on the arena grid."""
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
    return float(-np.sum(p * np.log2(p)))   # low = peaks collapsed into one spot


def seed_is_clean(data_dir, periods, min_entropy: float = MIN_ENTROPY) -> bool:
    """Whether every listed period escapes the peak-entropy collapse gate."""
    ents = [peak_entropy(data_dir, p) for p in periods]
    if any(e is None for e in ents):
        return False              # missing period or no place cells = not clean
    return min(ents) >= min_entropy

# ---------------------------------------------------------------------------
# Read-out dissociation and relocation asymmetry
# ---------------------------------------------------------------------------

def _mean_delta_to(data_dir, period_pre: str, period_post: str, reward,
                   skaggs_threshold: float = SKAGGS_THRESHOLD, min_cells: int = 10):
    """Mean change in distance to `reward` between two periods; negative = toward."""
    try:
        sk_a, _, pk_a = load_period_centers(str(data_dir), period_pre)
        sk_b, _, pk_b = load_period_centers(str(data_dir), period_post)
    except Exception:
        return None
    gate = (sk_a >= skaggs_threshold) & (sk_b >= skaggs_threshold)
    if gate.sum() < min_cells:
        return None
    reward = np.asarray(reward, dtype=float)
    # negative = moved toward reward; opposite sign to frozen_replay_delta above.
    d = (np.linalg.norm(pk_b[gate] - reward, axis=1)
         - np.linalg.norm(pk_a[gate] - reward, axis=1))
    return {"n": int(gate.sum()), "mean": float(d.mean()),
            "frac_toward": float((d < 0).mean()), "per_cell": d}


def readout_dissociation(nohead_dir, head_dir, reward,
                         behavioural=("cycle_1", "cycle_1_goal"),
                         frozen=("replay_c1_pred", "replay_c1_postrl"),
                         skaggs_threshold: float = SKAGGS_THRESHOLD):
    """Per-seed behavioural vs frozen-replay shift for a matched no-head/head pair."""
    out = {}
    # All four measurements use the same cells and the same gate -- that is what
    # makes the condition x read-out interaction interpretable.
    for cond, d in (("nh", nohead_dir), ("hd", head_dir)):
        for tag, (pre, post) in (("beh", behavioural), ("frz", frozen)):
            r = _mean_delta_to(d, pre, post, reward, skaggs_threshold)
            if r is None:
                return None      # all-or-nothing: a partial seed would skew it
            out[f"{cond}_{tag}"] = r["mean"]
            out[f"{cond}_{tag}_frac"] = r["frac_toward"]
            out[f"{cond}_{tag}_n"] = r["n"]
    # "beh" is occupancy-confounded, "frz" is weights-only; the caller aggregates
    # this interaction across seeds and tests it.
    out["interaction"] = (out["hd_beh"] - out["hd_frz"]) - (out["nh_beh"] - out["nh_frz"])
    return out


def relocation_asymmetry(data_dir, reward_a, reward_b,
                         periods=("replay_reloc_pred", "replay_reloc_postA",
                                  "replay_reloc_postB"),
                         skaggs_threshold: float = 0.15):  # not the module's 0.3; matches fig4
    """Phase-A and phase-B frozen-replay shifts for one relocation seed; negative = toward."""
    pre, post_a, post_b = periods
    ra = _mean_delta_to(data_dir, pre, post_a, reward_a, skaggs_threshold)   # toward A
    rb = _mean_delta_to(data_dir, post_a, post_b, reward_b, skaggs_threshold)  # toward B
    if ra is None or rb is None:
        return None
    return {
        "n_A": ra["n"], "n_B": rb["n"],
        "delta_A": ra["mean"], "delta_B": rb["mean"],
        "frac_toward_A": ra["frac_toward"], "frac_toward_B": rb["frac_toward"],
        # abs_gap lets a paired test across seeds establish asymmetry as a
        # tested claim, rather than reporting delta_A and delta_B separately.
        "abs_gap": abs(ra["mean"]) - abs(rb["mean"]),
    }
