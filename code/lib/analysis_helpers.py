"""Period loaders, the frozen-replay shift, the entropy-collapse gate and two composite contrasts.

Used by most figure generators and by aggregate/ and runs/.
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
    """Paths of a period's behaviour and activation pickles."""
    beh = os.path.join(data_dir, f"df_behavior_{period}.pkl")
    act = os.path.join(data_dir, f"df_activations_{period}.pkl")
    return beh, act

def has_period(data_dir: str, period: str) -> bool:
    """Whether both pickles of a period exist."""
    beh, act = period_paths(data_dir, period)
    return os.path.exists(beh) and os.path.exists(act)


def load_period_centers(data_dir: str, period: str, m: int = 32, hidden_dim: int = 512):
    """Skaggs score, centre of mass and peak for every unit of one period."""
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
    """Raw activity and trajectory of one period as (X, x, y, theta)."""
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
    """Euclidean distance from each centre to the reward."""
    reward = np.asarray(reward, dtype=float)
    return np.linalg.norm(centers - reward[None, :], axis=1)


def frozen_replay_gate(sk_pre, sk_post, skaggs_threshold: float):
    """Units passing the Skaggs threshold in both periods, and their count."""
    gate = (np.asarray(sk_pre) >= skaggs_threshold) & (np.asarray(sk_post) >= skaggs_threshold)
    n = int(gate.sum())
    return gate, n


def frozen_replay_delta_from_centers(sk_pre, centers_pre, sk_post, centers_post,
                                      reward_pos, skaggs_threshold: float = 0.3,
                                      min_cells: int = 10):
    """Distance to reward before and after for gated units; None if fewer than min_cells."""
    gate, n = frozen_replay_gate(sk_pre, sk_post, skaggs_threshold)
    if n < min_cells:
        return None
    reward_pos = np.asarray(reward_pos, dtype=float)
    d_pre = dist_to_reward(np.asarray(centers_pre)[gate], reward_pos)
    d_post = dist_to_reward(np.asarray(centers_post)[gate], reward_pos)
    return {"n": n, "gate": gate, "d_pre": d_pre, "d_post": d_post}


def frozen_replay_delta(data_dir, period_pre: str, period_post: str, reward_pos,
                         skaggs_threshold: float = 0.3, min_cells: int = 10):
    """Mean peak shift and fraction moved toward reward (positive = toward)."""
    try:
        sk_pre, _, pk_pre = load_period_centers(str(data_dir), period_pre)
        sk_post, _, pk_post = load_period_centers(str(data_dir), period_post)
    except Exception as e:
        print(f"[skip] {data_dir} {period_pre}->{period_post}: {type(e).__name__}: {e}", file=sys.stderr)
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


MIN_ENTROPY = 5.5
SKAGGS_THRESHOLD = 0.3
M_GRID = 32


def peak_entropy(data_dir, period: str, skaggs_threshold: float = SKAGGS_THRESHOLD):
    """Shannon entropy (bits) of field-peak locations; low = fields collapsed."""
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
    """True if every listed period has peak entropy >= min_entropy."""
    ents = [peak_entropy(data_dir, p) for p in periods]
    if any(e is None for e in ents):
        return False              # missing period or no place cells = not clean
    return min(ents) >= min_entropy


def _mean_delta_to(data_dir, period_pre: str, period_post: str, reward,
                   skaggs_threshold: float = SKAGGS_THRESHOLD, min_cells: int = 10):
    """Mean peak shift toward `reward` for units gated in both periods (negative = toward)."""
    try:
        sk_a, _, pk_a = load_period_centers(str(data_dir), period_pre)
        sk_b, _, pk_b = load_period_centers(str(data_dir), period_post)
    except Exception as e:
        print(f"[skip] {data_dir} {period_pre}->{period_post}: {type(e).__name__}: {e}", file=sys.stderr)
        return None
    gate = (sk_a >= skaggs_threshold) & (sk_b >= skaggs_threshold)
    if gate.sum() < min_cells:
        return None
    reward = np.asarray(reward, dtype=float)
    d = (np.linalg.norm(pk_b[gate] - reward, axis=1)
         - np.linalg.norm(pk_a[gate] - reward, axis=1))   # negative = toward reward
    return {"n": int(gate.sum()), "mean": float(d.mean()),
            "frac_toward": float((d < 0).mean()), "per_cell": d}


def readout_dissociation(nohead_dir, head_dir, reward,
                         behavioural=("cycle_1", "cycle_1_goal"),
                         frozen=("replay_c1_pred", "replay_c1_postrl"),
                         skaggs_threshold: float = SKAGGS_THRESHOLD):
    """Behavioural vs frozen-replay shift for a matched pair, and their interaction."""
    out = {}
    for cond, d in (("nh", nohead_dir), ("hd", head_dir)):
        for tag, (pre, post) in (("beh", behavioural), ("frz", frozen)):
            r = _mean_delta_to(d, pre, post, reward, skaggs_threshold)
            if r is None:
                return None      # all-or-nothing per seed
            out[f"{cond}_{tag}"] = r["mean"]
            out[f"{cond}_{tag}_frac"] = r["frac_toward"]
            out[f"{cond}_{tag}_n"] = r["n"]
    out["interaction"] = (out["hd_beh"] - out["hd_frz"]) - (out["nh_beh"] - out["nh_frz"])
    return out


def relocation_asymmetry(data_dir, reward_a, reward_b,
                         periods=("replay_reloc_pred", "replay_reloc_postA",
                                  "replay_reloc_postB"),
                         skaggs_threshold: float = 0.15):  # relocation uses 0.15
    """Frozen-replay shifts toward A and then B for one relocation seed (negative = toward)."""
    pre, post_a, post_b = periods
    ra = _mean_delta_to(data_dir, pre, post_a, reward_a, skaggs_threshold)   # toward A
    rb = _mean_delta_to(data_dir, post_a, post_b, reward_b, skaggs_threshold)  # toward B
    if ra is None or rb is None:
        return None
    return {
        "n_A": ra["n"], "n_B": rb["n"],
        "delta_A": ra["mean"], "delta_B": rb["mean"],
        "frac_toward_A": ra["frac_toward"], "frac_toward_B": rb["frac_toward"],
        "abs_gap": abs(ra["mean"]) - abs(rb["mean"]),
    }
