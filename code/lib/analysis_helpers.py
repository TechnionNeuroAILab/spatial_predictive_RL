"""Period loaders, the frozen-replay shift, the entropy-collapse gate and two composite contrasts.

Used by most figure generators and by aggregate/ and runs/.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
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
    beh = os.path.join(data_dir, f"df_behavior_{period}.parquet")
    act = os.path.join(data_dir, f"df_activations_{period}.parquet")
    return beh, act

def has_period(data_dir: str, period: str) -> bool:
    """Whether both pickles of a period exist."""
    beh, act = period_paths(data_dir, period)
    return os.path.exists(beh) and os.path.exists(act)


# Goal-directed probes run short episodes: on every goal arrival the agent is
# teleported to a random start and the recurrent state is reset. The first
# steps after a reset carry that reset transient rather than position-driven
# activity, so behavioural rate maps drop them (as the occupancy-matched
# estimator always has). Random-walk probes and replays never reset.
TELEPORT_THR = 0.05          # a step longer than this is a teleport
N_TRIM_AFTER_RESET = 5       # steps dropped after each reset


def is_goal_directed(period: str) -> bool:
    """Goal-directed probes are the only periods that reset after goal arrival."""
    return period.endswith("_goal") or period == "goal_conditioned"


def _resolve_trim(period: str, trim_resets) -> bool:
    """None means the default rule: trim exactly the goal-directed periods."""
    return is_goal_directed(period) if trim_resets is None else bool(trim_resets)


def reset_keep_mask(x: np.ndarray, y: np.ndarray, n_trim: int = N_TRIM_AFTER_RESET) -> np.ndarray:
    """True for every step except the first `n_trim` after each teleport."""
    keep = np.ones(len(x), dtype=bool)
    step = np.hypot(np.diff(x), np.diff(y))
    for r in np.where(step > TELEPORT_THR)[0] + 1:
        keep[r:r + n_trim] = False
    return keep


def load_period_frames(data_dir: str, period: str, hidden_dim: int = 512,
                       trim_resets: bool | None = None):
    """Behaviour and activation frames of one period, optionally without reset transients."""
    beh_path, act_path = period_paths(data_dir, period)
    if not (os.path.exists(beh_path) and os.path.exists(act_path)):
        raise FileNotFoundError(f"Missing period '{period}' in {data_dir}")
    beh = pd.read_parquet(beh_path)
    act = pd.read_parquet(act_path)
    if _resolve_trim(period, trim_resets):
        keep = reset_keep_mask(beh["x"].to_numpy(float), beh["y"].to_numpy(float))
        a = act["activity"].to_numpy().reshape(len(beh), hidden_dim)[keep]
        beh = beh.loc[keep].reset_index(drop=True)
        act = pd.DataFrame({"activity": a.reshape(-1)})
    return beh, act


def load_period_centers(data_dir: str, period: str, m: int = 32, hidden_dim: int = 512,
                        trim_resets: bool | None = None):
    """Skaggs score, centre of mass and peak for every unit of one period."""
    beh, act = load_period_frames(data_dir, period, hidden_dim, trim_resets)
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


# v2: occupancy-weighted Skaggs information and reset-trimmed goal-directed periods
_CENTERS_CACHE = Path(__file__).resolve().parents[2] / "results" / "aggregate" / "centers_cache_v2"


def load_period_centers_cached(data_dir: str, period: str, trim_resets: bool | None = None):
    """`load_period_centers` memoised on disk, keyed by run path and parquet mtime."""
    trim_resets = _resolve_trim(period, trim_resets)
    data_dir = Path(data_dir).resolve()
    _, act_path = period_paths(str(data_dir), period)
    if not os.path.exists(act_path):
        raise FileNotFoundError(f"Missing period '{period}' in {data_dir}")
    rel = data_dir.parent.relative_to(_CENTERS_CACHE.parents[1])
    path = _CENTERS_CACHE / rel / f"{period}{'_trim' if trim_resets else ''}.npz"
    mtime = os.path.getmtime(act_path)
    if path.exists():
        z = np.load(path)
        if float(z["mtime"]) == mtime:
            return z["sk"], z["com"], z["peak"]
    sk, com, peak = load_period_centers(str(data_dir), period, trim_resets=trim_resets)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, sk=sk, com=com, peak=peak, mtime=mtime)
    os.replace(tmp, path)
    return sk, com, peak


def load_period_activity(data_dir: str, period: str, hidden_dim: int = 512,
                         trim_resets: bool | None = None):
    """Raw activity and trajectory of one period as (X, x, y, theta).

    Goal-directed periods lose the steps right after each reset (see
    reset_keep_mask) unless `trim_resets=False`.
    """
    beh_path, act_path = period_paths(data_dir, period)
    if not (os.path.exists(beh_path) and os.path.exists(act_path)):
        raise FileNotFoundError(f"Missing period '{period}' in {data_dir}")
    beh = pd.read_parquet(beh_path)
    act = pd.read_parquet(act_path)
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
    if _resolve_trim(period, trim_resets):
        keep = reset_keep_mask(x, y)
        X, x, y, theta = X[keep], x[keep], y[keep], theta[keep]
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


# Entropy-collapse gate. 5.0 bits is the highest single threshold that keeps
# at least 50 networks in every family under the unified rate-map rules
# (occupancy-weighted spatial information, reset steps dropped).
MIN_ENTROPY = 5.0
SKAGGS_THRESHOLD = 0.3
M_GRID = 32
_REPO = Path(__file__).resolve().parents[2]


def campaign_passers(family: str) -> set[int]:
    """Seeds in the analysed cohort of a family (`paired`, `reloc`, `extinction`).

    The cohort is every launched network whose minimum field-peak entropy over
    the family's gate periods is >= MIN_ENTROPY, computed with the same rate
    maps as the analyses. Written by code/aggregate/cohorts.py to
    results/aggregate/cohorts.json.
    """
    data = json.loads((_REPO / "results" / "aggregate" / "cohorts.json").read_text())
    return {int(s) for s in data[family]["seeds"]}


def peak_entropy(data_dir, period: str, skaggs_threshold: float = SKAGGS_THRESHOLD,
                 cached: bool = False):
    """Shannon entropy (bits) of field-peak locations; low = fields collapsed."""
    if not has_period(str(data_dir), period):
        return None
    load = load_period_centers_cached if cached else load_period_centers
    sk, _, peak = load(str(data_dir), period)
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


def seed_is_clean(data_dir, periods, min_entropy: float | None = None) -> bool:
    """True if every listed period has peak entropy >= min_entropy (default MIN_ENTROPY)."""
    min_entropy = MIN_ENTROPY if min_entropy is None else min_entropy
    ents = [peak_entropy(data_dir, p) for p in periods]
    if any(e is None for e in ents):
        return False              # missing period or no place cells = not clean
    return min(ents) >= min_entropy


def _mean_delta_to(data_dir, period_pre: str, period_post: str, reward,
                   skaggs_threshold: float = SKAGGS_THRESHOLD, min_cells: int = 10,
                   center: str = "peak", cached: bool = False, trim_post: bool | None = None):
    """Mean field shift toward `reward` for units gated in both periods (negative = toward).

    `center` is "peak" or "com"; `cached` reads centres through the on-disk cache.
    """
    load = load_period_centers_cached if cached else load_period_centers
    idx = {"com": 1, "peak": 2}[center]
    try:
        a = load(str(data_dir), period_pre)                     # goal-directed periods trim
        b = load(str(data_dir), period_post, trim_resets=trim_post)   # by default
        sk_a, pk_a = a[0], a[idx]
        sk_b, pk_b = b[0], b[idx]
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
            "frac_toward": float((d < 0).mean()), "per_cell": d,
            "abs_disp": float(np.linalg.norm(pk_b[gate] - pk_a[gate], axis=1).mean())}


def readout_dissociation(nohead_dir, head_dir, reward,
                         behavioural=("replay_c1_pred", "cycle_1_goal"),
                         frozen=("replay_c1_pred", "replay_c1_postrl"),
                         skaggs_threshold: float = SKAGGS_THRESHOLD):
    """Behavioural vs frozen-replay shift for a matched pair, and their interaction."""
    out = {}
    for cond, d in (("nh", nohead_dir), ("hd", head_dir)):
        for tag, (pre, post) in (("beh", behavioural), ("frz", frozen)):
            # behavioural: pre-learning random walk -> post-learning goal-directed
            # probe without reset transients; frozen: same trajectory, two weight sets
            r = _mean_delta_to(d, pre, post, reward, skaggs_threshold,
                               trim_post=(tag == "beh"))
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
