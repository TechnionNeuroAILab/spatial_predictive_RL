#!/usr/bin/env python3
"""
Occupancy-matched estimator of weight-driven place-field remapping.

THE PROBLEM. An experimenter comparing rate maps between a free-foraging epoch
and a goal-directed epoch confounds two things: the weights changed (learning)
and the sampling changed (behaviour). Occupancy-NORMALIZED rate maps do not fix
this — in this model the naive normalized contrast misestimates the weight-driven
shift badly enough to reverse the head vs no-head ordering. The frozen-weight
replay control fixes it in silico but has no experimental counterpart.

THE ESTIMATOR. Rebuild both periods' rate maps from timestep subsets whose
sampling distributions are IDENTICAL by construction:
  - position-matched: per 32x32 bin, keep min(count_pre, count_post) samples;
  - joint position x heading matched: same, on 32x32x8 strata (heading is
    exactly 8-valued in this model -- matched by exact value, never binned).
    Heading matters because the network's input depends on it; the biological
    analogue is head-direction sampling bias near a goal.
Plus an inverse-propensity REWEIGHTING variant (deterministic cross-check) and
the naive (unmatched) estimator for reference.

VALIDATION TARGETS (computed by the driver, cached to Repro/aggregate/):
  - per-seed agreement with frozen-replay ground truth (replay_c1_pred ->
    replay_c1_postrl), 40 no-head + 40 head seeds;
  - specificity: the same-weights contrast cycle_1 -> cycle_1_goal must return
    ~0 (cycle_1, cycle_1_goal and replay_c1_postrl share identical weights);
  - ordering: paired head-vs-no-head must recover the ground-truth ordering
    (head more toward-reward) that the naive contrast reverses.

CONVENTIONS. Delta-d NEGATIVE = toward reward (matches the manuscript).
Skaggs is the occupancy-WEIGHTED form (the canonical compute_skaggs in
plot_goal_directed_maps uses an unweighted mean over bins -- do not use it
here). Rate maps / centre extraction reuse the existing pipeline unchanged,
including its documented (y,x) transposition, which is distance-preserving for
the on-diagonal rewards used throughout this project.

PITFALLS HANDLED.
  - run_goal_conditioned_probe zeroes the RNN hidden state at every goal
    arrival with no warm-up (~2.5k resets per 50k steps): teleports are
    detected via |dpos| > 0.05 (one diagonal grid step = 0.0442) and
    N_TRIM_AFTER_RESET steps after each reset are dropped BEFORE matching.
  - Subsampling is stochastic: K repetitions with rng default_rng([seed, rep]);
    inference must use per-seed rep-means only.
  - Subsampling pins most peaks exactly: frac_toward is reported tie-aware
    (ties excluded from the denominator).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

_REPO = Path(__file__).resolve().parents[2]
_NEW_CODE = _REPO / "paper_code" / "lib"
if str(_NEW_CODE) not in sys.path:
    sys.path.insert(0, str(_NEW_CODE))

from plot_goal_directed_maps import field_center_of_mass  # noqa: E402
from plot_replay_v11 import field_peak  # noqa: E402

M = 32
HIDDEN = 512
SIGMA = 1.5
SKAGGS_THR = 0.3
REWARD = np.array([0.3, 0.3])
TELEPORT_THR = 0.05
N_TRIM_AFTER_RESET = 5   # steps dropped after each hidden-state reset
N_HEADINGS = 8


# ---------------------------------------------------------------------------
# Loading (fast paths; the long activation frame costs 5x more RAM than needed)
# ---------------------------------------------------------------------------

def load_period_arrays(data_dir: Path, period: str):
    """(pos (n,2) float64, theta (n,) float64, act (n,512) float32)."""
    data_dir = Path(data_dir)
    beh = pd.read_pickle(data_dir / f"df_behavior_{period}.pkl")
    act = pd.read_pickle(data_dir / f"df_activations_{period}.pkl")
    n = len(beh)
    a = act["activity"].to_numpy(dtype=np.float32)
    assert len(a) == n * HIDDEN, f"{period}: activation rows != n*512"
    a = a.reshape(n, HIDDEN)
    pos = beh[["x", "y"]].to_numpy(dtype=np.float64)
    theta = beh["theta"].to_numpy(dtype=np.float64)
    return pos, theta, a


def trim_teleports(pos: np.ndarray, n_trim: int = N_TRIM_AFTER_RESET) -> np.ndarray:
    """Boolean keep-mask dropping n_trim steps after every teleport (hidden-state
    reset). Step t is a teleport if |pos_t - pos_{t-1}| > TELEPORT_THR."""
    n = len(pos)
    keep = np.ones(n, dtype=bool)
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    resets = np.where(step > TELEPORT_THR)[0] + 1     # index of the post-jump step
    for r in resets:
        keep[r:r + n_trim] = False
    return keep


# ---------------------------------------------------------------------------
# Strata + matching
# ---------------------------------------------------------------------------

def position_bins(pos: np.ndarray) -> np.ndarray:
    xb = np.clip((pos[:, 0] * M).astype(int), 0, M - 1)
    yb = np.clip((pos[:, 1] * M).astype(int), 0, M - 1)
    return xb * M + yb                                  # canonical gc index


def heading_codes(theta: np.ndarray) -> np.ndarray:
    """Map the 8 exact theta values to 0..7 (exact-value join, never binned)."""
    vals = np.unique(theta)
    assert len(vals) <= N_HEADINGS, f"{len(vals)} distinct headings (> {N_HEADINGS})"
    code = np.searchsorted(vals, theta)
    return code


def strata_ids(pos: np.ndarray, theta: np.ndarray | None) -> np.ndarray:
    """Stratum index per timestep: position bin, or position x heading."""
    gc = position_bins(pos)
    if theta is None:
        return gc
    return gc * N_HEADINGS + heading_codes(theta)


def match_indices(strata_a: np.ndarray, strata_b: np.ndarray,
                  rng: np.random.Generator):
    """Per-stratum min-count subsampling. Returns (idx_a, idx_b) such that the
    stratum histograms of a[idx_a] and b[idx_b] are identical."""
    n_strata = int(max(strata_a.max(), strata_b.max())) + 1
    ca = np.bincount(strata_a, minlength=n_strata)
    cb = np.bincount(strata_b, minlength=n_strata)
    target = np.minimum(ca, cb)

    idx_a, idx_b = [], []
    order_a = np.argsort(strata_a, kind="stable")
    order_b = np.argsort(strata_b, kind="stable")
    bounds_a = np.searchsorted(strata_a[order_a], np.arange(n_strata + 1))
    bounds_b = np.searchsorted(strata_b[order_b], np.arange(n_strata + 1))
    for s in np.nonzero(target)[0]:
        t = target[s]
        pool_a = order_a[bounds_a[s]:bounds_a[s + 1]]
        pool_b = order_b[bounds_b[s]:bounds_b[s + 1]]
        idx_a.append(pool_a if len(pool_a) == t else rng.choice(pool_a, t, replace=False))
        idx_b.append(pool_b if len(pool_b) == t else rng.choice(pool_b, t, replace=False))
    return np.concatenate(idx_a), np.concatenate(idx_b)


def propensity_weights(strata: np.ndarray, strata_other: np.ndarray) -> np.ndarray:
    """Inverse-propensity weights taking this period's stratum distribution to
    the per-stratum min of the two periods (deterministic reweighting variant)."""
    n_strata = int(max(strata.max(), strata_other.max())) + 1
    c_self = np.bincount(strata, minlength=n_strata).astype(float)
    c_other = np.bincount(strata_other, minlength=n_strata).astype(float)
    target = np.minimum(c_self, c_other)
    with np.errstate(divide="ignore", invalid="ignore"):
        w_per = np.where(c_self > 0, target / c_self, 0.0)
    return w_per[strata]


# ---------------------------------------------------------------------------
# Rate maps, Skaggs (occupancy-weighted), centres, delta
# ---------------------------------------------------------------------------

def rate_maps_from(act: np.ndarray, gc: np.ndarray,
                   weights: np.ndarray | None = None):
    """(512, M, M) mean-activity maps + occupancy probability map."""
    nb = M * M
    if weights is None:
        weights = np.ones(len(gc), dtype=np.float64)
    occ = np.zeros(nb)
    np.add.at(occ, gc, weights)
    sums = np.zeros((nb, HIDDEN), dtype=np.float64)
    np.add.at(sums, gc, act * weights[:, None])
    with np.errstate(divide="ignore", invalid="ignore"):
        rates = np.where(occ[:, None] > 0, sums / occ[:, None], 0.0)
    rm = rates.T.reshape(HIDDEN, M, M)
    occ_p = (occ / occ.sum()).reshape(M, M)
    return rm.astype(np.float32), occ_p


def skaggs_weighted(rm: np.ndarray, occ_p: np.ndarray) -> np.ndarray:
    """Occupancy-WEIGHTED Skaggs information (bits/spike)."""
    p = occ_p.reshape(-1)
    r = rm.reshape(HIDDEN, -1)
    mean_rate = r @ p
    out = np.zeros(HIDDEN)
    valid = mean_rate > 1e-12
    ratio = np.where(mean_rate[:, None] > 1e-12, r / mean_rate[:, None], 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(ratio > 0, p[None, :] * ratio * np.log2(ratio), 0.0)
    out[valid] = term[valid].sum(axis=1)
    return out


def centres_from(rm: np.ndarray):
    """Smoothed peak + COM per cell, existing pipeline conventions."""
    peaks = np.zeros((HIDDEN, 2))
    coms = np.zeros((HIDDEN, 2))
    for i in range(HIDDEN):
        sm = gaussian_filter(rm[i], sigma=SIGMA)
        mx = sm.max()
        smn = sm / mx if mx > 0 else sm
        peaks[i] = field_peak(smn)
        coms[i] = field_center_of_mass(smn)
    return peaks, coms


def delta_stats(c_pre, c_post, sk_pre, sk_post, reward=REWARD,
                thr: float = SKAGGS_THR):
    """Gated per-cell Delta-d. NEGATIVE = toward reward. Tie-aware frac."""
    gate = (sk_pre >= thr) & (sk_post >= thr)
    n = int(gate.sum())
    if n < 10:
        return None
    d = (np.linalg.norm(c_post[gate] - reward, axis=1)
         - np.linalg.norm(c_pre[gate] - reward, axis=1))
    moved = d != 0
    return {
        "n": n,
        "mean_dd": float(d.mean()),
        "frac_toward_moved": float((d[moved] < 0).mean()) if moved.any() else float("nan"),
        "n_moved": int(moved.sum()),
    }


# ---------------------------------------------------------------------------
# One full estimate
# ---------------------------------------------------------------------------

def estimate(pos_pre, th_pre, act_pre, pos_post, th_post, act_post,
             mode: str, seed: int, n_reps: int = 20, frac: float = 1.0,
             reward=REWARD):
    """mode in {'naive', 'pos', 'joint', 'reweight'}. Returns dict with
    mean_dd (rep-mean), rep_sd, n_cells (rep-mean), n_matched, coverage."""
    if mode == "naive":
        rows = [_single_pass(pos_pre, act_pre, pos_post, act_post,
                             None, None, reward)]
        n_matched = len(pos_post)
    elif mode == "reweight":
        sa = strata_ids(pos_pre, th_pre)
        sb = strata_ids(pos_post, th_post)
        wa = propensity_weights(sa, sb)
        wb = propensity_weights(sb, sa)
        rows = [_single_pass(pos_pre, act_pre, pos_post, act_post,
                             wa, wb, reward, weighted=True)]
        n_matched = int(round(wb.sum()))
    else:
        theta_a = th_pre if mode == "joint" else None
        theta_b = th_post if mode == "joint" else None
        sa = strata_ids(pos_pre, theta_a)
        sb = strata_ids(pos_post, theta_b)
        rows, n_matched = [], 0
        for rep in range(n_reps):
            rng = np.random.default_rng([seed, rep, hash(mode) % (2**31)])
            ia, ib = match_indices(sa, sb, rng)
            if frac < 1.0:
                ka = rng.choice(len(ia), max(50, int(len(ia) * frac)), replace=False)
                kb = rng.choice(len(ib), max(50, int(len(ib) * frac)), replace=False)
                ia, ib = ia[ka], ib[kb]
            n_matched = len(ib)
            r = _single_pass(pos_pre[ia], act_pre[ia], pos_post[ib], act_post[ib],
                             None, None, reward)
            if r:
                rows.append(r)
    rows = [r for r in rows if r]
    if not rows:
        return None
    dd = np.array([r["mean_dd"] for r in rows])
    return {
        "mean_dd": float(dd.mean()),
        "rep_sd": float(dd.std(ddof=1)) if len(dd) > 1 else 0.0,
        "n_cells": float(np.mean([r["n"] for r in rows])),
        "frac_toward_moved": float(np.nanmean([r["frac_toward_moved"] for r in rows])),
        "n_matched": int(n_matched),
        "n_reps": len(rows),
    }


def _single_pass(pos_a, act_a, pos_b, act_b, w_a, w_b, reward, weighted=False):
    gc_a, gc_b = position_bins(pos_a), position_bins(pos_b)
    rm_a, occ_a = rate_maps_from(act_a, gc_a, w_a if weighted else None)
    rm_b, occ_b = rate_maps_from(act_b, gc_b, w_b if weighted else None)
    sk_a = skaggs_weighted(rm_a, occ_a)
    sk_b = skaggs_weighted(rm_b, occ_b)
    pk_a, _ = centres_from(rm_a)
    pk_b, _ = centres_from(rm_b)
    return delta_stats(pk_a, pk_b, sk_a, sk_b, reward)


# ---------------------------------------------------------------------------
# Per-seed row (the driver calls this)
# ---------------------------------------------------------------------------

def estimator_row(run_dir: Path, seed: int, n_reps: int = 20,
                  post_period: str = "goal_conditioned",
                  size_fracs=(0.125, 0.25, 0.5, 1.0)) -> dict | None:
    """All estimators + ground truth + specificity control for one network."""
    d = Path(run_dir) / "data"
    try:
        pos_pre, th_pre, act_pre = load_period_arrays(d, "replay_c1_pred")
        pos_post, th_post, act_post = load_period_arrays(d, post_period)
        pos_gt, th_gt, act_gt = load_period_arrays(d, "replay_c1_postrl")
        pos_rw, th_rw, act_rw = load_period_arrays(d, "cycle_1")
        pos_cg, th_cg, act_cg = load_period_arrays(d, "cycle_1_goal")
    except Exception as e:
        print(f"  seed {seed}: load failed ({e})")
        return None

    # trim hidden-state transients in every goal-directed period
    kp = trim_teleports(pos_post)
    pos_post, th_post, act_post = pos_post[kp], th_post[kp], act_post[kp]
    kc = trim_teleports(pos_cg)
    pos_cg, th_cg, act_cg = pos_cg[kc], th_cg[kc], act_cg[kc]

    row: dict = {"seed": seed, "post_period": post_period,
                 "post_kept_after_trim": int(kp.sum()), "post_total": int(len(kp))}

    # ground truth: frozen replay pair, same metric/gate as the estimators
    gt = _single_pass(pos_pre, act_pre, pos_gt, act_gt, None, None, REWARD)
    if gt is None:
        return None
    row["gt_mean_dd"] = gt["mean_dd"]
    row["gt_n_cells"] = gt["n"]

    for mode in ("naive", "pos", "joint", "reweight"):
        r = estimate(pos_pre, th_pre, act_pre, pos_post, th_post, act_post,
                     mode, seed, n_reps=n_reps)
        row[f"est_{mode}"] = r["mean_dd"] if r else None
        if r:
            row[f"est_{mode}_sd"] = r["rep_sd"]
            row[f"est_{mode}_ncells"] = r["n_cells"]
            row[f"est_{mode}_nmatched"] = r["n_matched"]

    # specificity: same-weights contrast (cycle_1 vs cycle_1_goal share weights
    # with replay_c1_postrl) -- true weight-driven shift is exactly zero.
    for mode in ("naive", "joint"):
        r = estimate(pos_rw, th_rw, act_rw, pos_cg, th_cg, act_cg,
                     mode, seed + 10_000, n_reps=n_reps)
        row[f"null_{mode}"] = r["mean_dd"] if r else None
        if r:
            row[f"null_{mode}_nmatched"] = r["n_matched"]

    # sample-size curve on the primary (joint) estimator
    row["size_curve"] = {}
    for f in size_fracs:
        r = estimate(pos_pre, th_pre, act_pre, pos_post, th_post, act_post,
                     "joint", seed + 20_000, n_reps=max(5, n_reps // 2), frac=f)
        if r:
            row["size_curve"][str(f)] = {"mean_dd": r["mean_dd"], "sd": r["rep_sd"],
                                         "n_matched": r["n_matched"]}
    return row
