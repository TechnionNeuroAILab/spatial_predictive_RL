"""
Occupancy-matched estimator of weight-driven place-field remapping.

Comparing a free-foraging epoch with a goal-directed one confounds two things:
the weights changed and the sampling changed. Occupancy-normalized rate maps do
not separate them -- in this model the naive contrast misestimates the
weight-driven shift badly enough to reverse the head vs no-head ordering.
Frozen-weight replay does separate them exactly, but only in silico: you cannot
replay an animal's trajectory through its own earlier brain. This module is the
experimentally realizable substitute: it rebuilds both periods' rate maps from
timestep subsets whose sampling is identical by construction (position, or
position x heading), plus an inverse-propensity reweighting variant and the
naive estimator for reference.

Delta-d negative = fields moved toward the reward.

Other modules reuse its loading / rate-map / weighted-Skaggs primitives:
extinction_analysis, reward_relative_analysis and runs/verify_extinction_match.
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

from ratemaps import field_center_of_mass, field_peak  # noqa: E402

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
    """Load one period as (pos (n,2), theta (n,), act (n,512) float32)."""
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
    """Keep-mask dropping `n_trim` steps after every hidden-state reset."""
    n = len(pos)
    keep = np.ones(n, dtype=bool)
    # run_goal_conditioned_probe zeroes the hidden state at every goal arrival
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    resets = np.where(step > TELEPORT_THR)[0] + 1     # index of the post-jump step
    for r in resets:
        keep[r:r + n_trim] = False
    return keep


# ---------------------------------------------------------------------------
# Strata + matching
# ---------------------------------------------------------------------------

def position_bins(pos: np.ndarray) -> np.ndarray:
    """Flat 32x32 position-bin index per timestep, in canonical gc ordering."""
    xb = np.clip((pos[:, 0] * M).astype(int), 0, M - 1)
    yb = np.clip((pos[:, 1] * M).astype(int), 0, M - 1)
    return xb * M + yb


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


def match_indices(strata_pre: np.ndarray, strata_post: np.ndarray,
                  rng: np.random.Generator):
    """Per-stratum min-count subsampling, the core of the matching estimator.

    For every stratum present in either period, keep min(count_pre, count_post)
    timesteps from each, drawing without replacement when a period has more
    than the target. The two returned subsets therefore have identical stratum
    histograms, so any remaining difference between their rate maps cannot come
    from sampling.

    Parameters
    ----------
    strata_pre, strata_post : ndarray (n,), int
        Stratum id per timestep for the two periods, from `strata_ids`.
    rng : numpy.random.Generator
        Draws the subsample. Results are stochastic; see `estimate`, which
        averages over repetitions.

    Returns
    -------
    (idx_pre, idx_post) : tuple of ndarray
        Indices into each period. Strata absent from either period contribute
        nothing, so both may be much shorter than the inputs.
    """
    n_strata = int(max(strata_pre.max(), strata_post.max())) + 1
    count_pre = np.bincount(strata_pre, minlength=n_strata)
    count_post = np.bincount(strata_post, minlength=n_strata)
    # keep as many as the scarcer period has -- samples can be dropped, not invented
    target = np.minimum(count_pre, count_post)

    idx_pre, idx_post = [], []
    order_pre = np.argsort(strata_pre, kind="stable")
    order_post = np.argsort(strata_post, kind="stable")
    bounds_pre = np.searchsorted(strata_pre[order_pre], np.arange(n_strata + 1))
    bounds_post = np.searchsorted(strata_post[order_post], np.arange(n_strata + 1))
    for s in np.nonzero(target)[0]:
        t = target[s]
        pool_pre = order_pre[bounds_pre[s]:bounds_pre[s + 1]]
        pool_post = order_post[bounds_post[s]:bounds_post[s + 1]]
        idx_pre.append(pool_pre if len(pool_pre) == t else rng.choice(pool_pre, t, replace=False))
        idx_post.append(pool_post if len(pool_post) == t else rng.choice(pool_post, t, replace=False))
    return np.concatenate(idx_pre), np.concatenate(idx_post)


def propensity_weights(strata: np.ndarray, strata_other: np.ndarray) -> np.ndarray:
    """Inverse-propensity weights: the deterministic alternative to subsampling."""
    n_strata = int(max(strata.max(), strata_other.max())) + 1
    c_self = np.bincount(strata, minlength=n_strata).astype(float)
    c_other = np.bincount(strata_other, minlength=n_strata).astype(float)
    # same per-stratum target match_indices subsamples to, but reached by
    # downweighting instead of discarding: no run-to-run variance, all samples used
    target = np.minimum(c_self, c_other)
    with np.errstate(divide="ignore", invalid="ignore"):
        w_per = np.where(c_self > 0, target / c_self, 0.0)   # unvisited strata -> 0
    return w_per[strata]


# ---------------------------------------------------------------------------
# Rate maps, Skaggs (occupancy-weighted), centres, delta
# ---------------------------------------------------------------------------

def rate_maps_from(act: np.ndarray, gc: np.ndarray,
                   weights: np.ndarray | None = None):
    """Per-cell mean-activity maps (512, M, M) plus the occupancy probability map."""
    nb = M * M
    # weights let the "reweight" mode match occupancy without discarding samples
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
    """Occupancy-weighted Skaggs information (bits/spike), r_bar = sum_i p_i * r_i."""
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
    """Smoothed field peak and COM per cell, using the shared ratemaps conventions."""
    peaks = np.zeros((HIDDEN, 2))
    coms = np.zeros((HIDDEN, 2))
    for i in range(HIDDEN):
        sm = gaussian_filter(rm[i], sigma=SIGMA)
        mx = sm.max()
        smn = sm / mx if mx > 0 else sm
        # both apply the project's (y,x) swap -- distance-preserving for the
        # on-diagonal rewards used here (see ratemaps).
        peaks[i] = field_peak(smn)
        coms[i] = field_center_of_mass(smn)
    return peaks, coms


def delta_stats(c_pre, c_post, sk_pre, sk_post, reward=REWARD,
                thr: float = SKAGGS_THR):
    """Gated per-cell Delta-d; negative = fields moved toward the reward."""
    gate = (sk_pre >= thr) & (sk_post >= thr)
    n = int(gate.sum())
    if n < 10:
        return None                      # too few gated cells to average
    d = (np.linalg.norm(c_post[gate] - reward, axis=1)
         - np.linalg.norm(c_pre[gate] - reward, axis=1))
    # subsampling pins most peaks to the same bin, so exact ties are excluded
    # from frac_toward_moved rather than counted as "away"
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
    """Run one estimator to completion and summarize it across repetitions.

    Parameters
    ----------
    pos_pre, th_pre, act_pre : ndarray
        Position, heading and activity for the earlier period.
    pos_post, th_post, act_post : ndarray
        The same for the later period.
    mode : {"naive", "pos", "joint", "reweight"}
        "naive" contrasts the periods as-is; "pos" matches on position bins;
        "joint" matches on position x heading (the primary estimator);
        "reweight" uses inverse-propensity weights instead of subsampling.
    seed : int
        Base seed; each repetition draws from default_rng([seed, rep, mode]).
    n_reps : int
        Repetitions for the subsampling modes. "naive" and "reweight" are
        deterministic and always run once.
    frac : float
        Keep only this fraction of the matched samples, for the sample-size
        curve. 1.0 keeps all of them.
    reward : array-like (2,)
        Reward position passed through to `delta_stats`.

    Returns
    -------
    dict or None
        {"mean_dd", "rep_sd", "n_cells", "frac_toward_moved", "n_matched",
        "n_reps"}. None if no repetition produced enough gated cells.

    Notes
    -----
    The subsampling modes are stochastic, so mean_dd is a mean over
    repetitions and rep_sd is their spread. Downstream inference must use these
    per-seed rep-means, never individual repetitions.
    """
    if mode == "naive":
        rows = [_single_pass(pos_pre, act_pre, pos_post, act_post,
                             None, None, reward)]
        n_matched = len(pos_post)
    elif mode == "reweight":
        s_pre = strata_ids(pos_pre, th_pre)
        s_post = strata_ids(pos_post, th_post)
        w_pre = propensity_weights(s_pre, s_post)
        w_post = propensity_weights(s_post, s_pre)
        rows = [_single_pass(pos_pre, act_pre, pos_post, act_post,
                             w_pre, w_post, reward, weighted=True)]
        n_matched = int(round(w_post.sum()))
    else:
        theta_pre = th_pre if mode == "joint" else None
        theta_post = th_post if mode == "joint" else None
        s_pre = strata_ids(pos_pre, theta_pre)
        s_post = strata_ids(pos_post, theta_post)
        rows, n_matched = [], 0
        for rep in range(n_reps):
            rng = np.random.default_rng([seed, rep, hash(mode) % (2**31)])
            i_pre, i_post = match_indices(s_pre, s_post, rng)
            if frac < 1.0:
                k_pre = rng.choice(len(i_pre), max(50, int(len(i_pre) * frac)), replace=False)
                k_post = rng.choice(len(i_post), max(50, int(len(i_post) * frac)), replace=False)
                i_pre, i_post = i_pre[k_pre], i_post[k_post]
            n_matched = len(i_post)
            r = _single_pass(pos_pre[i_pre], act_pre[i_pre],
                             pos_post[i_post], act_post[i_post],
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


def _single_pass(pos_pre, act_pre, pos_post, act_post, w_pre, w_post,
                 reward, weighted=False):
    """One pass: rate maps -> weighted Skaggs -> peaks -> gated Delta-d."""
    gc_pre, gc_post = position_bins(pos_pre), position_bins(pos_post)
    rm_pre, occ_pre = rate_maps_from(act_pre, gc_pre, w_pre if weighted else None)
    rm_post, occ_post = rate_maps_from(act_post, gc_post, w_post if weighted else None)
    sk_pre = skaggs_weighted(rm_pre, occ_pre)
    sk_post = skaggs_weighted(rm_post, occ_post)
    pk_pre, _ = centres_from(rm_pre)
    pk_post, _ = centres_from(rm_post)
    return delta_stats(pk_pre, pk_post, sk_pre, sk_post, reward)


# ---------------------------------------------------------------------------
# Per-seed row (the driver calls this)
# ---------------------------------------------------------------------------

def estimator_row(run_dir: Path, seed: int, n_reps: int = 20,
                  post_period: str = "goal_conditioned",
                  size_fracs=(0.125, 0.25, 0.5, 1.0)) -> dict | None:
    """All estimators, plus ground truth and controls, for one network.

    Parameters
    ----------
    run_dir : Path
        Run directory; periods are read from `run_dir/data`.
    seed : int
        Seed label for this network, also the base rng seed. The control and
        size-curve calls offset it so they do not share draws.
    n_reps : int
        Repetitions per subsampling estimator.
    post_period : str
        The goal-directed period to contrast against `replay_c1_pred`.
    size_fracs : sequence of float
        Fractions of the matched sample for the sample-size curve.

    Returns
    -------
    dict or None
        One flat row: gt_*, est_<mode>*, null_<mode>*, and a nested
        "size_curve". None if any period fails to load or ground truth does
        not clear the cell gate.

    Notes
    -----
    Three things the caller is expected to check across seeds:
    agreement of each est_* with gt_mean_dd, the frozen-replay ground truth
    (replay_c1_pred -> replay_c1_postrl, same metric and gate); specificity,
    where null_* must come out near zero because cycle_1, cycle_1_goal and
    replay_c1_postrl share identical weights so the true weight-driven shift is
    exactly 0; and ordering, where paired head vs no-head must recover the
    ground-truth direction that the naive contrast reverses.
    """
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
