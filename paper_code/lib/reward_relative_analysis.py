"""
Reward-relative vs place-anchored subpopulations across reward relocation.

A 2D translation of Sosa, Plitt & Giocomo (2025, Nat Neurosci): after the reward
moves A->B, does a cell hold its firing at the same position relative to the
reward, or at the same allocentric position? Each cell gets one cross-phase
correlation per frame r_reward in reward-centred coordinates, r_place in
allocentric ones and is classified by which wins against a cell-identity
shuffle null. The two are geometrically coupled (a perfectly stable allocentric
field still scores a nonzero r_reward), so only their contrast is interpretable,
never either one alone.

Both read-outs are computed: behavioural (cycle_1_goal vs cycle_2_goal) and
occupancy-controlled (the two frozen replays, identical trajectory). The latter
is primary reward-frame sampling is then identical by construction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

_REPO = Path(__file__).resolve().parents[2]
_NEW_CODE = _REPO / "paper_code" / "lib"
if str(_NEW_CODE) not in sys.path:
    sys.path.insert(0, str(_NEW_CODE))

from occupancy_matched_estimator import (  # noqa: E402
    HIDDEN, M, SIGMA, load_period_arrays, position_bins, rate_maps_from,
    skaggs_weighted, trim_teleports,
)

REWARD_A = np.array([0.3, 0.3])
REWARD_B = np.array([0.7, 0.7])
SKAGGS_THR = 0.15
FRAME_HALF = 0.3          # overlap window half-width in reward-centred coords
MIN_VALID_BINS = 30       # per-cell minimum comparable bins per read-out
N_SHUFFLE = 200

# period -> active reward, for the relocation runs handled here
ACTIVE_REWARD = {
    "cycle_1_goal": REWARD_A,
    "replay_reloc_postA": REWARD_A,
    "cycle_2_goal": REWARD_B,
    "replay_reloc_postB": REWARD_B,
}


def reward_frame_maps(pos: np.ndarray, act: np.ndarray, reward: np.ndarray):
    """(512, F, F) mean-activity maps in reward-centred coordinates, F bins of
    width 1/M spanning u in [-FRAME_HALF, FRAME_HALF]^2, plus visit counts."""
    u = pos - reward
    nb = int(round(2 * FRAME_HALF * M))          # 19 with defaults
    ub = np.floor((u + FRAME_HALF) * M).astype(int)
    ok = (ub[:, 0] >= 0) & (ub[:, 0] < nb) & (ub[:, 1] >= 0) & (ub[:, 1] < nb)
    idx = ub[ok, 0] * nb + ub[ok, 1]
    a = act[ok]
    counts = np.bincount(idx, minlength=nb * nb).astype(float)
    sums = np.zeros((nb * nb, HIDDEN))
    np.add.at(sums, idx, a)
    with np.errstate(divide="ignore", invalid="ignore"):
        rates = np.where(counts[:, None] > 0, sums / counts[:, None], np.nan)
    return rates.T.reshape(HIDDEN, nb, nb).astype(np.float32), counts.reshape(nb, nb)


def _smoothed(m2d: np.ndarray) -> np.ndarray:
    filled = np.nan_to_num(m2d, nan=0.0)
    return gaussian_filter(filled, sigma=SIGMA)


def _percell_corr(maps_a: np.ndarray, maps_b: np.ndarray,
                  valid: np.ndarray) -> np.ndarray:
    """Pearson r per cell over `valid` bins of smoothed maps; NaN if degenerate."""
    out = np.full(HIDDEN, np.nan)
    if valid.sum() < MIN_VALID_BINS:
        return out
    for i in range(HIDDEN):
        a = _smoothed(maps_a[i])[valid]
        b = _smoothed(maps_b[i])[valid]
        if a.std() > 1e-9 and b.std() > 1e-9:
            out[i] = float(np.corrcoef(a, b)[0, 1])
    return out


def _shuffle_null_95(maps_a, maps_b, valid, gate_idx, rng) -> float:
    """95th percentile of cell-identity-shuffled correlations (cell i in A vs a
    random different gated cell j in B). Population-level null threshold."""
    vals = []
    gi = np.asarray(gate_idx)
    for _ in range(N_SHUFFLE):
        i = rng.choice(gi)
        j = rng.choice(gi)
        if i == j:
            continue
        a = _smoothed(maps_a[i])[valid]
        b = _smoothed(maps_b[j])[valid]
        if a.std() > 1e-9 and b.std() > 1e-9:
            vals.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.percentile(vals, 95)) if len(vals) > 20 else float("nan")


def readout(run_dir: Path, period_a: str, period_b: str, seed: int) -> dict | None:
    """One read-out (behavioural or occupancy-controlled) for one network."""
    d = Path(run_dir) / "data"
    try:
        pos_a, _, act_a = load_period_arrays(d, period_a)
        pos_b, _, act_b = load_period_arrays(d, period_b)
    except Exception as e:
        print(f"  {run_dir}: load failed ({e})")
        return None
    # trim hidden-state transients in behavioural goal-directed periods
    if period_a.endswith("_goal"):
        k = trim_teleports(pos_a); pos_a, act_a = pos_a[k], act_a[k]
    if period_b.endswith("_goal"):
        k = trim_teleports(pos_b); pos_b, act_b = pos_b[k], act_b[k]

    rew_a, rew_b = ACTIVE_REWARD[period_a], ACTIVE_REWARD[period_b]

    # allocentric maps + gate
    rm_a, occ_a = rate_maps_from(act_a, position_bins(pos_a))
    rm_b, occ_b = rate_maps_from(act_b, position_bins(pos_b))
    sk_a, sk_b = skaggs_weighted(rm_a, occ_a), skaggs_weighted(rm_b, occ_b)
    gate = (sk_a >= SKAGGS_THR) & (sk_b >= SKAGGS_THR)
    gate_idx = np.where(gate)[0]
    if len(gate_idx) < 30:
        return None

    valid_place = (occ_a > 0) & (occ_b > 0)
    r_place = _percell_corr(rm_a, rm_b, valid_place)

    fm_a, cnt_a = reward_frame_maps(pos_a, act_a, rew_a)
    fm_b, cnt_b = reward_frame_maps(pos_b, act_b, rew_b)
    valid_frame = (cnt_a > 0) & (cnt_b > 0)
    r_reward = _percell_corr(fm_a, fm_b, valid_frame)

    rng = np.random.default_rng(seed)
    null95_place = _shuffle_null_95(rm_a, rm_b, valid_place, gate_idx, rng)
    null95_reward = _shuffle_null_95(fm_a, fm_b, valid_frame, gate_idx, rng)

    ok = gate & np.isfinite(r_reward) & np.isfinite(r_place)
    idx = np.where(ok)[0]
    rr = r_reward[idx]
    rp = r_place[idx]
    is_rr = (rr > rp) & (rr > null95_reward)
    is_pa = (rp > rr) & (rp > null95_place)
    return {
        "n_gated": int(gate.sum()),
        "n_classified": int(len(idx)),
        "attrition": float(1 - len(idx) / max(1, gate.sum())),
        "frac_reward_relative": float(is_rr.mean()),
        "frac_place_anchored": float(is_pa.mean()),
        "frac_other": float((~is_rr & ~is_pa).mean()),
        "rri_mean": float((rr - rp).mean()),
        "rri_hist": np.histogram(rr - rp, bins=np.linspace(-1, 1, 21))[0].tolist(),
        "null95_reward": null95_reward,
        "null95_place": null95_place,
        "valid_frame_bins": int(valid_frame.sum()),
        "cells": {  # per-cell detail for example-cell selection
            "idx": idx.tolist(),
            "r_reward": np.round(rr, 4).tolist(),
            "r_place": np.round(rp, 4).tolist(),
        },
    }


def per_seed_row(run_dir: Path, seed: int) -> dict | None:
    """Both read-outs for one network; None if the controlled one is unavailable."""
    row = {"seed": seed}
    beh = readout(run_dir, "cycle_1_goal", "cycle_2_goal", seed)
    ctl = readout(run_dir, "replay_reloc_postA", "replay_reloc_postB", seed + 5000)
    if ctl is None:
        return None
    row["behavioural"] = beh
    row["controlled"] = ctl
    return row
