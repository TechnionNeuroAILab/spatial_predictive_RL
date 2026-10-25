"""
sr_direction_analysis.py — SR-predicted shift direction test for the A→B
(occupancy-driven, RL-policy) remapping contrast.

The retired Figure 5 panel f tested the frozen-replay shift magnitude against a
trivial SR = 0 null (occupancy is fixed there, so
SR predicts no shift by construction). This module instead tests whether SR
explains shift direction for the A→B contrast, where SR's prediction is
genuinely non-trivial (occupancy really does change between the
prediction-only exploration phase and the RL-trained goal-directed phase).

For each seed:
  1. Build the empirical policy transition matrix from cycle_1_goal behavior
     (the RL-policy trajectory), and its SR matrix.
  2. sr_predicted_shift_vectors() gives, for every grid state, the direction
     SR predicts a cell starting there should move.
  3. For each shared place cell (Skaggs-gated in phases A and B), look up the
     SR-predicted direction at its Phase-A location and compare it to
     the cell's actual shift direction (com_B - com_A) via the angle between
     them (V-test toward 0: V = mean cos(angle), 0 = no relationship; stored
     under the old key names "R" / "p_rayleigh").
  4. The residual — the actual shift component orthogonal to the SR
     prediction — is correlated against distance-to-reward: if the residual
     grows specifically as cells approach the reward, that's evidence for an
     SR-independent, reward-proximity-driven signal (H2) on top of the
     occupancy-explained (H1/SR) component.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]

_NEW_CODE = _REPO / "code" / "lib"
if str(_NEW_CODE) not in sys.path:
    sys.path.insert(0, str(_NEW_CODE))
from sr_common import get_shared_random as _get_shared_random, M, WB, GAMMA  # noqa: E402
from successor_representation import (  # noqa: E402
    transition_from_behavior_positions,
    compute_sr_matrix,
    sr_predicted_shift_vectors,
    nearest_state_for_points,
    angle_between_vectors,
    rayleigh_from_angles,
)

# Figure 5's own analysis.py (imported earlier by generate_figure5_supp.py)
# leaves figure 5's own directory at the front of sys.path (it pushes _HERE
# there at its own tail, after it's done needing figure 2's utils.py). By the
# time this module runs, figure 1/figure 2's directories may already be
# present in sys.path but stuck behind figure 5's -- an `if _p not in
# sys.path` guard would then wrongly skip re-inserting them at the front.
# So we unconditionally insert (duplicates in sys.path are harmless) to
# guarantee figure 2's directory -- and hence its utils.py -- wins the
# `from utils import ...` lookup inside figure 2's analysis.py exec below.
_saved_utils = sys.modules.pop("utils", None)
_FIG1_DIR = _REPO / "code" / "figures" / "fig1"
_FIG2_DIR = _REPO / "code" / "figures" / "fig2"
sys.path.insert(0, str(_FIG1_DIR))
sys.path.insert(0, str(_FIG2_DIR))
sys.modules.pop("utils", None)
import importlib.util as _ilu
_spec2 = _ilu.spec_from_file_location("fig2_analysis_srdir", _FIG2_DIR / "analysis.py")
_fig2 = _ilu.module_from_spec(_spec2)
sys.modules.pop("utils", None)
_spec2.loader.exec_module(_fig2)
sys.modules.pop("utils", None)
if _saved_utils is not None:
    sys.modules["utils"] = _saved_utils
sys.path.insert(0, str(_HERE))

load_phases = _fig2.load_phases
shared_place_cell_ids = _fig2.shared_place_cell_ids

REWARD = np.array([0.3, 0.3])
SKAGGS_THR = 0.05


def seed_direction_metrics(data_dir: Path, skaggs_threshold: float = SKAGGS_THR) -> dict | None:
    """SR-predicted vs actual shift direction for the A→B (policy) contrast
    on one seed. Returns None if behavioral data is missing or too few cells
    pass the Skaggs gate."""
    _, SR_random, state_xy = _get_shared_random()

    beh_path = Path(data_dir) / "df_behavior_cycle_1_goal.parquet"
    if not beh_path.exists():
        return None
    try:
        pos = pd.read_parquet(beh_path)[["x", "y"]].to_numpy(dtype=float)
    except Exception:
        return None

    T_policy, _ = transition_from_behavior_positions(pos, M, WB)
    SR_policy, _, _ = compute_sr_matrix(T_policy, GAMMA, "auto", 500, 1e10)
    pred_vec = sr_predicted_shift_vectors(SR_random, SR_policy, state_xy)   # (S, 2)

    try:
        phases = load_phases(data_dir)
    except Exception:
        return None
    ids = shared_place_cell_ids(phases, skaggs_threshold=skaggs_threshold)
    if len(ids) < 10:
        return None

    com_a = phases["A"]["com"][ids]
    com_b = phases["B"]["com"][ids]
    actual_shift = com_b - com_a                     # (N, 2)

    state_idx = nearest_state_for_points(com_a, state_xy)
    pred_shift = pred_vec[state_idx]                  # (N, 2)

    angles = angle_between_vectors(actual_shift, pred_shift)   # radians, NaN where either vector is ~0
    R, p_rayleigh = rayleigh_from_angles(angles)

    # Residual: component of the actual shift orthogonal to the SR-predicted
    # direction (projects out whatever SR already explains).
    pred_norm = np.linalg.norm(pred_shift, axis=1, keepdims=True)
    pred_hat = np.divide(pred_shift, pred_norm, out=np.zeros_like(pred_shift),
                          where=pred_norm > 1e-12)
    proj_len = np.sum(actual_shift * pred_hat, axis=1, keepdims=True)
    residual_vec = actual_shift - proj_len * pred_hat
    residual_mag = np.linalg.norm(residual_vec, axis=1)

    dist_to_reward = np.linalg.norm(com_a - REWARD, axis=1)

    return {
        "n": int(len(ids)),
        "angles": angles,
        "R": R,
        "p_rayleigh": p_rayleigh,
        "residual_mag": residual_mag,
        "dist_to_reward": dist_to_reward,
    }


def load_all_seeds_direction(cond_dir: Path, prefix: str, seeds: list[int],
                              skaggs_threshold: float = SKAGGS_THR) -> list[dict]:
    """Per-seed direction-test metrics for a condition (nohead or head)."""
    rows = []
    for seed in seeds:
        data_dir = Path(cond_dir) / f"{prefix}seed{seed}" / "data"
        m = seed_direction_metrics(data_dir, skaggs_threshold)
        if m is None:
            continue
        # per-seed summary stats for aggregation
        r_s, p_s = spearmanr(m["residual_mag"], m["dist_to_reward"])
        rows.append({
            "seed": seed,
            "n": m["n"],
            "R": m["R"],
            "p_rayleigh": m["p_rayleigh"],
            "residual_vs_dist_r": r_s,
            "residual_vs_dist_p": p_s,
            "angles": m["angles"],
            "residual_mag": m["residual_mag"],
            "dist_to_reward": m["dist_to_reward"],
        })
    return rows
