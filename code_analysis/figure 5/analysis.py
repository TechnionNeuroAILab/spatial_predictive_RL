"""
Figure 5 analysis helpers — dissociating "policy-type" (occupancy-driven)
remapping from "real" (reward-prediction-error-driven) remapping.

Reuses:
  - Figure 2's analysis.py (load_phases, shared_place_cell_ids, paired_distance_stats)
    for the head-OFF vs head-ON frozen-replay contrast (panel b).
  - Figure 4's analysis.py (load_reloc_states, paired_tracking_stats)
    for the relocation seeds (panels c, d).
  - New_Code/analysis_helpers.load_period_centers for the entropy collapse gate.

No duplication of rate-map / Skaggs / field-center code.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]

# Both figure 2's and figure 4's analysis.py do a bare `from utils import ...`,
# which resolves against whatever `utils` module is currently cached in
# sys.modules. Since each figure has its own utils.py, we must pop the cache
# between loads so each analysis module picks up ITS OWN utils.py, not a
# previously-cached one from a different figure directory.
_saved_utils = sys.modules.pop("utils", None)

# ── Load Figure 2's analysis.py by absolute path ─────────────────────────────
_FIG2_DIR = _REPO / "code_analysis" / "figure 2"
_FIG1_DIR = _REPO / "code_analysis" / "figure 1"
for _p in (str(_FIG1_DIR), str(_FIG2_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.modules.pop("utils", None)
_spec2 = importlib.util.spec_from_file_location("fig2_analysis", _FIG2_DIR / "analysis.py")
_fig2 = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_fig2)

# ── Load Figure 4's analysis.py by absolute path ─────────────────────────────
_FIG4_DIR = _REPO / "code_analysis" / "figure 4"
if str(_FIG4_DIR) not in sys.path:
    sys.path.insert(0, str(_FIG4_DIR))
sys.modules.pop("utils", None)
_spec4 = importlib.util.spec_from_file_location("fig4_analysis", _FIG4_DIR / "analysis.py")
_fig4 = importlib.util.module_from_spec(_spec4)
_spec4.loader.exec_module(_fig4)

# Restore (or clear) the utils cache so THIS file's own `from utils import ...`
# below picks up figure 5's utils.py, not figure 2's/4's.
sys.modules.pop("utils", None)
if _saved_utils is not None:
    sys.modules["utils"] = _saved_utils

# ── New_Code utilities for the entropy collapse gate + frozen-replay diagnostic ──
_NEW_CODE = _REPO / "New_Code"
if str(_NEW_CODE) not in sys.path:
    sys.path.insert(0, str(_NEW_CODE))
from analysis_helpers import (  # noqa: E402
    load_period_centers, has_period,
    peak_entropy, seed_is_clean,
    frozen_replay_delta as _frozen_replay_delta_core,
)

# Force figure 5's own utils.py to win the lookup (sys.path was polluted with
# figure 2's / figure 4's directories above, both ahead of _HERE).
sys.path.insert(0, str(_HERE))
sys.modules.pop("utils", None)
from utils import MIN_ENTROPY, SKAGGS_THRESHOLD  # noqa: E402

REWARD_A = np.array([0.3, 0.3])
M_GRID = 32


# ---------------------------------------------------------------------------
# Entropy collapse gate + frozen-replay diagnostic
#
# Both peak_entropy()/seed_is_clean() and the frozen-replay delta logic now
# live in New_Code/analysis_helpers.py (shared with figure 2/analysis.py and
# select_best_seed_fig4.py) -- these are thin wrappers that pin figure 5's
# own defaults (MIN_ENTROPY/SKAGGS_THRESHOLD from this figure's utils.py).
# ---------------------------------------------------------------------------

def _frozen_replay_delta(data_dir: Path, skaggs_thr: float = SKAGGS_THRESHOLD) -> dict | None:
    """See analysis_helpers.frozen_replay_delta — this just fixes the periods/reward
    for figure 5's cycle-1 frozen-replay diagnostic (positive = toward reward)."""
    return _frozen_replay_delta_core(
        data_dir, "replay_c1_pred", "replay_c1_postrl", REWARD_A, skaggs_thr
    )


def load_double_dissociation(nohead_dir: Path, head_dir: Path, seeds=range(10),
                              min_entropy: float = MIN_ENTROPY,
                              skaggs_thr: float = SKAGGS_THRESHOLD):
    """
    Per-seed frozen-replay Δd for no-head and head-ON conditions.

    The frozen-replay contrast (H1 vs H2 diagnostic) compares:
        replay_c1_pred   — pre-RL weights on the fixed cycle-1 trajectory
        replay_c1_postrl — post-RL weights on the SAME fixed trajectory

    Because the trajectory is identical, any field-center shift can only come
    from weight changes — a genuine H2 (RPE-driven, real-remapping) signal.
    SR predicts zero shift here (occupancy is unchanged), so head > no-head
    in frozen replay is the hallmark of real remapping that SR cannot explain.

    Seeds are restricted to those passing the entropy collapse gate in BOTH
    conditions (paired comparison — same seed = same phase-1 initialization).

    min_entropy/skaggs_thr are exposed for sensitivity sweeps (see
    generate_figure5_supp.py panels a/b); both default to this figure's
    utils.py constants.
    """
    rows = []
    for seed in seeds:
        nh_dir = Path(nohead_dir) / f"nohead_seed{seed}" / "data"
        hd_dir = Path(head_dir) / f"head_seed{seed}" / "data"
        if not nh_dir.exists() or not hd_dir.exists():
            continue

        nh_clean = seed_is_clean(nh_dir, ["replay_c1_pred", "replay_c1_postrl"], min_entropy)
        hd_clean = seed_is_clean(hd_dir, ["replay_c1_pred", "replay_c1_postrl"], min_entropy)
        if not (nh_clean and hd_clean):
            continue

        stats_nh = _frozen_replay_delta(nh_dir, skaggs_thr)
        stats_hd = _frozen_replay_delta(hd_dir, skaggs_thr)

        if stats_nh is None or stats_hd is None:
            continue

        rows.append({
            "seed":        seed,
            "n_nohead":    stats_nh["n"],
            "n_head":      stats_hd["n"],
            # positive = toward reward (H2 signature)
            "delta_nohead": stats_nh["mean_delta"],
            "delta_head":   stats_hd["mean_delta"],
            "frac_nohead":  stats_nh["frac_toward"],
            "frac_head":    stats_hd["frac_toward"],
        })
    return rows


def double_dissociation_stats(rows: list[dict]) -> dict:
    d_nh = np.array([r["delta_nohead"] for r in rows])
    d_hd = np.array([r["delta_head"] for r in rows])
    try:
        _, p_paired = wilcoxon(d_nh, d_hd)
    except Exception:
        p_paired = float("nan")
    return {
        "n_seeds": len(rows),
        "mean_nohead": float(np.mean(d_nh)),
        "sd_nohead": float(np.std(d_nh, ddof=1)) if len(d_nh) > 1 else float("nan"),
        "mean_head": float(np.mean(d_hd)),
        "sd_head": float(np.std(d_hd, ddof=1)) if len(d_hd) > 1 else float("nan"),
        "wilcoxon_p_paired": float(p_paired),
    }


# ---------------------------------------------------------------------------
# Panels c, d — relocation: behavior/remapping decoupling + proactive interference
# ---------------------------------------------------------------------------

def load_relocation_summary(reloc_dir: Path, seeds: list[int]) -> list[dict]:
    """Per-seed behavioral metrics + tracking (frac/delta toward A and B)."""
    rows = []
    for seed in seeds:
        run_dir = Path(reloc_dir) / f"reloc_seed{seed}"
        met_path = run_dir / "metrics.json"
        if not met_path.exists():
            continue
        met = json.loads(met_path.read_text())
        tsr = met.get("test_success_rates", [float("nan"), float("nan")])

        states = _fig4.load_reloc_states(str(run_dir / "data"))
        ts = _fig4.paired_tracking_stats(states)

        rows.append({
            "seed": seed,
            "succ_c1": tsr[0] if len(tsr) > 0 else float("nan"),
            "succ_c2": tsr[1] if len(tsr) > 1 else float("nan"),
            "succ_last10": met.get("phase2_mean_success_last10", float("nan")),
            "frac_A": ts["shift_A"]["frac_toward"],
            "frac_B": ts["shift_B"]["frac_toward"],
            "delta_A": ts["shift_A"]["mean_delta"],
            "delta_B": ts["shift_B"]["mean_delta"],
        })
    return rows
