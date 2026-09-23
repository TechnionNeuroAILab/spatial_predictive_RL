"""Figure 4 shared constants and Nature styling."""
from __future__ import annotations

from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Publication style: single source of truth in paper_code/lib/pubstyle.py.
# Names are re-exported so existing panel code keeps importing them from here.
# ---------------------------------------------------------------------------
import sys as _sys
_LIB = Path(__file__).resolve().parents[2] / "lib"
if str(_LIB) not in _sys.path:
    _sys.path.insert(0, str(_LIB))

from pubstyle import (  # noqa: E402,F401
    MM_TO_PT, PAGE_W, COL_SINGLE, COL_WIDE, COL_DOUBLE,
    FS_BASE, FS_TICK, FS_SMALL, FS_PANEL, LABEL_SIZE,
    C, OI, PHASE_COLORS,
    COLOR_NOHEAD, COLOR_HEAD, COLOR_A, COLOR_B, COLOR_RIGID,
    COLOR_MODEL, COLOR_DATA, COLOR_PRED, COLOR_POSTLEARN, COLOR_POSTEXT,
    COLOR_DISTANCE, COLOR_GOALDIR, COLOR_HD, COLOR_REWARD, COLOR_MUTED,
    RATEMAP, RATEMAP_CMAP, DIVERGING, SEQUENTIAL,
    apply_nature_style, ensure_dir, save_figure,
    panel_label, title, fmt_p, stat_annot, despine, arena_axes,
    reward_marker, thin_colorbar, zero_line, style_violin, hist_outline,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
PANELS_DIR = FIGURES_DIR / "panels"
NEW_CODE = REPO_ROOT / "paper_code" / "lib"

PAGE_MARGIN = 5.0 * MM_TO_PT

M = 32
HIDDEN_DIM = 512

# Reward positions
REWARD_A = np.array([0.3, 0.3])
REWARD_B = np.array([0.7, 0.7])

# Period names in the data directory
PERIOD_PRE = "replay_reloc_pred"
PERIOD_POST_A = "replay_reloc_postA"
PERIOD_POST_B = "replay_reloc_postB"

PERIODS = [
    (PERIOD_PRE, "baseline\n(pre-RL)"),
    (PERIOD_POST_A, "after A"),
    (PERIOD_POST_B, "after B\n(relocated)"),
]

# Phase colours for tracking curves
PERIOD_COLORS = {
    PERIOD_PRE: "#999999",
    PERIOD_POST_A: COLOR_A,
    PERIOD_POST_B: COLOR_B,
}

# Default run: the displayed network is the median entropy-gate-passing seed
# by tracking quality (rank 17 of 32 clean seeds; see
# Repro/multi_seed_reloc/aggregate/fig4_multi_seed_summary.json and
# paper_code/runs/select_best_seed_fig4.py for the ranking). Entropy gate for this
# seed: baseline 7.55 / after A 7.60 / after B 7.47 bits (all >= 5.5).
DEFAULT_RUN = REPO_ROOT / "Repro" / "multi_seed_reloc" / "reloc_seed9"
SKAGGS_THRESHOLD = 0.15

# Entropy collapse gate (matches Figure 5's diagnostic; see paper_code/lib/analysis_helpers.py)
MIN_ENTROPY = 5.5

