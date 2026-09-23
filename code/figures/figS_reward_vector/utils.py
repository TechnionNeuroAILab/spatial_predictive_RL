"""Reward-vector supplementary figure — shared constants and Nature styling.

Tests two hippocampal-physiology phenomena, previously cited in the paper
only as narrative analogies, as actual quantitative results in the model:
  - Gauthier & Tank 2018 — reward-distance-modulated activity ("distance cells")
  - Sarel et al. 2017 — egocentric goal-vector tuning ("goal-direction cells"),
    with an allocentric head-direction control and a relocation (A->B)
    rotation test to establish goal-specificity.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Publication style: single source of truth in code/lib/pubstyle.py.
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
FIGURES_DIR = Path(__file__).resolve().parent / "figures_supp"
PANELS_DIR = FIGURES_DIR / "panels"
NEW_CODE = REPO_ROOT / "code" / "lib"

PAGE_MARGIN = 5.0 * MM_TO_PT

NOHEAD_DIR = REPO_ROOT / "results" / "multi_seed"
HEAD_DIR = REPO_ROOT / "results" / "multi_seed_head"
RELOC_DIR = REPO_ROOT / "results" / "multi_seed_reloc"

MIN_ENTROPY = 5.5           # collapse gate (grid-bin Shannon entropy of peak locations)
SKAGGS_THRESHOLD = 0.3


