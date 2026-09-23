"""Figure 3 shared constants and Nature styling.

Data structure is identical to Figure 2 (same phase names, same file layout).
Only the run directory and interpretation differ (head=ON → fields do move).
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
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
PANELS_DIR = FIGURES_DIR / "panels"
FIG1_DIR = Path(__file__).resolve().parents[1] / "fig1"
FIG2_DIR = Path(__file__).resolve().parents[1] / "fig2"

GRID_COLS = 4
GRID_GAP = 2.0 * MM_TO_PT
ROW_GAP = 0.0
PAGE_MARGIN = 1.0 * MM_TO_PT

# Same layout as Figure 2: three equal-height rows
FIGURE_LAYOUT = [
    ("a", 0, 0, 4),
    ("b", 1, 0, 3),
    ("c", 1, 3, 1),
    ("d", 2, 0, 4),
]
PANEL_ORDER = ["a", "b", "c", "d"]

M = 32
HIDDEN_DIM = 512

# Default run: best head-condition seed (updated by select_best_seed_fig3.py)
DEFAULT_RUN = REPO_ROOT / "results" / "multi_seed_head" / "head_seed0"
DEFAULT_REWARD = (0.3, 0.3)

# Same phase labels as Figure 2 (data files are identically named)
PHASE_A = "cycle_1"
PHASE_B = "cycle_1_goal"
PHASE_C_PRE = "replay_c1_pred"
PHASE_C_POST = "replay_c1_postrl"


