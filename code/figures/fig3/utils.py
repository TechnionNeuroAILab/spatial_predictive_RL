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
    plain_log_ticks,
)
import figlayout  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[3]
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
PANELS_DIR = FIGURES_DIR / "panels"
FIG1_DIR = Path(__file__).resolve().parents[1] / "fig1"
FIG2_DIR = Path(__file__).resolve().parents[1] / "fig2"

# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------
# Two equal rows. Panel c (the violin) and panel d (the histograms) are both
# a little shorter than the previous 2.05 / 1.95 in rows. Manuscript Figure 3
# uses this same grid.
#   row 1   a  paired example fields     c  change in distance to reward
#   row 2   b  field-centre clouds        d  distance distributions
LAYOUT = figlayout.Layout(
    [("a", 0, 0, 4), ("c", 0, 4, 2),
     ("b", 1, 0, 3), ("d", 1, 3, 3)],
    ncols=6,
    row_h_in=[1.80, 2.10],
    caption_lines=7,
    name="fig_proximity",
)

PANEL_ORDER = LAYOUT.keys()


def slot_size_in(key: str) -> tuple[float, float]:
    """The figsize a panel must be authored at to survive assembly unscaled."""
    return LAYOUT.slot_size_in(key)

M = 32
HIDDEN_DIM = 512

# The manuscript uses head_seed28, whose replay shifts are close to the cohort
# means. These well-tuned units make the reward-directed component visible in
# panel a; the full distribution, including shifts away from reward, is shown
# in the population panels.
DEFAULT_RUN = REPO_ROOT / "results" / "multi_seed_head" / "head_seed28"
PANEL_A_NEURON_IDS = [35, 64, 184, 214, 356, 390, 413, 83]
DEFAULT_REWARD = (0.3, 0.3)

# Same phase labels as Figure 2 (data files are identically named)
PHASE_A = "replay_c1_pred"   # exploration before learning (as fig2/utils.py)
PHASE_B = "cycle_1_goal"
PHASE_C_PRE = "replay_c1_pred"
PHASE_C_POST = "replay_c1_postrl"


