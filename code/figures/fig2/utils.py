"""Figure 2 shared constants and Nature styling."""
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

# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------
# Same two-row plate as manuscript Figure 4 (code/figures/fig3): a wide
# spatial panel beside a narrower statistic, then the clouds beside the
# distance histograms. Both rows are the same height.
#
#   row 1   a  paired example fields          c  change in distance to reward
#   row 2   b  field-centre clouds             d  distance distributions
LAYOUT = figlayout.Layout(
    [("a", 0, 0, 4), ("c", 0, 4, 2),
     ("b", 1, 0, 3), ("d", 1, 3, 3)],
    ncols=6,
    row_h_in=[1.80, 1.80],
    caption_lines=7,
    name="fig_replay",
)

PANEL_ORDER = LAYOUT.keys()


def slot_size_in(key: str) -> tuple[float, float]:
    """The figsize a panel must be authored at to survive assembly unscaled."""
    return LAYOUT.slot_size_in(key)

# Shared with Figure 1 rate_maps import (same symbols expected by that module).

M = 32
HIDDEN_DIM = 512
# Canonical exemplar shared by manuscript Figures 1 and 2.  Keep this default
# aligned with code/regenerate_all.py and code/verify_paper.py.
DEFAULT_RUN = REPO_ROOT / "results" / "multi_seed" / "nohead_seed16"
DEFAULT_REWARD = (0.3, 0.3)

# Phase labels used throughout Figure 2.
PHASE_A = "replay_c1_pred"       # exploration before learning (pre-learning weights, random walk)
PHASE_B = "cycle_1_goal"         # goal-directed probe after learning (reset transients dropped)
PHASE_C_PRE = "replay_c1_pred"   # Phase C — frozen replay, pre-RL weights
PHASE_C_POST = "replay_c1_postrl"  # Phase C — frozen replay, post-RL weights


