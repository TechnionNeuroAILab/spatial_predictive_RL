"""Figure 1 shared constants, page geometry and Nature styling.

Re-exports the publication style from lib/pubstyle.py so panel modules can
import it from here, and defines the composite page layout: which panel sits
in which grid slot, and how wide that slot is on the final page.

"""
from __future__ import annotations

from pathlib import Path

import numpy as np

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

# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------
# The architecture schematic used to be panel a of this figure, squeezed into a
# 150 pt row cap that scaled it to 0.548 -- its 9.75 pt labels printed at
# 4.7 pt, and 77 pt of blank page sat either side of it. It is now a figure in
# its own right (`MODEL_STEM` below, assembled by generate_figure1.py), which
# lets it run the full text width at its native aspect ratio.
#
# The remaining panels therefore each move up one letter: what this figure
# used to call b-f is now a-e.
#
#   a  environment & egocentric observations      b  hidden-state manifold
#   c  example units: prediction vs goal read-out d  field peak   e  centre of mass
LAYOUT = figlayout.Layout(
    [("a", 0, 0, 2), ("b", 0, 2, 2),
     ("c", 1, 0, 2), ("d", 1, 2, 1), ("e", 1, 3, 1)],
    ncols=4,
    # Panel a shares this row with the manifold. 1.85 in is a modest shrink
    # from 2.10 so the environment panel is a little smaller on the page.
    row_h_in=[1.85, 1.45],
    caption_lines=8,
    name="fig_map",
)

PANEL_ORDER = LAYOUT.keys()

# Short titles set beside the panel letter. Panels d/e carry their own titles
# inside the rendered panel PDF, so they are omitted here to avoid doubling up.
PANEL_TITLES = {
    "a": "Environment and egocentric observation",
    "b": "Hidden-state manifold (Phase A)",
    "c": "Example units: exploration before learning vs goal-directed after",
}

MODEL_STEM = "figure_model_raw"      # the standalone architecture figure


def slot_width_in(key: str) -> float:
    """On-page width of a panel's slot, in inches.

    Kept as a module-level function because the panel factories in
    `panel_d.py` and `rate_maps.py` take it as `width_in=`. It now just
    forwards to the layout, which owns the height as well.
    """
    return LAYOUT.slot_width_in(key)


def slot_size_in(key: str) -> tuple[float, float]:
    """Both dimensions of a panel's slot -- what its figsize must be."""
    return LAYOUT.slot_size_in(key)

# Perceptually uniform colormap for rate maps (Nature-friendly).

M = 32
HIDDEN_DIM = 512
# No-head, lr=5e-5, reward at (0.3, 0.3).
# Canonical exemplar shared by manuscript Figures 1 and 2.  Keep this default
# aligned with code/regenerate_all.py and code/verify_paper.py.
DEFAULT_RUN = REPO_ROOT / "results" / "multi_seed" / "nohead_seed16"
DEFAULT_REWARD = (0.3, 0.3)
DEFAULT_ARCHITECTURE = (
    REPO_ROOT / "code" / "figures" / "fig1" / "figures" / "panels" / "v11_architecture_vectorized.pdf"
)
DEFAULT_ARCHITECTURE_PNG = (
    REPO_ROOT / "code" / "figures" / "fig1" / "figures" / "panels" / "panel_a_architecture.png"
)

GRID_CENTERS = np.array([[(x / M, y / M) for y in range(M)] for x in range(M)])
HDS = np.array([
    (1, 0), (np.sqrt(2) / 2, np.sqrt(2) / 2), (0, 1), (-np.sqrt(2) / 2, np.sqrt(2) / 2),
    (-1, 0), (-np.sqrt(2) / 2, -np.sqrt(2) / 2), (0, -1), (np.sqrt(2) / 2, -np.sqrt(2) / 2),
])

OBJECT_TYPE_COLORS = [
    (0.12, 0.47, 0.71),  # type 0 — matplotlib C0
    (1.00, 0.50, 0.05),  # type 1 — C1
    (0.17, 0.63, 0.17),  # type 2 — C2
]
DIST_RAY_COLOR = (0.55, 0.34, 0.75)


def png_to_pdf(png_path: Path, pdf_path: Path) -> None:
    """Convert a rendered PNG panel to a single-page PDF at 150 dpi."""
    from PIL import Image

    ensure_dir(pdf_path.parent)
    with Image.open(png_path) as img:
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.save(pdf_path, "PDF", resolution=150.0)
