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

# Nature double-column width (183 mm) in matplotlib points.
GRID_COLS = 4
GRID_GAP = 2.0 * MM_TO_PT
ROW_GAP = 2.0 * MM_TO_PT  # small consistent white gap between rows
PAGE_MARGIN = 1.0 * MM_TO_PT
TITLE_SIZE = 6.0  # panel title font (sans-serif, matches apply_nature_style)
TITLE_H = 9.0  # points reserved above each row for panel letters/titles
TITLE_LINE_H = 7.0  # extra points per additional title line (multi-line titles)
# Legacy equal-row constants (kept for compatibility; assembly now uses
# per-row heights computed from the actual panel aspect ratios).
ROW0_FRAC = ROW1_FRAC = ROW2_FRAC = 1.0 / 3.0
PAGE_H = 3 * 56.0 * MM_TO_PT + 2 * ROW_GAP + 2 * PAGE_MARGIN

# Composite layout: (panel_key, row, col, colspan)
# a = architecture; b = FoV; c = PCA 3D; d = remapping; e = peak; f = COM
# Row 1 gives panel b two columns instead of one: its content is ~300 pt wide,
# so a single 125 pt column scaled it to ~42% and its labels fell below the
# 5 pt legibility floor. Panel c keeps enough width for three 3D views now that
# two of its three redundant colourbars are gone.
FIGURE_LAYOUT = [
    ("a", 0, 0, 4),
    ("b", 1, 0, 2),
    ("c", 1, 2, 2),
    ("d", 2, 0, 2),
    ("e", 2, 2, 1),
    ("f", 2, 3, 1),
]

# Per-row height caps (points). Row 0 is the architecture schematic.
ROW_MAX_H = {0: 150.0}


def slot_width_pt(key: str) -> float:
    """Final on-page width of a panel's slot, in points.

    Panels must be RENDERED at the width they are DISPLAYED at. The assembler
    contain-fits each panel PDF into its slot, so a panel rendered 570 pt wide
    and shown in a 253 pt slot has all of its type scaled to 44% -- which is how
    7 pt labels ended up at ~3 pt on the page. Renderers take their figsize from
    this function so the type scale survives assembly.
    """
    col_w = (PAGE_W - 2 * PAGE_MARGIN - (GRID_COLS - 1) * GRID_GAP) / GRID_COLS
    for k, _row, _col, colspan in FIGURE_LAYOUT:
        if k == key:
            return colspan * col_w + (colspan - 1) * GRID_GAP
    return col_w


def slot_width_in(key: str) -> float:
    return slot_width_pt(key) / 72.0
PANEL_ORDER = ["a", "b", "c", "d", "e", "f"]

# Short titles drawn next to the panel letters in the assembled composite.
# Panels e/f already carry titles inside the rendered panel PDFs, so they are
# intentionally omitted here to avoid duplication.
PANEL_TITLES = {
    "a": "Model architecture",
    "b": "Environment &\negocentric observations",
    "c": "Hidden-state manifold mirrors the arena",
    "d": "Example units: prediction probe vs goal-directed read-out",
}

# Perceptually uniform colormap for rate maps (Nature-friendly).

M = 32
HIDDEN_DIM = 512
# No-head, lr=5e-5, reward at (0.3, 0.3).
DEFAULT_RUN = REPO_ROOT / "Repro" / "nohead_seed42"
DEFAULT_REWARD = (0.3, 0.3)
DEFAULT_ARCHITECTURE = (
    REPO_ROOT / "paper_code" / "figures" / "fig1" / "figures" / "panels" / "v11_architecture_vectorized.pdf"
)
DEFAULT_ARCHITECTURE_PNG = (
    REPO_ROOT / "paper_code" / "figures" / "fig1" / "figures" / "panels" / "panel_a_architecture.png"
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
    from PIL import Image

    ensure_dir(pdf_path.parent)
    with Image.open(png_path) as img:
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.save(pdf_path, "PDF", resolution=150.0)
