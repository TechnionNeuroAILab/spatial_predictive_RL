from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
PANELS_DIR = FIGURES_DIR / "panels"

# Nature double-column width (183 mm) in matplotlib points.
MM_TO_PT = 72.0 / 25.4
PAGE_W = 183.0 * MM_TO_PT
GRID_COLS = 4
GRID_GAP = 2.0 * MM_TO_PT
ROW_GAP = 2.0 * MM_TO_PT  # small consistent white gap between rows
PAGE_MARGIN = 1.0 * MM_TO_PT
LABEL_SIZE = 8.0
TITLE_SIZE = 6.0  # panel title font (sans-serif, matches apply_nature_style)
TITLE_H = 9.0  # points reserved above each row for panel letters/titles
TITLE_LINE_H = 7.0  # extra points per additional title line (multi-line titles)
# Legacy equal-row constants (kept for compatibility; assembly now uses
# per-row heights computed from the actual panel aspect ratios).
ROW0_FRAC = ROW1_FRAC = ROW2_FRAC = 1.0 / 3.0
PAGE_H = 3 * 56.0 * MM_TO_PT + 2 * ROW_GAP + 2 * PAGE_MARGIN

# Composite layout: (panel_key, row, col, colspan)
# a = architecture; b = FoV; c = PCA 3D; d = remapping; e = peak; f = COM
FIGURE_LAYOUT = [
    ("a", 0, 0, 4),
    ("b", 1, 0, 1),
    ("c", 1, 1, 3),
    ("d", 2, 0, 2),
    ("e", 2, 2, 1),
    ("f", 2, 3, 1),
]
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
RATEMAP_CMAP = "inferno"

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


def apply_nature_style() -> None:
    """Matplotlib defaults aligned with Nature figure guidelines."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 0.75,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig: plt.Figure, stem: Path, dpi: int = 300) -> tuple[Path, Path]:
    """Save figure as PNG and PDF; stem may be with or without suffix."""
    stem = stem.with_suffix("")
    ensure_dir(stem.parent)
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    fig.savefig(png_path, format="png", dpi=dpi, facecolor="white", edgecolor="none", bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", dpi=dpi, facecolor="white", edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")
    return png_path, pdf_path


def png_to_pdf(png_path: Path, pdf_path: Path) -> None:
    from PIL import Image

    ensure_dir(pdf_path.parent)
    with Image.open(png_path) as img:
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.save(pdf_path, "PDF", resolution=150.0)
