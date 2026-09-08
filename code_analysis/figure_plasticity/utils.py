"""Figure 3 / Supplementary Figure 3 — shared constants."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
PANELS_DIR = FIGURES_DIR / "panels"
SUPP_FIGURES_DIR = Path(__file__).resolve().parent / "figures_supp"
SUPP_PANELS_DIR = SUPP_FIGURES_DIR / "panels"
FIG1_DIR = Path(__file__).resolve().parents[1] / "figure 1"
FIG2_DIR = Path(__file__).resolve().parents[1] / "figure 2"

MM_TO_PT = 72.0 / 25.4
PAGE_W = 183.0 * MM_TO_PT
GRID_COLS = 4
GRID_GAP = 2.0 * MM_TO_PT
ROW_GAP = 0.0
PAGE_MARGIN = 1.0 * MM_TO_PT
LABEL_SIZE = 8.0

# Main Figure 3: a (remap), b (centres), c (plasticity boxplots)
FIGURE_LAYOUT = [
    ("a", 0, 0, 4),
    ("b", 1, 0, 4),
    ("c", 2, 0, 4),
]
PANEL_ORDER = ["a", "b", "c"]

# Supplementary: 2×2
SUPP_LAYOUT = [
    ("a", 0, 0, 2),
    ("b", 0, 2, 2),
    ("c", 1, 0, 2),
    ("d", 1, 2, 2),
]
SUPP_PANEL_ORDER = ["a", "b", "c", "d"]

RATEMAP_CMAP = "inferno"
M = 32
HIDDEN_DIM = 512
DEFAULT_REWARD = (0.3, 0.3)

# High-plasticity single-cycle demo (no head, lr=1e-4, seed 42).
DEFAULT_RUN_1E4 = REPO_ROOT / "New_Results_v3" / "dose_sweep_lr_nohead" / "lr_0.0001"
# Plasticity sweep (no head, seed 42 per LR).
DEFAULT_SWEEP_ROOT = REPO_ROOT / "New_Results_v3" / "dose_sweep_lr_nohead"
# Extra seeds at 1e-4 (replay-only; used where A/B pickles are absent).
MATCHED_1E4_GLOB = "matched_nohead_1e-4_seed*"

LR_ORDER = [
    ("lr_2e-06", 2e-6),
    ("lr_1e-05", 1e-5),
    ("lr_5e-05", 5e-5),
    ("lr_0.0001", 1e-4),
    ("lr_0.0003", 3e-4),
]


def apply_nature_style() -> None:
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
    stem = Path(stem).with_suffix("")
    ensure_dir(stem.parent)
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    fig.savefig(png_path, format="png", dpi=dpi, facecolor="white", edgecolor="none",
                bbox_inches="tight", pad_inches=0.05)
    fig.savefig(pdf_path, format="pdf", dpi=dpi, facecolor="white", edgecolor="none",
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")
    return png_path, pdf_path
