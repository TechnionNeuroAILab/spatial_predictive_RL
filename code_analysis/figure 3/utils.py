"""Figure 3 shared constants and Nature styling.

Data structure is identical to Figure 2 (same phase names, same file layout).
Only the run directory and interpretation differ (head=ON → fields DO move).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
PANELS_DIR = FIGURES_DIR / "panels"
FIG1_DIR = Path(__file__).resolve().parents[1] / "figure 1"
FIG2_DIR = Path(__file__).resolve().parents[1] / "figure 2"

MM_TO_PT = 72.0 / 25.4
PAGE_W = 183.0 * MM_TO_PT
GRID_COLS = 4
GRID_GAP = 2.0 * MM_TO_PT
ROW_GAP = 0.0
PAGE_MARGIN = 1.0 * MM_TO_PT
LABEL_SIZE = 8.0

# Same layout as Figure 2: three equal-height rows
FIGURE_LAYOUT = [
    ("a", 0, 0, 4),
    ("b", 1, 0, 3),
    ("c", 1, 3, 1),
    ("d", 2, 0, 4),
]
PANEL_ORDER = ["a", "b", "c", "d"]

RATEMAP_CMAP = "inferno"
M = 32
HIDDEN_DIM = 512

# Default run: best head-condition seed (updated by select_best_seed_fig3.py)
DEFAULT_RUN = REPO_ROOT / "Repro" / "multi_seed_head" / "head_seed0"
DEFAULT_REWARD = (0.3, 0.3)

# Same phase labels as Figure 2 (data files are identically named)
PHASE_A = "cycle_1"
PHASE_B = "cycle_1_goal"
PHASE_C_PRE = "replay_c1_pred"
PHASE_C_POST = "replay_c1_postrl"

PHASE_COLORS = {
    "A": "#4C72B0",
    "B": "#C44E52",
    "C_pre": "#55A868",
    "C_post": "#8172B2",
}


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
