"""Figure 2 shared constants and Nature styling."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
PANELS_DIR = FIGURES_DIR / "panels"
FIG1_DIR = Path(__file__).resolve().parents[1] / "fig1"

MM_TO_PT = 72.0 / 25.4
PAGE_W = 183.0 * MM_TO_PT
GRID_COLS = 4
GRID_GAP = 2.0 * MM_TO_PT
ROW_GAP = 0.0
PAGE_MARGIN = 1.0 * MM_TO_PT
LABEL_SIZE = 8.0

# Three equal-height rows: a | b+c | d
FIGURE_LAYOUT = [
    ("a", 0, 0, 4),
    ("b", 1, 0, 3),
    ("c", 1, 3, 1),
    ("d", 2, 0, 4),
]
PANEL_ORDER = ["a", "b", "c", "d"]

# Shared with Figure 1 rate_maps import (same symbols expected by that module).
RATEMAP_CMAP = "inferno"

M = 32
HIDDEN_DIM = 512
DEFAULT_RUN = REPO_ROOT / "Repro" / "nohead_seed42"
DEFAULT_REWARD = (0.3, 0.3)

# Phase labels used throughout Figure 2.
PHASE_A = "cycle_1"              # Phase A — prediction / open-loop RW probe
PHASE_B = "cycle_1_goal"         # Phase B — RL / goal-conditioned probe
PHASE_C_PRE = "replay_c1_pred"   # Phase C — frozen replay, pre-RL weights
PHASE_C_POST = "replay_c1_postrl"  # Phase C — frozen replay, post-RL weights

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
