"""Figure 4 shared constants and Nature styling."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
PANELS_DIR = FIGURES_DIR / "panels"
NEW_CODE = REPO_ROOT / "New_Code"

MM_TO_PT = 72.0 / 25.4
PAGE_W = 183.0 * MM_TO_PT
PAGE_MARGIN = 5.0 * MM_TO_PT
LABEL_SIZE = 8.0

M = 32
HIDDEN_DIM = 512

# Reward positions
REWARD_A = np.array([0.3, 0.3])
REWARD_B = np.array([0.7, 0.7])

# Period names in the data directory
PERIOD_PRE = "replay_reloc_pred"
PERIOD_POST_A = "replay_reloc_postA"
PERIOD_POST_B = "replay_reloc_postB"

PERIODS = [
    (PERIOD_PRE, "baseline\n(pre-RL)"),
    (PERIOD_POST_A, "after A"),
    (PERIOD_POST_B, "after B\n(relocated)"),
]

# Phase colours for tracking curves
COLOR_A = "#2a9d8f"   # teal — reward A
COLOR_B = "#e76f51"   # orange — reward B
PERIOD_COLORS = {
    PERIOD_PRE: "#999999",
    PERIOD_POST_A: COLOR_A,
    PERIOD_POST_B: COLOR_B,
}

# Default run: the displayed network is the MEDIAN entropy-gate-passing seed
# by tracking quality (rank 17 of 32 clean seeds; see
# Repro/multi_seed_reloc/aggregate/fig4_multi_seed_summary.json and
# New_Code/select_best_seed_fig4.py for the ranking). Entropy gate for this
# seed: baseline 7.55 / after A 7.60 / after B 7.47 bits (all >= 5.5).
DEFAULT_RUN = REPO_ROOT / "Repro" / "multi_seed_reloc" / "reloc_seed9"
SKAGGS_THRESHOLD = 0.15

# Entropy collapse gate (matches Figure 5's diagnostic; see New_Code/analysis_helpers.py)
MIN_ENTROPY = 5.5


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
