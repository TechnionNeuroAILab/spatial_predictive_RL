"""Figure 5 shared constants and Nature styling.

Figure 5 — Dissociating "policy-type" (occupancy-driven) remapping from
"real" (reward-prediction-error-driven) representational remapping, and
translating the simulation evidence into falsifiable experimental predictions.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
PANELS_DIR = FIGURES_DIR / "panels"
NEW_CODE = REPO_ROOT / "paper_code" / "lib"

MM_TO_PT = 72.0 / 25.4
PAGE_W = 183.0 * MM_TO_PT
PAGE_MARGIN = 5.0 * MM_TO_PT

# Data locations (gated / entropy-clean seed lists, see analysis.py for the gate)
NOHEAD_DIR = REPO_ROOT / "Repro" / "multi_seed"
HEAD_DIR = REPO_ROOT / "Repro" / "multi_seed_head"
RELOC_DIR = REPO_ROOT / "Repro" / "multi_seed_reloc"

MIN_ENTROPY = 5.5           # collapse gate (grid-bin Shannon entropy of peak locations)
SKAGGS_THRESHOLD = 0.3

COLOR_NOHEAD = "#4c72b0"    # blue  — policy-type / occupancy control (Fig 2)
COLOR_HEAD = "#8B4513"      # brown — real remapping (Fig 3)
COLOR_A = "#2a9d8f"         # teal  — reward A (Fig 4)
COLOR_B = "#e76f51"         # orange — reward B (Fig 4)


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
