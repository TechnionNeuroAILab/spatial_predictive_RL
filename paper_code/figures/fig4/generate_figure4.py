#!/usr/bin/env python3
"""
Figure 4 — reward relocation (Paper Figure 4).

Place fields leave reward A and migrate to reward B when the reward is relocated.
Demonstrated on a frozen random-walk trajectory replayed through three weight
snapshots (baseline / postA / postB) — occupancy is identical in all three.

    python "code_analysis/figure 4/generate_figure4.py" --no_show
    python "code_analysis/figure 4/generate_figure4.py" \\
        --run_dir Repro/multi_seed_reloc/reloc_seed9 --no_show

Panels
  a  Example tracking cells: 3×N rate maps (baseline / postA / postB)
  b  Field-centre (COM) clouds: three periods, teal/orange stars at A/B
  c  Mean distance-to-A and distance-to-B across periods (crossing curves)
  d  Δd violins: shift toward A (pre→postA) and toward B (postA→postB)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from shutil import copy2

import matplotlib.pyplot as plt
import numpy as np
from pypdf import PdfReader, PdfWriter, Transformation

from analysis import (
    load_reloc_states,
    paired_tracking_stats,
    fig_tracking_curves,
    fig_shift_violins,
    fig_com_scatter_three,
    fig_example_cells_grid,
    seed_entropy_status,
)
from utils import (
    DEFAULT_RUN,
    FIGURES_DIR,
    LABEL_SIZE,
    MM_TO_PT,
    PAGE_MARGIN,
    apply_nature_style,
    ensure_dir,
    save_figure,
    SKAGGS_THRESHOLD,
    MIN_ENTROPY,
    PERIODS,
)

apply_nature_style()

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "paper_code" / "lib"))

# Assembled-figure page width. Narrower than the 183 mm double column the
# panels used to be stretched across — rows keep a sensible width:height
# ratio and panels sit near their natural (matplotlib) size.
FIG_PAGE_W = 150.0 * MM_TO_PT


def _pdf_to_png(pdf_path: Path, png_path: Path, dpi: int = 300) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out_prefix = Path(tmp) / "page"
        subprocess.run(
            ["pdftoppm", "-png", "-r", str(dpi), "-singlefile", str(pdf_path), str(out_prefix)],
            check=True,
        )
        copy2(out_prefix.with_suffix(".png"), png_path)


def _write_letter_overlay_pdf(placements: list[dict], page_h: float, out_path: Path) -> Path:
    """Transparent overlay with bold panel letters at each slot's top-left."""
    ensure_dir(out_path.parent)
    fig = plt.figure(figsize=(FIG_PAGE_W / 72.0, page_h / 72.0))
    fig.patch.set_facecolor("none")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_PAGE_W)
    ax.set_ylim(0, page_h)
    ax.axis("off")
    ax.set_facecolor("none")
    for p in placements:
        ax.text(p["left"] + 2.0, p["bottom"] + p["slot_h"] - 2.0, p["key"],
                fontsize=LABEL_SIZE, fontweight="bold", va="top", ha="left", color="black")
    fig.savefig(out_path, format="pdf", transparent=True, facecolor="none",
                edgecolor="none", bbox_inches=None, pad_inches=0)
    plt.close(fig)
    return out_path


def assemble_figure(panel_pdfs: dict[str, Path], stem, save_dpi: int = 300):
    """Layout: [a] / [b] / [c d], with panel letters."""
    stem = Path(stem).with_suffix("")
    ensure_dir(stem.parent)
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")

    MARGIN = PAGE_MARGIN
    GAP = 3 * MM_TO_PT
    ROW_GAP = 3 * MM_TO_PT

    def get_size(pdf):
        box = PdfReader(str(pdf)).pages[0].mediabox
        return float(box.width), float(box.height)

    sizes = {k: get_size(v) for k, v in panel_pdfs.items() if v is not None}
    available = {k for k in panel_pdfs if k in sizes}

    # Row definitions
    rows_def = [("a",), ("b",), ("c", "d")]
    row_keys = [[k for k in row if k in available] for row in rows_def]
    row_keys = [r for r in row_keys if r]

    def row_height(keys):
        ncols = len(keys)
        cw = (FIG_PAGE_W - 2 * MARGIN - (ncols - 1) * GAP) / ncols
        return max((sizes[k][1] * (cw / max(sizes[k][0], 1e-6)) for k in keys), default=0.0)

    rhs = [row_height(r) for r in row_keys]
    page_h = 2 * MARGIN + sum(rhs) + (len(rhs) - 1) * ROW_GAP

    writer = PdfWriter()
    page = writer.add_blank_page(width=FIG_PAGE_W, height=page_h)

    placements = []
    y = MARGIN
    for keys, rh in zip(reversed(row_keys), reversed(rhs)):
        ncols = len(keys)
        cw = (FIG_PAGE_W - 2 * MARGIN - (ncols - 1) * GAP) / ncols
        for i, k in enumerate(keys):
            iw, ih = sizes[k]
            scale = min(cw / iw, rh / ih)
            sw, sh = iw * scale, ih * scale
            left = MARGIN + i * (cw + GAP) + (cw - sw) / 2
            bottom = y + (rh - sh) / 2
            src = PdfReader(str(panel_pdfs[k])).pages[0]
            page.merge_transformed_page(src, Transformation().scale(scale).translate(left, bottom), over=True)
            placements.append({"key": k, "left": MARGIN + i * (cw + GAP),
                               "bottom": y, "slot_h": rh})
        y += rh + ROW_GAP

    overlay_path = stem.parent / "_figure4_letters_overlay.pdf"
    _write_letter_overlay_pdf(placements, page_h, overlay_path)
    page.merge_page(PdfReader(str(overlay_path)).pages[0], over=True)
    with open(pdf_path, "wb") as f:
        writer.write(f)
    try:
        overlay_path.unlink(missing_ok=True)
    except OSError:
        pass

    _pdf_to_png(pdf_path, png_path, dpi=save_dpi)
    print(f"  Saved: {pdf_path}")
    print(f"  Saved: {png_path}")
    return png_path, pdf_path


def generate_all(args):
    run_dir = Path(args.run_dir).resolve()
    data_dir = run_dir / "data"
    panels_dir = Path(args.output_dir).resolve() / "panels"
    figures_dir = Path(args.output_dir).resolve()
    ensure_dir(panels_dir)

    panel_pdfs: dict[str, Path] = {}

    # Entropy-collapse gate (shared diagnostic; must hold in EVERY period)
    print("Entropy-collapse gate (>= %.1f bits in every period):" % MIN_ENTROPY)
    from analysis_helpers import peak_entropy  # noqa: E402 (New_Code on path)
    ents = {p: peak_entropy(data_dir, p) for p, _ in PERIODS}
    for (p, label), e in zip(PERIODS, ents.values()):
        print(f"  {label.replace(chr(10), ' '):22s} {p}: "
              + (f"{e:.3f} bits" if e is not None else "n/a"))
    if not seed_entropy_status(data_dir):
        print(f"  WARNING: this seed FAILS the entropy gate (min < {MIN_ENTROPY}) — "
              "pick a clean seed (see select_best_seed_fig4.py)")
    else:
        print("  PASS")

    if "a" in args.panels:
        print("Panel a (example tracking cells)...")
        _, pdf = fig_example_cells_grid(run_dir, panels_dir / "panel_a",
                                        n_examples=args.n_examples,
                                        skaggs_threshold=args.skaggs_threshold)
        panel_pdfs["a"] = pdf

    # Load states for b, c, d
    need_states = args.panels & {"b", "c", "d"}
    states = None
    ts = None
    if need_states:
        print("Loading relocation states...")
        states = load_reloc_states(data_dir, skaggs_threshold=args.skaggs_threshold)
        ts = paired_tracking_stats(states, skaggs_threshold=args.skaggs_threshold)
        print(f"  Shift toward A (pre→postA): mean={ts['shift_A']['mean_delta']:+.4f}  "
              f"frac={100*ts['shift_A']['frac_toward']:.0f}%  p={ts['shift_A']['wilcoxon_p']:.1e}")
        print(f"  Shift toward B (postA→postB): mean={ts['shift_B']['mean_delta']:+.4f}  "
              f"frac={100*ts['shift_B']['frac_toward']:.0f}%  p={ts['shift_B']['wilcoxon_p']:.1e}")

    if "b" in args.panels and states is not None:
        print("Panel b (field-centre scatter, three periods)...")
        _, pdf = fig_com_scatter_three(states, panels_dir / "panel_b",
                                       skaggs_threshold=args.skaggs_threshold)
        panel_pdfs["b"] = pdf

    if "c" in args.panels and ts is not None:
        print("Panel c (tracking curves — distances to A and B)...")
        _, pdf = fig_tracking_curves(ts, panels_dir / "panel_c")
        panel_pdfs["c"] = pdf

    if "d" in args.panels and ts is not None:
        print("Panel d (shift violins)...")
        _, pdf = fig_shift_violins(ts, panels_dir / "panel_d")
        panel_pdfs["d"] = pdf

    if not args.no_assemble and panel_pdfs:
        print("Assembling vector composite PDF...")
        assemble_figure(panel_pdfs, figures_dir / "figure4_raw")

    print("\nDone.")
    print(f"  Figures: {figures_dir}")
    print(f"  Run dir: {run_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Generate Figure 4 (reward relocation).")
    p.add_argument("--run_dir", default=str(DEFAULT_RUN))
    p.add_argument("--output_dir", default=str(FIGURES_DIR))
    p.add_argument("--n_examples", type=int, default=6, help="Example cells in panel a")
    p.add_argument("--panels", default="a,b,c,d")
    p.add_argument("--skaggs_threshold", type=float, default=SKAGGS_THRESHOLD)
    p.add_argument("--no_show", action="store_true")
    p.add_argument("--no_assemble", action="store_true")
    args = p.parse_args()
    args.panels = {x.strip().lower() for x in args.panels.split(",") if x.strip()}
    return args


if __name__ == "__main__":
    generate_all(parse_args())
