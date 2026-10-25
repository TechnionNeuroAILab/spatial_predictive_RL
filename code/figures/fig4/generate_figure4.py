#!/usr/bin/env python3
"""
Figure 4 — reward relocation (Paper Figure 4).

Place fields leave reward A and migrate to reward B when the reward is relocated.
Demonstrated on a frozen random-walk trajectory replayed through three weight
snapshots (baseline / postA / postB) — occupancy is identical in all three.

    python code/figures/fig4/generate_figure4.py --no_show
    python code/figures/fig4/generate_figure4.py \\
        --run_dir results/multi_seed_reloc/reloc_seed9 --no_show

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
    fig_shift_violins_both,
    fig_network_shifts_reloc,
    fig_com_scatter_three,
    fig_example_cells_grid,
    load_example_cell_maps,
    seed_entropy_status,
)
from utils import (
    DEFAULT_RUN,
    FIGURES_DIR,
    LAYOUT,
    MIN_ENTROPY,
    N_EXAMPLE_CELLS,
    PERIODS,
    SKAGGS_THRESHOLD,
    apply_nature_style,
    ensure_dir,
)
import figlayout
from figure_cache import add_plot_only, load_figure_data, require_keys, update_figure_data

apply_nature_style()

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code" / "lib"))


def _pdf_to_png(pdf_path: Path, png_path: Path, dpi: int = 300) -> None:
    """Render a one-page PDF to PNG at `dpi`, via Poppler's pdftoppm."""
    with tempfile.TemporaryDirectory() as tmp:
        out_prefix = Path(tmp) / "page"
        subprocess.run(
            ["pdftoppm", "-png", "-r", str(dpi), "-singlefile", str(pdf_path), str(out_prefix)],
            check=True,
        )
        copy2(out_prefix.with_suffix(".png"), png_path)


def assemble_figure(panel_pdfs: dict[str, Path], stem, save_dpi: int = 300):
    """Compose the panel PDFs onto the two-row page defined by `utils.LAYOUT`.

    Replaces a hand-rolled row solver that gave panel c a 0.852 scale beside
    panel d at 0.941 -- two panels side by side, the same declared 7 pt title,
    6.0 pt and 6.6 pt on the page.
    """
    return figlayout.compose(LAYOUT, panel_pdfs, stem, dpi=save_dpi)


def _figure_paths(args):
    """COM writes the manuscript stem; peak writes a sibling copy and its own panels."""
    figures_dir = Path(args.output_dir).resolve()
    if args.center == "peak":
        return figures_dir / "panels_peak", figures_dir / "figure4_peak_raw"
    return figures_dir / "panels", figures_dir / "figure4_raw"


def generate_all(args):
    """Report the entropy gate, render every requested panel, then assemble the figure."""
    run_dir = Path(args.run_dir).resolve()
    data_dir = run_dir / "data"
    panels_dir, figure_stem = _figure_paths(args)
    figures_dir = figure_stem.parent
    ensure_dir(panels_dir)

    panel_pdfs: dict[str, Path] = {}

    if getattr(args, "assemble_only", False):
        print("Assembly-only mode: reusing existing panel PDFs...")
        assemble_figure(figlayout.existing_panels(
            panels_dir, [k for k in LAYOUT.keys() if k in args.panels]),
            figure_stem)
        return

    cache_path = figures_dir / "figure_data.npz"
    if args.plot_only:
        saved = require_keys(
            load_figure_data(cache_path),
            "states", "ts_com", "ts_peak", "example_maps",
            path=cache_path,
        )
        states = saved["states"]
        example_maps = saved["example_maps"]
    else:
        # Entropy-collapse gate (shared diagnostic; must hold in every period)
        print("Entropy-collapse gate (>= %.1f bits in every period):" % MIN_ENTROPY)
        from analysis_helpers import peak_entropy  # noqa: E402 (lib/ on path)
        ents = {p: peak_entropy(data_dir, p) for p, _ in PERIODS}
        for (p, label), e in zip(PERIODS, ents.values()):
            print(f"  {label.replace(chr(10), ' '):22s} {p}: "
                  + (f"{e:.3f} bits" if e is not None else "n/a"))
        if not seed_entropy_status(data_dir):
            print(f"  WARNING: this seed FAILS the entropy gate (min < {MIN_ENTROPY}) — "
                  "pick a clean seed (see select_best_seed_fig4.py)")
        else:
            print("  PASS")

        print("Loading relocation states...")
        example_maps = load_example_cell_maps(
            run_dir, n_examples=args.n_examples, skaggs_threshold=args.skaggs_threshold)
        states = load_reloc_states(data_dir, skaggs_threshold=args.skaggs_threshold)
        saved = {
            "states": states,
            "example_maps": example_maps,
            "ts_com": paired_tracking_stats(
                states, skaggs_threshold=args.skaggs_threshold, center="com"),
            "ts_peak": paired_tracking_stats(
                states, skaggs_threshold=args.skaggs_threshold, center="peak"),
        }
        update_figure_data(cache_path, saved)
    ts = saved["ts_peak"] if args.center == "peak" else saved["ts_com"]
    print(f"  Shift toward A (pre→postA): mean={ts['shift_A']['mean_delta']:+.4f}  "
          f"frac={100*ts['shift_A']['frac_toward']:.0f}%  p={ts['shift_A']['wilcoxon_p']:.1e}")
    print(f"  Shift toward B (postA→postB): mean={ts['shift_B']['mean_delta']:+.4f}  "
          f"frac={100*ts['shift_B']['frac_toward']:.0f}%  p={ts['shift_B']['wilcoxon_p']:.1e}")

    if "a" in args.panels:
        print("Panel a (example tracking cells)...")
        _, pdf = fig_example_cells_grid(run_dir, panels_dir / "panel_a",
                                        size_in=LAYOUT.slot_size_in("a"),
                                        n_examples=args.n_examples,
                                        skaggs_threshold=args.skaggs_threshold,
                                        maps=example_maps)
        panel_pdfs["a"] = pdf

    if "b" in args.panels:
        print(f"Panel b ({args.center} scatter, three periods)...")
        _, pdf = fig_com_scatter_three(states, panels_dir / "panel_b",
                                       size_in=LAYOUT.slot_size_in("b"),
                                       skaggs_threshold=args.skaggs_threshold,
                                       center=args.center)
        panel_pdfs["b"] = pdf

    if "c" in args.panels:
        print("Panel c (per-network shifts toward A and B, n = 50)...")
        _, pdf = fig_network_shifts_reloc(panels_dir / "panel_c",
                                          size_in=LAYOUT.slot_size_in("c"))
        panel_pdfs["c"] = pdf

    if "d" in args.panels:
        print("Panel d (example network per-cell shifts, peak and COM)...")
        _, pdf = fig_shift_violins_both(saved["ts_peak"], saved["ts_com"],
                                        panels_dir / "panel_d",
                                        size_in=LAYOUT.slot_size_in("d"))
        panel_pdfs["d"] = pdf

    if not args.no_assemble and panel_pdfs:
        print("Assembling vector composite PDF...")
        assemble_figure(panel_pdfs, figure_stem)

    print("\nDone.")
    print(f"  Figures: {figures_dir}")
    print(f"  Run dir: {run_dir}")


def parse_args():
    """Command-line arguments: run directory, output paths, panel subset."""
    p = argparse.ArgumentParser(description="Generate Figure 4 (reward relocation).")
    p.add_argument("--run_dir", default=str(DEFAULT_RUN))
    p.add_argument("--output_dir", default=str(FIGURES_DIR))
    p.add_argument("--n_examples", type=int, default=N_EXAMPLE_CELLS,
                   help="Example cells in panel a")
    p.add_argument("--panels", default="a,b,c,d")
    p.add_argument("--skaggs_threshold", type=float, default=SKAGGS_THRESHOLD)
    p.add_argument("--center", choices=("com", "peak"), default="com",
                   help="field centre for shift, distance and scatter panels "
                        "(com = centre of mass, the manuscript default; "
                        "peak writes figure4_peak_raw)")
    p.add_argument("--no_show", action="store_true")
    p.add_argument("--no_assemble", action="store_true")
    add_plot_only(p)
    p.add_argument("--assemble_only", action="store_true",
                   help="recompose from the existing panel PDFs, skipping the "
                        "data load")
    args = p.parse_args()
    args.panels = {x.strip().lower() for x in args.panels.split(",") if x.strip()}
    return args


if __name__ == "__main__":
    generate_all(parse_args())
