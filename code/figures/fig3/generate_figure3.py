#!/usr/bin/env python3
"""
Figure 3 — reward-prediction head (Paper Figure 3).

Frozen-replay control with reward-prediction head ON.
Same four-panel layout as Figure 2, but now the head causes genuine
representational remapping toward the reward.

    python code/figures/fig3/generate_figure3.py --no_show
    python code/figures/fig3/generate_figure3.py \
        --run_dir results/multi_seed_head/head_seed28 \
        --reward_pos 0.3 0.3 --no_show

Panels
  a  Frozen-replay remapping grid of illustrative, well-tuned fields
  b  Field COM centres for Phases A, B, C
  c  Violin plots of COM Δd for A→B (policy) and C:pre→post (frozen replay)
  d  Matched plasticity control: near-rigid, plastic and plastic+teaching
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from shutil import copy2

import matplotlib.pyplot as plt
import numpy as np
from pypdf import PdfReader, PdfWriter, Transformation

from analysis import (
    fig_centers_abc,
    fig_network_shifts,
    fig_replay_remap_grid,
    load_phases,
    paired_distance_stats,
    shared_place_cell_ids,
)
from utils import (
    C,
    FS_SMALL,
    FS_TICK,
    RATEMAP,
    DEFAULT_REWARD,
    DEFAULT_RUN,
    FIGURES_DIR,
    LAYOUT,
    M,
    PANEL_A_NEURON_IDS,
    apply_nature_style,
    ensure_dir,
    save_figure,
    title,
    zero_line,
)
import figlayout
from figure_cache import add_plot_only, load_figure_data, require_keys, update_figure_data
from pubstyle import METRIC_ALPHA, METRIC_LABEL, load_headline_stats, network_strip

apply_nature_style()


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
    """Compose the panel PDFs onto the page defined by `utils.LAYOUT`.

    Replaces a geometry solver, a contain-fit and a hand-rolled letter overlay
    (~120 lines) that between them scaled each panel by a different factor, so
    the same declared font size printed at a different size in each row.
    `figlayout.compose` refuses to scale at all.
    """
    return figlayout.compose(LAYOUT, panel_pdfs, stem, dpi=save_dpi)


def _jsonable_stats(stats: dict) -> dict:
    """Narrow the stats dict to plain JSON-serialisable scalars for the cached summary."""
    return {
        "n": int(stats["n"]),
        "n_ac": int(stats["n_ac"]),
        "com_mean_delta_AB": float(stats["com_mean_delta_AB"]),
        "com_frac_toward_AB": float(stats["com_frac_toward_AB"]),
        "com_mean_delta_AC": float(stats["com_mean_delta_AC"]),
        "com_frac_toward_AC": float(stats["com_frac_toward_AC"]),
        "peak_mean_delta_AB": float(stats["peak_mean_delta_AB"]),
        "peak_frac_toward_AB": float(stats["peak_frac_toward_AB"]),
        "peak_mean_delta_AC": float(stats["peak_mean_delta_AC"]),
        "peak_frac_toward_AC": float(stats["peak_frac_toward_AC"]),
        "com_mean_A": float(np.mean(stats["com"]["A"])),
        "com_mean_B": float(np.mean(stats["com"]["B"])),
        "com_mean_C_pre": float(np.mean(stats["com"]["C_pre"])),
        "com_mean_C": float(np.mean(stats["com"]["C"])),
        "ac_contrast": "replay_c1_pred -> replay_c1_postrl (frozen trajectory, weight-only)",
    }


def fig_plasticity_control(stem, size_in):
    """Matched fixed-trajectory shifts with near-rigid and plastic recurrent weights."""
    hs = load_headline_stats()
    seeds = set(hs["paired"]["cohort"]["seeds"])
    rows = [r for r in hs["paired"]["rows"] if r["seed"] in seeds]
    fig, ax = plt.subplots(figsize=size_in)
    arms = (("rg", C["rigid"]), ("nh", C["nohead"]), ("hd", C["head"]))
    for metric, x0 in (("peak", 0.0), ("com", 3.6)):
        vals = [np.array([r[f"{arm}_frz_{metric}"]["mean"] for r in rows])
                for arm, _ in arms]
        network_strip(
            ax, vals, [x0, x0 + 1, x0 + 2], [colour for _, colour in arms],
            alpha=0.55 * METRIC_ALPHA[metric] + 0.1, seed=int(x0),
        )
        ax.text(
            x0 + 1, -0.17, METRIC_LABEL[metric],
            transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=FS_TICK,
        )
    zero_line(ax)
    ax.set_xticks([0, 1, 2, 3.6, 4.6, 5.6])
    ax.set_xticklabels(
        ["Rigid", "Plastic", "Plastic +\nteaching"] * 2,
        fontsize=FS_SMALL,
    )
    ax.set_xlim(-0.6, 6.2)
    ax.set_ylabel(r"$\Delta$ distance to reward (replay)")
    title(ax, f"Plasticity control (n = {len(rows)} triples)")
    fig.tight_layout(pad=0.35)
    return save_figure(fig, stem, exact_size=True)


def _figure_paths(args):
    """COM writes the manuscript stem; peak writes a sibling copy and its own panels."""
    figures_dir = Path(args.output_dir).resolve()
    if args.center == "peak":
        return figures_dir / "panels_peak", figures_dir / "figure3_peak_raw"
    return figures_dir / "panels", figures_dir / "figure3_raw"


def generate_all(args):
    """Render every requested panel, then assemble the composite figure."""
    run_dir = Path(args.run_dir).resolve()
    data_dir = run_dir / "data"
    panels_dir, figure_stem = _figure_paths(args)
    figures_dir = figure_stem.parent
    ensure_dir(panels_dir)

    reward_pos = np.array(args.reward_pos, dtype=float)
    reward_bin = (int(reward_pos[0] * M), int(reward_pos[1] * M))
    panel_pdfs: dict[str, Path] = {}

    if getattr(args, "assemble_only", False):
        print("Assembly-only mode: reusing existing panel PDFs...")
        panel_pdfs = figlayout.existing_panels(
            panels_dir, [k for k in LAYOUT.keys() if k in args.panels])
        assemble_figure(panel_pdfs, figure_stem)
        return

    cache_path = figures_dir / "figure_data.npz"
    if args.plot_only:
        saved = require_keys(
            load_figure_data(cache_path), "phases", "stats", "neuron_ids", path=cache_path)
        phases = saved["phases"]
        stats = saved["stats"]
        neuron_ids = [int(x) for x in saved["neuron_ids"]]
    else:
        print("Loading phases A / B / C ...")
        phases = load_phases(data_dir)
        if args.neuron_ids:
            neuron_ids = [int(x) for x in args.neuron_ids]
        else:
            neuron_ids = list(PANEL_A_NEURON_IDS[:args.n_neurons])
        print(f"  Neuron ids: {neuron_ids}")
        ids = shared_place_cell_ids(phases, skaggs_threshold=args.skaggs_threshold)
        stats = paired_distance_stats(phases, reward_pos, ids)
        update_figure_data(cache_path, {
            "phases": phases, "stats": stats, "neuron_ids": neuron_ids,
        })
    print(
        f"  Shared place cells (A∩B): n={stats['n']} "
        f"(Skaggs ≥ {args.skaggs_threshold})"
    )
    print(
        f"  A→B: COM Δd={stats['com_mean_delta_AB']:+.3f} "
        f"({100 * stats['com_frac_toward_AB']:.0f}% toward)"
    )
    print(
        f"  C:pre→post (frozen replay, n={stats['n_ac']}): COM Δd={stats['com_mean_delta_AC']:+.3f} "
        f"({100 * stats['com_frac_toward_AC']:.0f}% toward)"
    )
    summary_path = panels_dir / "phase_abc_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_jsonable_stats(stats), f, indent=2)
    print(f"  Saved: {summary_path}")

    if "a" in args.panels:
        print("Panel a (frozen-replay remapping — WITH head, fields MOVE)...")
        _, pdf = fig_replay_remap_grid(
            phases["C_pre"]["rm"], phases["C_post"]["rm"], neuron_ids, reward_bin,
            stem=panels_dir / "panel_a",
            cmap=RATEMAP,
            size_in=LAYOUT.slot_size_in("a"),
            # No sentence title: it collided with the panel letter and is
            # caption material. The head-ON condition is stated in the caption.
        )
        panel_pdfs["a"] = pdf

    if "b" in args.panels:
        print(f"Panel b ({args.center} centres A / B / C)...")
        _, pdf = fig_centers_abc(phases, reward_pos, stats["neuron_ids"],
                                 panels_dir / "panel_b",
                                 size_in=LAYOUT.slot_size_in("b"),
                                 center=args.center)
        panel_pdfs["b"] = pdf

    if "c" in args.panels:
        print("Panel c (Δd violins A→B and C:pre→post frozen replay)...")
        _, pdf = fig_network_shifts(args.cohort, panels_dir / "panel_c",
                                    size_in=LAYOUT.slot_size_in("c"))
        panel_pdfs["c"] = pdf

    if "d" in args.panels:
        print("Panel d (matched plasticity control)...")
        _, pdf = fig_plasticity_control(
            panels_dir / "panel_d", size_in=LAYOUT.slot_size_in("d"))
        panel_pdfs["d"] = pdf

    if not args.no_assemble and panel_pdfs:
        print("Assembling vector composite PDF...")
        assemble_figure(panel_pdfs, figure_stem)

    print("\nDone.")
    print(f"  Figures: {figures_dir}")
    print(f"  Run dir: {run_dir}")


def parse_args():
    """Command-line arguments: run directory, reward position, output paths, panel subset."""
    p = argparse.ArgumentParser(description="Generate Figure 3 (reward-prediction head).")
    p.add_argument("--run_dir", default=str(DEFAULT_RUN))
    p.add_argument("--output_dir", default=str(FIGURES_DIR))
    p.add_argument("--reward_pos", type=float, nargs=2, default=list(DEFAULT_REWARD))
    p.add_argument("--n_neurons", type=int, default=8)
    p.add_argument("--neuron_ids", type=int, nargs="*", default=None)
    p.add_argument("--panels", default="a,b,c,d")
    p.add_argument("--skaggs_threshold", type=float, default=0.05)
    p.add_argument("--center", choices=("com", "peak"), default="com",
                   help="field centre for shift, distance and scatter panels "
                        "(com = centre of mass, the manuscript default; "
                        "peak writes figure3_peak_raw)")
    p.add_argument("--cohort", choices=("nh", "hd"), default="hd",
                   help="across-network panel c: without (nh) or with (hd) proximity teaching")
    p.add_argument("--no_show", action="store_true")
    p.add_argument("--no_assemble", action="store_true")
    add_plot_only(p)
    p.add_argument("--assemble_only", action="store_true",
                   help="recompose from the existing panel PDFs, skipping the "
                        "data load -- layout iteration in seconds, not minutes")
    args = p.parse_args()
    args.panels = {x.strip().lower() for x in args.panels.split(",") if x.strip()}
    return args


if __name__ == "__main__":
    generate_all(parse_args())
