#!/usr/bin/env python3
"""
Supplementary Figure 3 — plasticity sweep map health and distance distributions.

    python "code_analysis/figure 3/generate_figure3_supp.py" --no_show

Panels
  a  Mean distance to reward vs LR (COM and peak)
  b  # active place cells vs LR
  c  Mean Skaggs information vs LR
  d  Violins of COM distance-to-reward (A vs B) per LR + means
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis import (
    collect_sweep_metrics,
    fig_supp_distance_violins,
    fig_supp_mean_distance,
    fig_supp_n_active,
    fig_supp_skaggs,
    rows_to_jsonable,
)
from generate_figure3 import assemble_figure
from utils import (
    DEFAULT_REWARD,
    DEFAULT_SWEEP_ROOT,
    SUPP_FIGURES_DIR,
    SUPP_LAYOUT,
    SUPP_PANEL_ORDER,
    apply_nature_style,
    ensure_dir,
)

apply_nature_style()


def generate_all(args):
    sweep_root = Path(args.sweep_root).resolve()
    panels_dir = Path(args.output_dir).resolve() / "panels"
    figures_dir = Path(args.output_dir).resolve()
    ensure_dir(panels_dir)

    reward_pos = np.array(args.reward_pos, dtype=float)
    print(f"Collecting plasticity sweep from {sweep_root} ...")
    rows = collect_sweep_metrics(
        sweep_root, reward_pos, skaggs_threshold=args.skaggs_threshold,
    )
    summary_path = panels_dir / "plasticity_sweep_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(rows_to_jsonable(rows), f, indent=2)
    print(f"  Saved: {summary_path}")

    panel_pdfs: dict[str, Path] = {}

    if "a" in args.panels:
        print("Supp panel a (mean distance to reward)...")
        _, pdf = fig_supp_mean_distance(rows, panels_dir / "panel_a")
        panel_pdfs["a"] = pdf

    if "b" in args.panels:
        print("Supp panel b (# active place cells)...")
        _, pdf = fig_supp_n_active(rows, panels_dir / "panel_b")
        panel_pdfs["b"] = pdf

    if "c" in args.panels:
        print("Supp panel c (mean Skaggs)...")
        _, pdf = fig_supp_skaggs(rows, panels_dir / "panel_c")
        panel_pdfs["c"] = pdf

    if "d" in args.panels:
        print("Supp panel d (distance violins A vs B)...")
        _, pdf = fig_supp_distance_violins(rows, panels_dir / "panel_d")
        panel_pdfs["d"] = pdf

    if not args.no_assemble and panel_pdfs:
        print("Assembling supplementary composite...")
        assemble_figure(panel_pdfs, figures_dir / "figure3_supp_raw", SUPP_LAYOUT)

    print("\nDone.")
    print(f"  Figures: {figures_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Generate Supplementary Figure 3.")
    p.add_argument("--sweep_root", default=str(DEFAULT_SWEEP_ROOT))
    p.add_argument("--output_dir", default=str(SUPP_FIGURES_DIR))
    p.add_argument("--reward_pos", type=float, nargs=2, default=list(DEFAULT_REWARD))
    p.add_argument("--panels", default="a,b,c,d")
    p.add_argument("--skaggs_threshold", type=float, default=0.05)
    p.add_argument("--no_show", action="store_true")
    p.add_argument("--no_assemble", action="store_true")
    args = p.parse_args()
    args.panels = {x.strip().lower() for x in args.panels.split(",") if x.strip()}
    return args


if __name__ == "__main__":
    generate_all(parse_args())
