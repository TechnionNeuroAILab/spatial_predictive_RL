#!/usr/bin/env python3
"""
Figure 3 — high plasticity (1e-4) remapping + plasticity (LR) sweep.

Uses existing no-head runs under New_Results_v3/dose_sweep_lr_nohead.

    python "code_analysis/figure 3/generate_figure3.py" --no_show

Panels
  a  Rate-map remapping examples at lr=1e-4 (stronger than Fig. 1d)
  b  COM centres A/B/C at lr=1e-4 (as Fig. 2b)
  c  Boxplots of per-cell Δd vs RL learning rate (plasticity enables remapping)
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
    collect_sweep_metrics,
    fig_centers_abc,
    fig_plasticity_boxplots,
    fig_replay_remap_grid,
    load_phases,
    pick_strong_remap_neurons,
    rows_to_jsonable,
    shared_place_cell_ids,
)
from utils import (
    DEFAULT_REWARD,
    DEFAULT_RUN_1E4,
    DEFAULT_SWEEP_ROOT,
    FIGURE_LAYOUT,
    FIGURES_DIR,
    GRID_COLS,
    GRID_GAP,
    LABEL_SIZE,
    M,
    MM_TO_PT,
    PAGE_MARGIN,
    PAGE_W,
    PANEL_ORDER,
    ROW_GAP,
    apply_nature_style,
    ensure_dir,
)

apply_nature_style()


def _pdf_to_png(pdf_path: Path, png_path: Path, dpi: int = 300) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out_prefix = Path(tmp) / "page"
        subprocess.run(
            ["pdftoppm", "-png", "-r", str(dpi), "-singlefile", str(pdf_path), str(out_prefix)],
            check=True,
        )
        copy2(out_prefix.with_suffix(".png"), png_path)


def assemble_figure(panel_pdfs: dict[str, Path], stem, layout, save_dpi: int = 300):
    stem = Path(stem).with_suffix("")
    ensure_dir(stem.parent)
    png_path, pdf_path = stem.with_suffix(".png"), stem.with_suffix(".pdf")

    panel_sizes = {}
    for key, pdf in panel_pdfs.items():
        box = PdfReader(str(pdf)).pages[0].mediabox
        panel_sizes[key] = (float(box.width), float(box.height))

    col_w = (PAGE_W - 2 * PAGE_MARGIN - (GRID_COLS - 1) * GRID_GAP) / GRID_COLS
    fitted_h = []
    for key, row, col, colspan in layout:
        if key not in panel_sizes:
            continue
        iw, ih = panel_sizes[key]
        slot_w = colspan * col_w + (colspan - 1) * GRID_GAP
        fitted_h.append(ih * (slot_w / max(iw, 1e-6)))
    row_h = float(np.max(fitted_h)) if fitted_h else 56.0 * MM_TO_PT
    n_rows = 1 + max(r for _, r, _, _ in layout)
    page_h = n_rows * row_h + (n_rows - 1) * ROW_GAP + 2 * PAGE_MARGIN
    row_heights = [row_h] * n_rows

    placements = []
    for key, row, col, colspan in layout:
        if key not in panel_sizes:
            continue
        iw, ih = panel_sizes[key]
        slot_w = colspan * col_w + (colspan - 1) * GRID_GAP
        slot_h = row_heights[row]
        left = PAGE_MARGIN + col * (col_w + GRID_GAP)
        below = sum(row_heights[r] + ROW_GAP for r in range(row + 1, n_rows))
        bottom = PAGE_MARGIN + below
        scale = min(slot_w / iw, slot_h / ih)
        sw, sh = iw * scale, ih * scale
        placements.append({
            "key": key,
            "left": left,
            "bottom": bottom,
            "slot_w": slot_w,
            "slot_h": slot_h,
            "img_left": left + (slot_w - sw) / 2.0,
            "img_bottom": bottom + (slot_h - sh) / 2.0,
            "scale": scale,
        })

    writer = PdfWriter()
    page = writer.add_blank_page(width=PAGE_W, height=page_h)
    for p in placements:
        src = PdfReader(str(panel_pdfs[p["key"]])).pages[0]
        page.merge_transformed_page(
            src,
            Transformation().scale(p["scale"]).translate(p["img_left"], p["img_bottom"]),
            over=True,
        )

    overlay = stem.parent / "_letters_overlay.pdf"
    fig = plt.figure(figsize=(PAGE_W / 72.0, page_h / 72.0))
    fig.patch.set_facecolor("none")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, PAGE_W)
    ax.set_ylim(0, page_h)
    ax.axis("off")
    ax.set_facecolor("none")
    for p in placements:
        ax.text(p["left"] + 2.0, p["bottom"] + p["slot_h"] - 2.0, p["key"],
                fontsize=LABEL_SIZE, fontweight="bold", va="top", ha="left")
    fig.savefig(overlay, format="pdf", transparent=True, facecolor="none",
                bbox_inches=None, pad_inches=0)
    plt.close(fig)
    page.merge_page(PdfReader(str(overlay)).pages[0], over=True)
    with open(pdf_path, "wb") as f:
        writer.write(f)
    try:
        overlay.unlink(missing_ok=True)
    except OSError:
        pass

    _pdf_to_png(pdf_path, png_path, dpi=save_dpi)
    print(f"  Saved: {pdf_path}")
    print(f"  Saved: {png_path}")
    return png_path, pdf_path


def generate_all(args):
    run_1e4 = Path(args.run_1e4).resolve()
    sweep_root = Path(args.sweep_root).resolve()
    panels_dir = Path(args.output_dir).resolve() / "panels"
    figures_dir = Path(args.output_dir).resolve()
    ensure_dir(panels_dir)

    reward_pos = np.array(args.reward_pos, dtype=float)
    reward_bin = (int(reward_pos[0] * M), int(reward_pos[1] * M))
    panel_pdfs: dict[str, Path] = {}

    print(f"Loading high-plasticity run: {run_1e4}")
    phases = load_phases(run_1e4 / "data")

    if "a" in args.panels:
        print("Panel a (strong remapping examples at lr=1e-4)...")
        if args.neuron_ids:
            neuron_ids = [int(x) for x in args.neuron_ids]
        else:
            neuron_ids = pick_strong_remap_neurons(
                phases, reward_pos, n=args.n_neurons, seed=args.seed,
            )
        print(f"  Neurons: {neuron_ids}")
        # Show prediction vs goal policy (same contrast as Fig. 1d), not frozen replay.
        _, pdf = fig_replay_remap_grid(
            phases["A"]["rm"], phases["B"]["rm"], neuron_ids, reward_bin,
            stem=panels_dir / "panel_a",
            ylabel_top="A: prediction",
            ylabel_bottom="B: goal policy",
            cmap="jet",
            title="High plasticity (lr=1e−4): prediction vs goal-policy remapping",
        )
        panel_pdfs["a"] = pdf

    if "b" in args.panels:
        print("Panel b (COM centres A/B/C at lr=1e-4)...")
        ids = shared_place_cell_ids(phases, skaggs_threshold=args.skaggs_threshold)
        _, pdf = fig_centers_abc(phases, reward_pos, ids, panels_dir / "panel_b")
        panel_pdfs["b"] = pdf

    rows = None
    if "c" in args.panels or args.save_sweep_json:
        print(f"Collecting plasticity sweep from {sweep_root} ...")
        rows = collect_sweep_metrics(
            sweep_root, reward_pos, skaggs_threshold=args.skaggs_threshold,
        )
        summary_path = panels_dir / "plasticity_sweep_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(rows_to_jsonable(rows), f, indent=2)
        print(f"  Saved: {summary_path}")

    if "c" in args.panels:
        print("Panel c (plasticity boxplots)...")
        _, pdf = fig_plasticity_boxplots(rows, panels_dir / "panel_c")
        panel_pdfs["c"] = pdf

    if not args.no_assemble and panel_pdfs:
        print("Assembling vector composite PDF...")
        assemble_figure(panel_pdfs, figures_dir / "figure3_raw", FIGURE_LAYOUT)

    print("\nDone.")
    print(f"  Figures: {figures_dir}")
    print(f"  Demo run: {run_1e4}")
    print(f"  Sweep: {sweep_root}")
    return rows


def parse_args():
    p = argparse.ArgumentParser(description="Generate Figure 3 (plasticity / remapping).")
    p.add_argument("--run_1e4", default=str(DEFAULT_RUN_1E4))
    p.add_argument("--sweep_root", default=str(DEFAULT_SWEEP_ROOT))
    p.add_argument("--output_dir", default=str(FIGURES_DIR))
    p.add_argument("--reward_pos", type=float, nargs=2, default=list(DEFAULT_REWARD))
    p.add_argument("--n_neurons", type=int, default=8)
    p.add_argument("--neuron_ids", type=int, nargs="*", default=None)
    p.add_argument("--panels", default="a,b,c")
    p.add_argument("--skaggs_threshold", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_sweep_json", action="store_true")
    p.add_argument("--no_show", action="store_true")
    p.add_argument("--no_assemble", action="store_true")
    args = p.parse_args()
    args.panels = {x.strip().lower() for x in args.panels.split(",") if x.strip()}
    return args


if __name__ == "__main__":
    generate_all(parse_args())
