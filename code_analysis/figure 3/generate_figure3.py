#!/usr/bin/env python3
"""
Figure 3 — reward-prediction head (Paper Figure 3).

Frozen-replay control with reward-prediction head ON.
Same four-panel layout as Figure 2, but now the head causes genuine
representational remapping toward the reward.

    python "code_analysis/figure 3/generate_figure3.py" --no_show
    python "code_analysis/figure 3/generate_figure3.py" \
        --run_dir Repro/multi_seed_head/head_seed0 \
        --reward_pos 0.3 0.3 --no_show

Panels
  a  Frozen-replay remapping grid (same cells as Fig. 1d)
  b  Field COM centres for Phases A, B, C
  c  Violin plots of COM Δd for A→B (policy) and C:pre→post (frozen replay)
  d  Distance-to-reward histograms for A–B (policy) and C:pre→post (frozen replay)
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
    fig_delta_violins,
    fig_distance_histograms,
    fig_replay_remap_grid,
    fig1_panel_d_neuron_ids,
    load_phases,
    paired_distance_stats,
    shared_place_cell_ids,
)
from utils import (
    DEFAULT_REWARD,
    DEFAULT_RUN,
    FIGURE_LAYOUT,
    FIGURES_DIR,
    GRID_COLS,
    GRID_GAP,
    LABEL_SIZE,
    M,
    MM_TO_PT,
    PAGE_MARGIN,
    PAGE_W,
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


def _panel_slot_geometry(panel_sizes: dict[str, tuple[float, float]]):
    col_w = (PAGE_W - 2 * PAGE_MARGIN - (GRID_COLS - 1) * GRID_GAP) / GRID_COLS
    n_rows = 1 + max(r for _, r, _, _ in FIGURE_LAYOUT)
    # Per-row heights: each row is only as tall as its own tallest fitted panel
    # (previously every row inherited the global max, wasting vertical space).
    row_fitted: dict[int, list[float]] = {r: [] for r in range(n_rows)}
    for key, row, col, colspan in FIGURE_LAYOUT:
        if key not in panel_sizes:
            continue
        iw, ih = panel_sizes[key]
        slot_w = colspan * col_w + (colspan - 1) * GRID_GAP
        row_fitted[row].append(ih * (slot_w / max(iw, 1e-6)))
    default_h = 56.0 * MM_TO_PT
    row_heights = [float(np.max(row_fitted[r])) if row_fitted[r] else default_h
                   for r in range(n_rows)]
    page_h = sum(row_heights) + (n_rows - 1) * ROW_GAP + 2 * PAGE_MARGIN

    placements = []
    for key, row, col, colspan in FIGURE_LAYOUT:
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
    return page_h, placements


def _write_letter_overlay_pdf(placements, page_h: float, out_path: Path) -> Path:
    ensure_dir(out_path.parent)
    fig = plt.figure(figsize=(PAGE_W / 72.0, page_h / 72.0))
    fig.patch.set_facecolor("none")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, PAGE_W)
    ax.set_ylim(0, page_h)
    ax.axis("off")
    ax.set_facecolor("none")
    for p in placements:
        ax.text(
            p["left"] + 2.0, p["bottom"] + p["slot_h"] - 2.0, p["key"],
            fontsize=LABEL_SIZE, fontweight="bold", va="top", ha="left", color="black",
        )
    fig.savefig(out_path, format="pdf", transparent=True, facecolor="none", edgecolor="none",
                bbox_inches=None, pad_inches=0)
    plt.close(fig)
    return out_path


def assemble_figure(panel_pdfs: dict[str, Path], stem, save_dpi: int = 300):
    stem = Path(stem).with_suffix("")
    ensure_dir(stem.parent)
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")

    panel_sizes = {}
    for key, pdf in panel_pdfs.items():
        box = PdfReader(str(pdf)).pages[0].mediabox
        panel_sizes[key] = (float(box.width), float(box.height))

    page_h, placements = _panel_slot_geometry(panel_sizes)
    writer = PdfWriter()
    page = writer.add_blank_page(width=PAGE_W, height=page_h)
    for p in placements:
        src = PdfReader(str(panel_pdfs[p["key"]])).pages[0]
        page.merge_transformed_page(
            src,
            Transformation().scale(p["scale"]).translate(p["img_left"], p["img_bottom"]),
            over=True,
        )

    overlay_path = stem.parent / "_figure3_letters_overlay.pdf"
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


def _jsonable_stats(stats: dict) -> dict:
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


def generate_all(args):
    run_dir = Path(args.run_dir).resolve()
    data_dir = run_dir / "data"
    panels_dir = Path(args.output_dir).resolve() / "panels"
    figures_dir = Path(args.output_dir).resolve()
    ensure_dir(panels_dir)

    reward_pos = np.array(args.reward_pos, dtype=float)
    reward_bin = (int(reward_pos[0] * M), int(reward_pos[1] * M))
    panel_pdfs: dict[str, Path] = {}

    print("Loading phases A / B / C ...")
    phases = load_phases(data_dir)

    if "a" in args.panels:
        print("Panel a (frozen-replay remapping — WITH head, fields MOVE)...")
        if args.neuron_ids:
            neuron_ids = [int(x) for x in args.neuron_ids]
        else:
            neuron_ids = fig1_panel_d_neuron_ids(phases, n=args.n_neurons, seed=args.seed)
        print(f"  Neuron ids: {neuron_ids}")
        _, pdf = fig_replay_remap_grid(
            phases["C_pre"]["rm"], phases["C_post"]["rm"], neuron_ids, reward_bin,
            stem=panels_dir / "panel_a",
            cmap="jet",
            title="Frozen replay (Phase C): reward-prediction head ON — fields shift toward reward",
        )
        panel_pdfs["a"] = pdf

    need_stats = args.panels & {"b", "c", "d"}
    stats = None
    if need_stats:
        ids = shared_place_cell_ids(phases, skaggs_threshold=args.skaggs_threshold)
        stats = paired_distance_stats(phases, reward_pos, ids)
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

    if "b" in args.panels:
        print("Panel b (COM centres A / B / C)...")
        _, pdf = fig_centers_abc(phases, reward_pos, stats["neuron_ids"], panels_dir / "panel_b")
        panel_pdfs["b"] = pdf

    if "c" in args.panels:
        print("Panel c (Δd violins A→B and C:pre→post frozen replay)...")
        _, pdf = fig_delta_violins(stats, panels_dir / "panel_c")
        panel_pdfs["c"] = pdf

    if "d" in args.panels:
        print("Panel d (distance histograms A–B and C:pre→post frozen replay)...")
        _, pdf = fig_distance_histograms(stats, panels_dir / "panel_d")
        panel_pdfs["d"] = pdf

    if not args.no_assemble and panel_pdfs:
        print("Assembling vector composite PDF...")
        assemble_figure(panel_pdfs, figures_dir / "figure3_raw")

    print("\nDone.")
    print(f"  Figures: {figures_dir}")
    print(f"  Run dir: {run_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Generate Figure 3 (reward-prediction head).")
    p.add_argument("--run_dir", default=str(DEFAULT_RUN))
    p.add_argument("--output_dir", default=str(FIGURES_DIR))
    p.add_argument("--reward_pos", type=float, nargs=2, default=list(DEFAULT_REWARD))
    p.add_argument("--n_neurons", type=int, default=8)
    p.add_argument("--neuron_ids", type=int, nargs="*", default=None)
    p.add_argument("--panels", default="a,b,c,d")
    p.add_argument("--skaggs_threshold", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_show", action="store_true")
    p.add_argument("--no_assemble", action="store_true")
    args = p.parse_args()
    args.panels = {x.strip().lower() for x in args.panels.split(",") if x.strip()}
    return args


if __name__ == "__main__":
    generate_all(parse_args())
