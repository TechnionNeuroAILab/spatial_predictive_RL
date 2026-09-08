#!/usr/bin/env python3
"""
Self-contained Figure 1 panel generator.

All outputs (PNG + PDF) go to Code/figure 1/figures/ by default.

    python "Code/figure 1/generate_figure1.py" --no_show
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter

from panel_b import render_panel_b
from panel_d import render_panel_d
from rate_maps import (
    analyse_peak_com_distances,
    build_rate_maps,
    compute_skaggs,
    fig_100cells,
    fig_dist_histogram,
    fig_pred_vs_rl_across_cycles,
    fig_pred_vs_rl_cycle1_many,
    load_cycle_rate_maps,
    select_random_neurons,
    smooth_and_norm,
)
from utils import (
    DEFAULT_ARCHITECTURE,
    DEFAULT_ARCHITECTURE_PNG,
    DEFAULT_REWARD,
    DEFAULT_RUN,
    FIGURE_LAYOUT,
    FIGURES_DIR,
    GRID_COLS,
    GRID_GAP,
    HIDDEN_DIM,
    LABEL_SIZE,
    M,
    MM_TO_PT,
    PAGE_MARGIN,
    PAGE_W,
    PANEL_TITLES,
    PANELS_DIR,
    ROW_GAP,
    TITLE_H,
    TITLE_LINE_H,
    TITLE_SIZE,
    apply_nature_style,
    ensure_dir,
    png_to_pdf,
    save_figure,
)

apply_nature_style()

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]


def placeholder_panel(stem, label: str):
    fig, ax = plt.subplots(figsize=(1.6, 1.6))
    ax.axis("off")
    ax.text(0.5, 0.55, label, ha="center", va="center", fontsize=7, transform=ax.transAxes)
    ax.text(0.5, 0.35, "(replace manually)", ha="center", va="center", fontsize=6, color="0.45",
            transform=ax.transAxes)
    ax.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, linestyle="--", color="0.6", lw=0.5))
    return save_figure(fig, stem)


def architecture_panel(arch_path: Path, stem):
    """Install architecture panel as vector PDF (preferred) plus optional PNG preview."""
    from shutil import copy2

    ensure_dir(stem.parent)
    stem = Path(stem).with_suffix("")
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    arch_path = Path(arch_path)

    if arch_path.suffix.lower() == ".pdf" and arch_path.exists():
        copy2(arch_path, pdf_path)
        # PNG preview for the editor composite (optional).
        preview_src = DEFAULT_ARCHITECTURE_PNG
        if preview_src.exists():
            copy2(preview_src, png_path)
        else:
            _pdf_to_png(pdf_path, png_path, dpi=150)
    elif arch_path.exists():
        copy2(arch_path, png_path)
        sibling_pdf = arch_path.with_suffix(".pdf")
        vector_default = DEFAULT_ARCHITECTURE
        if vector_default.exists() and vector_default.suffix.lower() == ".pdf":
            copy2(vector_default, pdf_path)
        elif sibling_pdf.exists():
            copy2(sibling_pdf, pdf_path)
        else:
            png_to_pdf(png_path, pdf_path)
    else:
        raise FileNotFoundError(f"Architecture asset not found: {arch_path}")

    print(f"  Saved: {pdf_path}")
    if png_path.exists():
        print(f"  Saved: {png_path}")
    return png_path if png_path.exists() else pdf_path, pdf_path


def _pdf_to_png(pdf_path: Path, png_path: Path, dpi: int = 300) -> None:
    """Rasterize first PDF page with pdftoppm (Poppler)."""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out_prefix = Path(tmp) / "page"
        subprocess.run(
            ["pdftoppm", "-png", "-r", str(dpi), "-singlefile", str(pdf_path), str(out_prefix)],
            check=True,
        )
        generated = out_prefix.with_suffix(".png")
        ensure_dir(png_path.parent)
        from shutil import copy2
        copy2(generated, png_path)


def _panel_slot_geometry(panel_sizes: dict[str, tuple[float, float]]):
    """Per-row heights from actual panel aspect ratios; contain-fit, top-aligned.

    Each row is only as tall as its tallest panel once fitted to its slot
    width, plus a small ``TITLE_H`` strip above the row for the panel
    letter/title.  The page height is the sum of the rows (no fixed page
    height, no dead centering space).
    """
    col_w = (PAGE_W - 2 * PAGE_MARGIN - (GRID_COLS - 1) * GRID_GAP) / GRID_COLS
    n_rows = max(row for _, row, _, _ in FIGURE_LAYOUT) + 1
    row_heights = [0.0] * n_rows
    title_strips = [TITLE_H] * n_rows
    for key, row, col, colspan in FIGURE_LAYOUT:
        if key not in panel_sizes:
            continue
        iw, ih = panel_sizes[key]
        slot_w = colspan * col_w + (colspan - 1) * GRID_GAP
        fitted = ih * (slot_w / max(iw, 1e-6))
        row_heights[row] = max(row_heights[row], fitted)
        n_lines = PANEL_TITLES.get(key, "").count("\n") + 1
        title_strips[row] = max(title_strips[row], TITLE_H + (n_lines - 1) * TITLE_LINE_H)
    row_heights = [h if h > 0 else 56.0 * MM_TO_PT for h in row_heights]
    page_h = (
        sum(row_heights)
        + sum(title_strips)
        + (n_rows - 1) * ROW_GAP
        + 2 * PAGE_MARGIN
    )

    placements = []
    for key, row, col, colspan in FIGURE_LAYOUT:
        if key not in panel_sizes:
            continue
        iw, ih = panel_sizes[key]
        slot_w = colspan * col_w + (colspan - 1) * GRID_GAP
        slot_h = row_heights[row]
        left = PAGE_MARGIN + col * (col_w + GRID_GAP)
        below = sum(
            row_heights[r] + title_strips[r] + ROW_GAP for r in range(row + 1, n_rows)
        )
        bottom = PAGE_MARGIN + below
        # Contain-fit: preserve aspect ratio (never stretch).
        scale = min(slot_w / iw, slot_h / ih)
        sw, sh = iw * scale, ih * scale
        img_left = left + (slot_w - sw) / 2.0
        # Top-align within the row so panels sit just under their titles.
        img_bottom = bottom + (slot_h - sh)
        # Shared first-line baseline for the letter/title strip of this row
        # (multi-line titles extend downward toward the panel top).
        title_baseline = bottom + slot_h + 2.5 + (title_strips[row] - TITLE_H)
        placements.append({
            "title_baseline": title_baseline,
            "key": key,
            "left": left,
            "bottom": bottom,
            "slot_w": slot_w,
            "slot_h": slot_h,
            "img_left": img_left,
            "img_bottom": img_bottom,
            "scale": scale,
            "sw": sw,
            "sh": sh,
        })
    return page_h, placements


def _write_letter_overlay_pdf(placements, page_h: float, out_path: Path) -> Path:
    """Vector PDF with panel letters and short titles (no opaque page fill)."""
    ensure_dir(out_path.parent)
    fig = plt.figure(figsize=(PAGE_W / 72.0, page_h / 72.0))
    fig.patch.set_facecolor("none")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, PAGE_W)
    ax.set_ylim(0, page_h)
    ax.axis("off")
    ax.set_facecolor("none")
    for p in placements:
        # Letter + title share the row's first-line baseline above the panels.
        baseline_y = p["title_baseline"]
        ax.text(
            p["left"],
            baseline_y,
            p["key"],
            fontsize=LABEL_SIZE,
            fontweight="bold",
            va="baseline",
            ha="left",
            color="black",
        )
        title = PANEL_TITLES.get(p["key"])
        if title:
            ax.text(
                p["left"] + 9.0,
                baseline_y,
                title,
                fontsize=TITLE_SIZE,
                va="baseline",
                ha="left",
                color="black",
            )
    # Avoid bbox_inches='tight' so page size matches the composite.
    fig.savefig(
        out_path, format="pdf", transparent=True, facecolor="none", edgecolor="none",
        bbox_inches=None, pad_inches=0,
    )
    plt.close(fig)
    return out_path


def assemble_figure(panel_paths: dict[str, Path], stem, save_dpi: int = 300):
    """Assemble a vector PDF from panel PDFs (aspect preserved, equal row heights).

    Also writes a PNG preview rasterized from that PDF (not used as an input).
    ``panel_paths`` may map to ``.pdf`` (preferred) or ``.png`` (legacy).
    """
    from pypdf import PdfReader, PdfWriter, Transformation

    stem = Path(stem).with_suffix("")
    ensure_dir(stem.parent)
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")

    panel_pdfs: dict[str, Path] = {}
    for key, path in panel_paths.items():
        path = Path(path)
        pdf_candidate = path if path.suffix.lower() == ".pdf" else path.with_suffix(".pdf")
        if not pdf_candidate.exists():
            raise FileNotFoundError(f"Missing vector panel PDF for '{key}': {pdf_candidate}")
        panel_pdfs[key] = pdf_candidate

    panel_sizes = {}
    for key, pdf in panel_pdfs.items():
        box = PdfReader(str(pdf)).pages[0].mediabox
        panel_sizes[key] = (float(box.width), float(box.height))

    page_h, placements = _panel_slot_geometry(panel_sizes)

    writer = PdfWriter()
    page = writer.add_blank_page(width=PAGE_W, height=page_h)
    for p in placements:
        src = PdfReader(str(panel_pdfs[p["key"]])).pages[0]
        # pypdf applies translate then scale if chained left-to-right carefully:
        # Transformation().scale(s).translate(tx, ty) scales about origin then moves.
        page.merge_transformed_page(
            src,
            Transformation().scale(p["scale"]).translate(p["img_left"], p["img_bottom"]),
            over=True,
        )

    overlay_path = stem.parent / "_figure1_letters_overlay.pdf"
    _write_letter_overlay_pdf(placements, page_h, overlay_path)
    overlay_page = PdfReader(str(overlay_path)).pages[0]
    page.merge_page(overlay_page, over=True)

    with open(pdf_path, "wb") as f:
        writer.write(f)
    try:
        overlay_path.unlink(missing_ok=True)
    except OSError:
        pass

    # PNG preview only (raster of the vector composite).
    _pdf_to_png(pdf_path, png_path, dpi=save_dpi)
    print(f"  Saved: {pdf_path}")
    print(f"  Saved: {png_path}")
    return png_path, pdf_path


def render_top_skaggs(df_beh, df_act, reward_bin, stem, m=32, hidden_dim=512, n_top=38, skaggs_threshold=0.05):
    n_steps = len(df_beh)
    if len(df_act) == n_steps * hidden_dim:
        act_mat = df_act["activity"].values.reshape(n_steps, hidden_dim)
        beh_sorted = df_beh.sort_values("timestep")
    else:
        act_pivot = df_act.pivot(index="timestep", columns="neuron_id", values="activity").fillna(0)
        beh_sorted = df_beh[df_beh["timestep"].isin(act_pivot.index)].sort_values("timestep")
        act_mat = act_pivot.reindex(beh_sorted["timestep"]).values

    x_bin = np.clip((beh_sorted["x"].values * m).astype(int), 0, m - 1)
    y_bin = np.clip((beh_sorted["y"].values * m).astype(int), 0, m - 1)
    gc = x_bin * m + y_bin
    n_bins = m * m
    bin_sum = np.zeros((n_bins, hidden_dim))
    np.add.at(bin_sum, gc, act_mat)
    bin_count = np.bincount(gc, minlength=n_bins).astype(float)
    mean_rate_per_bin = np.divide(bin_sum, bin_count[:, None], out=np.zeros_like(bin_sum),
                                   where=bin_count[:, None] > 0)
    mean_rate = act_mat.mean(axis=0)
    ratio = np.where(mean_rate[None, :] > 1e-10, mean_rate_per_bin / mean_rate[None, :], 0.0)
    log_ratio = np.where(ratio > 0, np.log2(np.maximum(ratio, 1e-300)), 0.0)
    occ = bin_count / len(beh_sorted)
    skaggs = (occ[:, None] * ratio * log_ratio).sum(axis=0)
    df_skaggs = pd.DataFrame({"skaggs_score": skaggs})
    n_pass = int((df_skaggs["skaggs_score"] >= skaggs_threshold).sum())
    top_idx = df_skaggs.nlargest(n_top, "skaggs_score").index.values

    rate_maps = np.zeros((n_top, m, m))
    for b in range(n_bins):
        mask = gc == b
        if mask.any():
            rate_maps[:, b // m, b % m] = act_mat[:, top_idx][mask].mean(axis=0)

    n_cols = 8
    n_rows = int(np.ceil(n_top / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.8, n_rows * 1.8))
    axes_flat = np.array(axes).flatten()
    rx, ry = reward_bin
    for i, nid in enumerate(top_idx):
        ax = axes_flat[i]
        rm = gaussian_filter(rate_maps[i], sigma=1.5)
        if rm.max() > 0:
            rm = (rm - rm.min()) / (rm.max() - rm.min())
        ax.imshow(rm, origin="lower", cmap="jet", vmin=0, vmax=1)
        ax.plot(rx, ry, "w*", markersize=4)
        ax.set_title(f"{nid}  {df_skaggs.loc[nid, 'skaggs_score']:.3f}", fontsize=6)
        ax.axis("off")
    for i in range(n_top, len(axes_flat)):
        axes_flat[i].axis("off")
    fig.suptitle(f"Top {n_top} by Skaggs — before | {n_pass}/{hidden_dim} place cells", fontsize=9)
    plt.tight_layout()
    return save_figure(fig, stem)


def stitch_pngs_horiz(png_paths: list[Path], stem):
    images = [Image.open(p).convert("RGB") for p in png_paths]
    gap = 8
    h = max(im.height for im in images)
    w = sum(im.width for im in images) + gap * (len(images) - 1)
    canvas = Image.new("RGB", (w, h), "white")
    x = 0
    for im in images:
        y = (h - im.height) // 2
        canvas.paste(im, (x, y))
        x += im.width + gap
    stem = Path(stem).with_suffix("")
    ensure_dir(stem.parent)
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    canvas.save(png_path)
    canvas.save(pdf_path, "PDF", resolution=150.0)
    print(f"  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")
    return png_path, pdf_path


def generate_all(args):
    run_dir = Path(args.run_dir).resolve()
    data_dir = run_dir / "data"
    panels_dir = Path(args.output_dir).resolve() / "panels"
    figures_dir = Path(args.output_dir).resolve()
    ensure_dir(panels_dir)

    reward_pos = np.array(args.reward_pos, dtype=float)
    reward_bin = (int(reward_pos[0] * M), int(reward_pos[1] * M))
    panel_pdfs: dict[str, Path] = {}

    if args.assemble_only:
        print("Assembly-only mode: reusing existing panel PDFs...")
        for key in sorted(args.panels):
            pdf = panels_dir / f"panel_{key}.pdf"
            if not pdf.exists():
                raise FileNotFoundError(f"--assemble_only: missing panel PDF {pdf}")
            panel_pdfs[key] = pdf
        assemble_figure(panel_pdfs, figures_dir / "figure1_raw")
        print("\nDone.")
        print(f"  Figures: {figures_dir}")
        return

    if "a" in args.panels:
        print("Panel a (architecture cartoon)...")
        arch = Path(args.architecture).resolve()
        if not arch.exists():
            arch = DEFAULT_ARCHITECTURE.resolve()
        _, pdf = architecture_panel(arch, panels_dir / "panel_a")
        panel_pdfs["a"] = pdf

    if "b" in args.panels:
        print("Panel b (FoV / observation)...")
        _, pdf = render_panel_b(
            panels_dir / "panel_b_examples",
            panels_dir / "panel_b",
            compact_pose_idx=args.panel_b_pose,
            seed=args.seed,
        )
        panel_pdfs["b"] = pdf

    if "c" in args.panels:
        print("Panel c (PCA cognitive map, 3D)...")
        _, pdf = render_panel_d(data_dir, panels_dir / "panel_c", tuple(reward_pos))
        panel_pdfs["c"] = pdf

    if args.panels & {"d", "e", "f", "h", "legacy_f", "legacy_g"}:
        df_beh_rw = pd.read_pickle(data_dir / "df_behavior_before.pkl")
        df_act_rw = pd.read_pickle(data_dir / "df_activations_before.pkl")
        df_beh_gc = pd.read_pickle(data_dir / "df_behavior_goal_conditioned.pkl")
        df_act_gc = pd.read_pickle(data_dir / "df_activations_goal_conditioned.pkl")
        rm_rw, occ_rw = build_rate_maps(df_beh_rw, df_act_rw, M, HIDDEN_DIM)
        rm_gc, occ_gc = build_rate_maps(df_beh_gc, df_act_gc, M, HIDDEN_DIM)
        skaggs_rw = compute_skaggs(rm_rw, occ_rw)
        skaggs_gc = compute_skaggs(rm_gc, occ_gc)
        top_idx_100 = np.argsort(skaggs_gc)[::-1][:100]
        rw_cycles = load_cycle_rate_maps(str(data_dir), M, HIDDEN_DIM, probe_type="rw")
        goal_cycles = load_cycle_rate_maps(str(data_dir), M, HIDDEN_DIM, probe_type="goal")

        if "d" in args.panels:
            print("Panel d (random active cells, cycle 1 before/after RL, jet)...")
            rw_key = next(k for k in rw_cycles if "cycle_1" in k and "goal" not in k)
            goal_key = next(k for k in goal_cycles if "cycle_1" in k)
            rm_c1 = rw_cycles[rw_key][0]
            rm_c1g = goal_cycles[goal_key][0]
            if args.neuron_ids:
                neuron_ids = [int(x) for x in args.neuron_ids]
            else:
                neuron_ids = select_random_neurons(
                    rm_c1, rm_c1g, n=args.n_cycle1_neurons, seed=args.seed,
                )
            print(f"  Random active neurons (seed={args.seed}): {neuron_ids}")
            _, pdf = fig_pred_vs_rl_cycle1_many(
                rw_cycles, goal_cycles, neuron_ids, reward_bin,
                stem=panels_dir / "panel_d",
                ylabel_top="Before RL",
                ylabel_bottom="After RL",
                cmap="jet",
            )
            panel_pdfs["d"] = pdf

        if args.panels & {"e", "f"}:
            # Peak / COM: prediction RW (cycle_1) vs goal-driven policy (cycle_1_goal).
            # Same contrast as panel d — not frozen replay (that belongs in Figure 2).
            rw_key = next(k for k in rw_cycles if "cycle_1" in k and "goal" not in k)
            goal_key = next(k for k in goal_cycles if "cycle_1" in k)
            rm_pre, _, sk_pre = rw_cycles[rw_key]
            rm_post, _, sk_post = goal_cycles[goal_key]
            period_label = "cycle_1 (prediction) → cycle_1_goal (goal policy)"

            stats = analyse_peak_com_distances(
                rm_pre, sk_pre, rm_post, sk_post, reward_pos,
                skaggs_threshold=args.skaggs_threshold,
            )
            print(
                f"  Peak/COM analysis ({period_label}): n={stats['n']} place cells "
                f"(Skaggs ≥ {args.skaggs_threshold})"
            )
            print(
                f"    Peak: mean d {stats['peak_mean_before']:.3f} → {stats['peak_mean_after']:.3f} "
                f"(Δ={stats['peak_mean_delta']:+.3f}, {100 * stats['peak_frac_toward']:.0f}% toward)"
            )
            print(
                f"    COM:  mean d {stats['com_mean_before']:.3f} → {stats['com_mean_after']:.3f} "
                f"(Δ={stats['com_mean_delta']:+.3f}, {100 * stats['com_frac_toward']:.0f}% toward)"
            )
            stats_path = panels_dir / "peak_com_distances.npz"
            np.savez_compressed(
                stats_path,
                neuron_ids=stats["neuron_ids"],
                peak_before=stats["peak_before"],
                peak_after=stats["peak_after"],
                com_before=stats["com_before"],
                com_after=stats["com_after"],
                skaggs_threshold=stats["skaggs_threshold"],
                period_label=np.array(period_label),
            )
            print(f"  Saved: {stats_path}")

            if "e" in args.panels:
                print("Panel e (peak distance to reward)...")
                _, pdf = fig_dist_histogram(
                    stats["peak_before"], stats["peak_after"],
                    stem=panels_dir / "panel_e",
                    title="Field peak",
                    label_before="Prediction",
                    label_after="Goal policy",
                )
                panel_pdfs["e"] = pdf

            if "f" in args.panels:
                print("Panel f (COM distance to reward)...")
                _, pdf = fig_dist_histogram(
                    stats["com_before"], stats["com_after"],
                    stem=panels_dir / "panel_f",
                    title="Centre of mass",
                    label_before="Prediction",
                    label_after="Goal policy",
                )
                panel_pdfs["f"] = pdf

        if "h" in args.panels:
            print(f"Panel h (legacy): cycle-1 before/after for {args.n_cycle1_neurons} neurons...")
            neuron_ids = top_idx_100[: args.n_cycle1_neurons]
            _, pdf = fig_pred_vs_rl_cycle1_many(
                rw_cycles, goal_cycles, neuron_ids, reward_bin,
                stem=panels_dir / "panel_h",
                cmap="jet",
            )
            panel_pdfs["h"] = pdf

        if "legacy_f" in args.panels:
            print(f"Legacy (neuron {args.panel_neuron} across cycles)...")
            _, pdf = fig_pred_vs_rl_across_cycles(
                rw_cycles, goal_cycles, args.panel_neuron, reward_bin,
                stem=panels_dir / "panel_legacy_cycles",
            )
            panel_pdfs["legacy_f"] = pdf

        if "legacy_g" in args.panels:
            print("Legacy (100-cell goal map)...")
            _, pdf = fig_100cells(
                rm_gc, skaggs_gc, top_idx_100, reward_bin,
                title="Goal-directed cognitive map", stem=panels_dir / "panel_legacy_100",
            )
            panel_pdfs["legacy_g"] = pdf

    if not args.no_assemble and panel_pdfs:
        print("Assembling vector composite PDF...")
        assemble_figure(panel_pdfs, figures_dir / "figure1_raw")

    print("\nDone.")
    print(f"  Figures: {figures_dir}")
    print(f"  Run dir: {run_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Generate Figure 1 panels (self-contained).")
    p.add_argument("--run_dir", default=str(DEFAULT_RUN))
    p.add_argument("--output_dir", default=str(FIGURES_DIR))
    p.add_argument("--reward_pos", type=float, nargs=2, default=list(DEFAULT_REWARD))
    p.add_argument("--panel_neuron", type=int, default=425)
    p.add_argument("--n_cycle1_neurons", type=int, default=8,
                   help="Neurons in panel d (before/after RL remapping grid)")
    p.add_argument("--neuron_ids", type=int, nargs="*", default=None,
                   help="Optional explicit neuron ids for panel d")
    p.add_argument("--architecture", default=str(DEFAULT_ARCHITECTURE),
                   help="Architecture cartoon for panel a (vector PDF preferred)")
    p.add_argument("--panels", default="a,b,c,d,e,f")
    p.add_argument("--skaggs_threshold", type=float, default=0.05,
                   help="Place-cell gate for peak/COM panels (Skaggs in both before and after)")
    p.add_argument("--panel_b_pose", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_show", action="store_true")
    p.add_argument("--no_assemble", action="store_true")
    p.add_argument("--assemble_only", action="store_true",
                   help="Skip panel generation; assemble the composite from existing panel PDFs")
    args = p.parse_args()
    args.panels = {x.strip().lower() for x in args.panels.split(",") if x.strip()}
    return args


if __name__ == "__main__":
    generate_all(parse_args())
