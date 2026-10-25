#!/usr/bin/env python3
"""
Self-contained Figure 1 panel generator.

All outputs (PNG + PDF) go to code/figures/fig1/figures/ by default.

    python code/figures/fig1/generate_figure1.py --no_show
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pypdf import PdfReader
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
    RATEMAP,
    DEFAULT_ARCHITECTURE,
    DEFAULT_ARCHITECTURE_PNG,
    DEFAULT_REWARD,
    DEFAULT_RUN,
    FIGURES_DIR,
    HIDDEN_DIM,
    LAYOUT,
    M,
    MODEL_STEM,
    PAGE_W,
    PANEL_TITLES,
    PANELS_DIR,
    slot_width_in,
    apply_nature_style,
    ensure_dir,
    png_to_pdf,
    save_figure,
)
import figlayout
from figure_cache import add_plot_only, load_figure_data, require_keys, update_figure_data

apply_nature_style()

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]


def _example_stats(stats: dict, key: str) -> str:
    """Per-cell shift for the example network: mean delta, % toward, Wilcoxon P."""
    from pubstyle import fmt_p
    return (f"\u0394d = {stats[f'{key}_mean_delta']:+.3f}, "
            f"{100 * stats[f'{key}_frac_toward']:.0f}% toward\n"
            f"{fmt_p(stats.get(f'{key}_wilcoxon_p'))}, n = {stats['n']}")


def placeholder_panel(stem, label: str):
    """Draw a dashed 'replace manually' box, for panels that are hand-made assets."""
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


def assemble_figure(panel_paths: dict[str, Path], stem, save_dpi: int = 300):
    """Compose the panel PDFs onto the page defined by `utils.LAYOUT`.

    This used to be ~170 lines: a geometry solver that derived row heights from
    the measured panel PDFs, a contain-fit that scaled each panel by anything
    from 0.55 to 1.02, and a hand-rolled letter overlay. The scaling is what
    made a 7 pt label print at 3.7 pt, so it is gone -- `figlayout.compose`
    refuses to scale at all, and panels are rendered at `LAYOUT.slot_size_in`
    instead.
    """
    return figlayout.compose(LAYOUT, panel_paths, stem,
                             titles=PANEL_TITLES, dpi=save_dpi)


def model_figure(arch_path: Path, stem, save_dpi: int = 300):
    """Lay the architecture schematic out as a figure of its own.

    It is externally authored artwork (Inkscape, Arial/Tahoma) at a fixed
    aspect ratio, so unlike every other panel in the paper it cannot be
    re-rendered at a given size -- it is scaled to the text width instead, and
    `fit=` marks that as deliberate. At 654 pt native it scales to 0.773, which
    puts its type at 6.4-13.3 pt: legible, where the old 0.548 in-figure slot
    put its smallest labels at 4.7 pt.
    """
    arch_path = Path(arch_path)
    box = PdfReader(str(arch_path)).pages[0].mediabox
    iw, ih = float(box.width), float(box.height)
    layout = figlayout.Layout(
        [("model",)], row_h_in=[ih * (PAGE_W / iw) / 72.0],
        letter_strip_pt=0.0, caption_lines=8, name="fig_model")
    return figlayout.compose(layout, {"model": arch_path}, stem,
                             letters=False, fit={"model"}, dpi=save_dpi)


def render_top_skaggs(df_beh, df_act, reward_bin, stem, m=32, hidden_dim=512, n_top=38, skaggs_threshold=0.05):
    """Rate-map grid of the highest-Skaggs units, each titled with its score."""
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
        ax.imshow(rm, origin="lower", cmap=RATEMAP, vmin=0, vmax=1)
        ax.plot(rx, ry, "w*", markersize=4)
        ax.set_title(f"{nid}  {df_skaggs.loc[nid, 'skaggs_score']:.3f}", fontsize=6)
        ax.axis("off")
    for i in range(n_top, len(axes_flat)):
        axes_flat[i].axis("off")
    fig.suptitle(f"Top {n_top} by Skaggs — before | {n_pass}/{hidden_dim} place cells", fontsize=9)
    plt.tight_layout()
    return save_figure(fig, stem)


def stitch_pngs_horiz(png_paths: list[Path], stem):
    """Paste PNG panels side by side on a white canvas; write the result as PNG and PDF."""
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
    """Render every requested panel, then assemble the composite figure."""
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
        panel_pdfs = figlayout.existing_panels(
            panels_dir, [k for k in LAYOUT.keys() if k in args.panels])
        assemble_figure(panel_pdfs, figures_dir / "figure1_raw")
        print("\nDone.")
        print(f"  Figures: {figures_dir}")
        return

    # The architecture schematic is its own figure now, not a panel of this
    # one, so it is written to its own stem and never enters panel_pdfs.
    if args.model_figure:
        print("Model schematic (standalone figure, drawn by model_schematic.py)...")
        import model_schematic
        model_schematic.main(figures_dir / MODEL_STEM)

    if "a" in args.panels:
        print("Panel a (environment / egocentric observations)...")
        _, pdf = render_panel_b(
            panels_dir / "panel_a_examples",
            panels_dir / "panel_a",
            compact_pose_idx=args.panel_b_pose,
            seed=args.seed,
            size_in=LAYOUT.slot_size_in("a"),
        )
        panel_pdfs["a"] = pdf

    if "b" in args.panels:
        print("Panel b (hidden-state manifold, 3D PCA)...")
        _, pdf = render_panel_d(data_dir, panels_dir / "panel_b", tuple(reward_pos),
                                size_in=LAYOUT.slot_size_in("b"))
        panel_pdfs["b"] = pdf

    if args.panels & {"c", "d", "e", "h", "legacy_f", "legacy_g"}:
        cache_path = figures_dir / "figure_data.npz"
        if args.plot_only:
            saved = require_keys(
                load_figure_data(cache_path),
                "rw_cycles", "goal_cycles", "rm_gc", "skaggs_gc",
                "top_idx_100", "stats",
                path=cache_path,
            )
            rw_cycles = saved["rw_cycles"]
            goal_cycles = saved["goal_cycles"]
            rm_gc = saved["rm_gc"]
            skaggs_gc = saved["skaggs_gc"]
            top_idx_100 = saved["top_idx_100"]
            stats = saved["stats"]
        else:
            df_beh_gc = pd.read_parquet(data_dir / "df_behavior_goal_conditioned.parquet")
            df_act_gc = pd.read_parquet(data_dir / "df_activations_goal_conditioned.parquet")
            rm_gc, occ_gc = build_rate_maps(df_beh_gc, df_act_gc, M, HIDDEN_DIM)
            skaggs_gc = compute_skaggs(rm_gc, occ_gc)
            top_idx_100 = np.argsort(skaggs_gc)[::-1][:100]
            # Exploration = the random walk recorded before reward learning
            # (replay_c1_pred: pre-learning weights). Goal-directed = the
            # post-learning goal-directed probe, without the steps right after
            # each reset (analysis_helpers.reset_keep_mask). Parameters and
            # trajectory both differ, as in a recording before and after
            # learning. The panel code addresses the two maps by cycle label.
            from analysis_helpers import load_period_frames
            maps = {}
            for key, period, trim in (("cycle_1", "replay_c1_pred", False),
                                      ("cycle_1_goal", "cycle_1_goal", True)):
                fb, fa = load_period_frames(str(data_dir), period, HIDDEN_DIM, trim_resets=trim)
                rm_, occ_ = build_rate_maps(fb, fa, M, HIDDEN_DIM)
                maps[key] = (rm_, occ_, compute_skaggs(rm_, occ_))
            rw_cycles = {"cycle_1": maps["cycle_1"]}
            goal_cycles = {"cycle_1_goal": maps["cycle_1_goal"]}
            rw_key = next(k for k in rw_cycles if "cycle_1" in k and "goal" not in k)
            goal_key = next(k for k in goal_cycles if "cycle_1" in k)
            rm_pre, _, sk_pre = rw_cycles[rw_key]
            rm_post, _, sk_post = goal_cycles[goal_key]
            stats = analyse_peak_com_distances(
                rm_pre, sk_pre, rm_post, sk_post, reward_pos,
                skaggs_threshold=args.skaggs_threshold,
            )
            update_figure_data(cache_path, {
                "rw_cycles": rw_cycles,
                "goal_cycles": goal_cycles,
                "rm_gc": rm_gc,
                "skaggs_gc": skaggs_gc,
                "top_idx_100": top_idx_100,
                "stats": stats,
            })

        if "c" in args.panels:
            print("Panel c (random active cells, cycle 1 before/after RL)...")
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
                stem=panels_dir / "panel_c",
                # Both rows are from cycle 1, but they bracket reward learning:
                # replay_c1_pred uses pre-RL weights on an exploratory path,
                # whereas cycle_1_goal uses post-RL weights and the learned
                # goal-directed policy.
                ylabel_top="Before",
                ylabel_bottom="After",
                cmap=RATEMAP,
                size_in=LAYOUT.slot_size_in("c"),
            )
            panel_pdfs["c"] = pdf

        if args.panels & {"d", "e"}:
            # Peak / COM: prediction RW (cycle_1) vs goal-driven policy (cycle_1_goal).
            # Same contrast as panel d — not frozen replay (that belongs in Figure 2).
            period_label = "cycle_1 (prediction) → cycle_1_goal (goal policy)"
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

            if "d" in args.panels:
                print("Panel d (peak distance to reward)...")
                _, pdf = fig_dist_histogram(
                    stats["peak_before"], stats["peak_after"],
                    stem=panels_dir / "panel_d",
                    size_in=LAYOUT.slot_size_in("d"),
                    title="Field peak",
                    label_before="Exploration, before",
                    label_after="Goal-directed, after",
                    stats_text=_example_stats(stats, "peak"),
                )
                panel_pdfs["d"] = pdf

            if "e" in args.panels:
                print("Panel e (COM distance to reward)...")
                _, pdf = fig_dist_histogram(
                    stats["com_before"], stats["com_after"],
                    stem=panels_dir / "panel_e",
                    size_in=LAYOUT.slot_size_in("e"),
                    title="Centre of mass",
                    label_before="Exploration, before",
                    label_after="Goal-directed, after",
                    stats_text=_example_stats(stats, "com"),
                )
                panel_pdfs["e"] = pdf

        if "h" in args.panels:
            print(f"Panel h (legacy): cycle-1 before/after for {args.n_cycle1_neurons} neurons...")
            neuron_ids = top_idx_100[: args.n_cycle1_neurons]
            _, pdf = fig_pred_vs_rl_cycle1_many(
                rw_cycles, goal_cycles, neuron_ids, reward_bin,
                stem=panels_dir / "panel_h",
                cmap=RATEMAP,
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

    # `h` and the `legacy_*` panels are opt-in diagnostics with no slot in the
    # layout; they are written to panels/ but never composited.
    composite = {k: v for k, v in panel_pdfs.items() if k in set(LAYOUT.keys())}
    if not args.no_assemble and composite:
        print("Assembling vector composite PDF...")
        assemble_figure(composite, figures_dir / "figure1_raw")

    print("\nDone.")
    print(f"  Figures: {figures_dir}")
    print(f"  Run dir: {run_dir}")


def parse_args():
    """Command-line arguments: run directory, reward position, output paths, panel subset."""
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
    p.add_argument("--panels", default="a,b,c,d,e",
                   help="panels of the map figure; the architecture schematic "
                        "is a separate figure (--no_model_figure to skip)")
    p.add_argument("--no_model_figure", dest="model_figure", action="store_false",
                   help="skip the standalone architecture figure")
    p.add_argument("--skaggs_threshold", type=float, default=0.05,
                   help="Place-cell gate for peak/COM panels (Skaggs in both before and after)")
    p.add_argument("--panel_b_pose", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_show", action="store_true")
    p.add_argument("--no_assemble", action="store_true")
    add_plot_only(p)
    p.add_argument("--assemble_only", action="store_true",
                   help="Skip panel generation; assemble the composite from existing panel PDFs")
    args = p.parse_args()
    args.panels = {x.strip().lower() for x in args.panels.split(",") if x.strip()}
    return args


if __name__ == "__main__":
    generate_all(parse_args())
