#!/usr/bin/env python3
"""
Supplementary Figure 1 — map quality and behavioural field shift, network by
network, for the 55 retained networks without proximity teaching (the
main-text cohort, analysis_helpers.campaign_passers("paired")).

Every statistic uses the main-text definition: field centres from
analysis_helpers.load_period_centers_cached, place cells = Skaggs >= 0.3 in
both periods compared, delta = d_after - d_before (negative = toward reward).

Panels:
  a  % of the 512 units that are place cells at the end of Phase A and at the
     end of Phase B, on the identical replayed trajectory (paired per network)
  b  mean Skaggs spatial information of those place cells, same two states
  c  behavioural shift (exploration before learning -> goal-directed after), peak and COM
  d  % of cells moving toward reward, peak and COM
  e  distance to reward of every place cell, pooled over networks, exploration
     vs goal-directed, peak and COM; points are network means
  f  Skaggs-threshold sensitivity: % place cells and behavioural shift
Error bars are s.e.m. across networks.

Usage (run from repo root):
    python code/figures/fig1/generate_figure1_supp.py --no_show
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code" / "lib"))

import figlayout  # noqa: E402
from analysis_helpers import campaign_passers, load_period_centers_cached  # noqa: E402
from figure_cache import add_plot_only, load_figure_data, require_keys, update_figure_data  # noqa: E402
from pubstyle import (  # noqa: E402
    C, FS_SMALL, FS_TICK, METRIC_ALPHA, METRIC_LABEL, apply_nature_style,
    ensure_dir, fmt_p, mean_sem_marker, network_strip, save_figure, sem,
    stat_annot, style_violin, title, zero_line,
)

apply_nature_style()

FIG1_DIR = REPO / "code" / "figures" / "fig1"
SUPP_FIGURES_DIR = FIG1_DIR / "figures_supp"
REWARD = np.array([0.3, 0.3])
HIDDEN = 512
SKAGGS = 0.3
# behavioural contrast: exploration before learning -> goal-directed after
# learning (reset transients dropped from the goal-directed probe)
EXPLORE, GOAL = "replay_c1_pred", "cycle_1_goal"
END_A, END_B = "replay_c1_pred", "replay_c1_postrl"    # same trajectory, pre/post Phase B
SWEEP = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]

LAYOUT = figlayout.Layout(
    [("a", 0, 0, 3), ("b", 0, 3, 3),
     ("c", 1, 0, 3), ("d", 1, 3, 3),
     ("e", 2, 0, 3), ("f", 2, 3, 3)],
    ncols=6, row_h_in=[1.75, 1.75, 1.85], caption_lines=6,
    name="figS_pipeline",
)


# ---------------------------------------------------------------------------
# Per-network rows
# ---------------------------------------------------------------------------

def load_seed(seed: int, multi_seed_dir: Path) -> dict | None:
    """Map-quality and behavioural-shift summary of one network."""
    data = multi_seed_dir / f"nohead_seed{seed}" / "data"
    try:
        P = {p: load_period_centers_cached(str(data), p, trim_resets=(p == GOAL))
             for p in (EXPLORE, GOAL, END_A, END_B)}
    except FileNotFoundError as exc:
        print(f"  seed {seed}: skip ({exc})")
        return None
    row: dict = {"seed": seed}
    for tag, period in (("A", END_A), ("B", END_B)):
        sk = P[period][0]
        place = sk >= SKAGGS
        row[f"pct_place_{tag}"] = 100.0 * place.mean()
        row[f"skaggs_{tag}"] = float(sk[place].mean())
    sk_e, sk_g = P[EXPLORE][0], P[GOAL][0]
    for metric, idx in (("peak", 2), ("com", 1)):
        d_e = np.linalg.norm(P[EXPLORE][idx] - REWARD, axis=1)
        d_g = np.linalg.norm(P[GOAL][idx] - REWARD, axis=1)
        gate = (sk_e >= SKAGGS) & (sk_g >= SKAGGS)
        delta = d_g[gate] - d_e[gate]
        row[f"delta_{metric}"] = float(delta.mean())
        row[f"pct_toward_{metric}"] = 100.0 * float((delta < 0).mean())
        row[f"dist_explore_{metric}"] = d_e[gate]
        row[f"dist_goal_{metric}"] = d_g[gate]
        for thr in SWEEP:
            g = (sk_e >= thr) & (sk_g >= thr)
            row[f"sweep_delta_{metric}_{thr}"] = float((d_g[g] - d_e[g]).mean()) if g.any() else np.nan
    for thr in SWEEP:
        row[f"sweep_pct_place_{thr}"] = 100.0 * float(((sk_e >= thr) & (sk_g >= thr)).mean())
    return row


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def _p_vs(vals, ref=0.0):
    v = np.asarray(vals, float) - ref
    return wilcoxon(v).pvalue


def _paired_before_after(rows, key, ylabel, ttl, stem, size_in, dec=2):
    """End of Phase A vs end of Phase B on the same trajectory, one line per network."""
    a = np.array([r[f"{key}_A"] for r in rows])
    b = np.array([r[f"{key}_B"] for r in rows])
    fig, ax = plt.subplots(figsize=size_in)
    network_strip(ax, [a, b], [0, 1], [C["frozen_pre"], C["frozen_post"]])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["End of\nPhase A", "End of\nPhase B"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel(ylabel)
    title(ax, ttl)
    stat_annot(ax, f"Phase A {a.mean():.{dec}f} ± {sem(a):.{dec}f}\n"
                   f"Phase B {b.mean():.{dec}f} ± {sem(b):.{dec}f}\n"
                   f"paired {fmt_p(wilcoxon(a, b).pvalue)}", loc="upper right")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.35 * (hi - lo))
    fig.tight_layout(pad=0.35)
    return save_figure(fig, stem, exact_size=True)


def fig_metric_strips(rows, key, ylabel, ttl, ref, ref_label, stem, size_in):
    """Per-network values for peak and COM, mean +/- s.e.m., Wilcoxon vs `ref`."""
    fig, ax = plt.subplots(figsize=size_in)
    vals = [np.array([r[f"{key}_{m}"] for r in rows]) for m in ("peak", "com")]
    for x, m, v in zip((0, 1), ("peak", "com"), vals):
        network_strip(ax, [v], [x], [C["policy"]], pair=False,
                      alpha=0.6 * METRIC_ALPHA[m] + 0.1, seed=x)
    ax.axhline(ref, color=C["zero"], lw=0.5, ls=(0, (2.5, 2)), zorder=0.5)
    ax.text(1.45, ref, ref_label, fontsize=FS_SMALL, color=C["muted"], va="bottom", ha="right")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([METRIC_LABEL["peak"], METRIC_LABEL["com"]])
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylabel(ylabel)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.30 * (hi - lo))
    for x, v in zip((0, 1), vals):
        ax.text(x, hi + 0.05 * (hi - lo),
                f"{v.mean():+.3f} ± {sem(v):.3f}\n{fmt_p(_p_vs(v, ref))}"
                if ref == 0 else
                f"{v.mean():.1f} ± {sem(v):.1f}%\n{fmt_p(_p_vs(v, ref))}",
                ha="center", va="bottom", fontsize=FS_SMALL, color=C["muted"],
                linespacing=1.3)
    title(ax, ttl)
    fig.tight_layout(pad=0.35)
    return save_figure(fig, stem, exact_size=True)


def fig_pooled_violins(rows, stem, size_in):
    """Pooled per-cell distance to reward, exploration vs goal-directed, peak and COM."""
    fig, ax = plt.subplots(figsize=size_in)
    xs = {"peak": (0.0, 1.0), "com": (2.4, 3.4)}
    rng = np.random.default_rng(1)
    for m, (x0, x1) in xs.items():
        e = np.concatenate([r[f"dist_explore_{m}"] for r in rows])
        g = np.concatenate([r[f"dist_goal_{m}"] for r in rows])
        for x, v, col in ((x0, e, C["pre"]), (x1, g, C["policy"])):
            style_violin(ax.violinplot([v], positions=[x], widths=0.8, showmeans=False,
                                       showmedians=False, showextrema=False),
                         col, alpha=0.35 + 0.35 * METRIC_ALPHA[m])
        me = np.array([r[f"dist_explore_{m}"].mean() for r in rows])
        mg = np.array([r[f"dist_goal_{m}"].mean() for r in rows])
        j = rng.uniform(-0.07, 0.07, len(rows))
        for i in range(len(rows)):
            ax.plot([x0 + j[i], x1 + j[i]], [me[i], mg[i]], color="#8C8C8C", lw=0.3, alpha=0.6)
        ax.scatter(x0 + j, me, s=4, color="#1A1A1A", lw=0, zorder=4)
        ax.scatter(x1 + j, mg, s=4, color="#1A1A1A", lw=0, zorder=4)
        ax.text((x0 + x1) / 2, -0.15, METRIC_LABEL[m], transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=FS_TICK)
        n_cells = len(e)
    ax.set_xticks([0, 1, 2.4, 3.4])
    ax.set_xticklabels(["Explor.", "Goal-dir.", "Explor.", "Goal-dir."])
    ax.set_xlim(-0.6, 4.0)
    ax.set_ylabel("Distance to reward")
    title(ax, f"Pooled cells ({len(rows)} networks)")
    fig.tight_layout(pad=0.35)
    return save_figure(fig, stem, exact_size=True)


def fig_threshold_sweep(rows, stem, size_in):
    """% place cells and behavioural shift (peak, COM) vs Skaggs threshold, mean +/- s.e.m."""
    fig, axes = plt.subplots(1, 2, figsize=size_in)
    pct = np.array([[r[f"sweep_pct_place_{t}"] for t in SWEEP] for r in rows])
    axes[0].errorbar(SWEEP, pct.mean(0), yerr=[sem(c) for c in pct.T], fmt="o-",
                     color=C["pre"], ms=2.5, lw=0.8, capsize=1.5, elinewidth=0.7)
    axes[0].set_ylabel("Place cells (% of units)")
    title(axes[0], "Place cells vs threshold")
    for m in ("peak", "com"):
        d = np.array([[r[f"sweep_delta_{m}_{t}"] for t in SWEEP] for r in rows])
        axes[1].errorbar(SWEEP, np.nanmean(d, 0), yerr=[sem(c) for c in d.T], fmt="o-",
                         color=C["policy"], alpha=METRIC_ALPHA[m], ms=2.5, lw=0.8,
                         capsize=1.5, elinewidth=0.7, label=METRIC_LABEL[m])
    zero_line(axes[1])
    axes[1].set_ylabel(r"$\Delta$ distance to reward")
    axes[1].legend(loc="upper left")
    title(axes[1], "Shift vs threshold")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(SWEEP)
        ax.set_xticklabels([f"{t:g}" for t in SWEEP])
        ax.minorticks_off()
        ax.axvline(SKAGGS, color=C["muted"], lw=0.5, ls=":", zorder=0)
        ax.set_xlabel("Skaggs threshold")
    fig.tight_layout(pad=0.35, w_pad=0.6)
    return save_figure(fig, stem, exact_size=True)


# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--multi_seed_dir", default=str(REPO / "results" / "multi_seed"))
    p.add_argument("--output_dir", default=str(SUPP_FIGURES_DIR))
    p.add_argument("--panels", default="a,b,c,d,e,f")
    p.add_argument("--no_show", action="store_true")
    p.add_argument("--no_assemble", action="store_true")
    add_plot_only(p)
    args = p.parse_args()
    args.panels = {x.strip() for x in args.panels.split(",") if x.strip()}
    return args


def main():
    args = parse_args()
    out = Path(args.output_dir)
    panels_dir = ensure_dir(out / "panels")
    cache_path = out / "figure_data.npz"
    if args.plot_only:
        rows = require_keys(load_figure_data(cache_path), "rows", path=cache_path)["rows"]
    else:
        seeds = sorted(campaign_passers("paired"))
        print(f"Loading {len(seeds)} retained networks ...")
        rows = [r for r in (load_seed(s, Path(args.multi_seed_dir)) for s in seeds) if r]
        update_figure_data(cache_path, {"rows": rows})
    print(f"  n = {len(rows)} networks")

    S = LAYOUT.slot_size_in
    makers = {
        "a": lambda: _paired_before_after(rows, "pct_place", "Place cells (% of units)",
                                          "Place cells", panels_dir / "panel_a", S("a"), dec=1),
        "b": lambda: _paired_before_after(rows, "skaggs", "Spatial information (bits)",
                                          "Spatial information", panels_dir / "panel_b", S("b")),
        "c": lambda: fig_metric_strips(rows, "delta", r"$\Delta$ distance to reward",
                                       "Behavioural shift per network", 0.0, "no shift",
                                       panels_dir / "panel_c", S("c")),
        "d": lambda: fig_metric_strips(rows, "pct_toward", "Cells moving toward reward (%)",
                                       "Fraction of cells toward reward", 50.0, "chance",
                                       panels_dir / "panel_d", S("d")),
        "e": lambda: fig_pooled_violins(rows, panels_dir / "panel_e", S("e")),
        "f": lambda: fig_threshold_sweep(rows, panels_dir / "panel_f", S("f")),
    }
    panel_pdfs = {}
    for k in LAYOUT.keys():
        if k in args.panels:
            _, panel_pdfs[k] = makers[k]()
    if not args.no_assemble and panel_pdfs:
        figlayout.compose(LAYOUT, panel_pdfs, out / "figure1_supp_raw")
    print(f"\nDone. Outputs: {out}")


if __name__ == "__main__":
    main()
