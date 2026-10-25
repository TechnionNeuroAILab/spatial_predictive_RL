#!/usr/bin/env python3
"""
Supplementary Figure 2 — robustness of the main-text statistics and
example-network field distributions.

Panels:
  a  Entropy-collapse gate: per-family shifts (mean +/- s.e.m.) and the P of
     each contrast as the gate threshold varies (headline_stats.json sweep)
  b  Skaggs place-cell threshold: fixed-trajectory shift with and without
     proximity teaching, peak and centre of mass, in the 55 matched pairs
  c  Distance-to-reward distributions in the proximity-teaching example
     network before and after learning on the identical replayed trajectory

Usage (run from repo root):
    python code/figures/fig5/generate_figure5_supp.py --no_show
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[3]
FIG5_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FIG5_DIR))

from utils import C, FS_SMALL, PHASE_COLORS, apply_nature_style, ensure_dir, save_figure, title, fmt_p, stat_annot, plain_log_ticks, hist_outline  # noqa: E402
import figlayout  # noqa: E402

_LIB = REPO / "code" / "lib"
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))
from analysis_helpers import MIN_ENTROPY, campaign_passers, load_period_centers_cached  # noqa: E402
from figure_cache import add_plot_only, load_figure_data, require_keys, update_figure_data  # noqa: E402
from pubstyle import METRIC_ALPHA, METRIC_LABEL, load_headline_stats, sem, zero_line  # noqa: E402

apply_nature_style()

SUPP_FIGURES_DIR = FIG5_DIR / "figures_supp"
PROXIMITY_CACHE = REPO / "code" / "figures" / "fig3" / "figures" / "figure_data.npz"
RES = REPO / "results"
REWARD = np.array([0.3, 0.3])
FRZ = ("replay_c1_pred", "replay_c1_postrl")
SKAGGS_SWEEP = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
OI_PURPLE = "#CC79A7"
GATE = MIN_ENTROPY

#   a  entropy-gate sweep (two axes)
#   b  Skaggs-threshold sweep            c  example-network distributions
LAYOUT = figlayout.Layout(
    [("a", 0, 0, 6),
     ("b", 1, 0, 3), ("c", 1, 3, 3)],
    ncols=6, row_h_in=[2.10, 2.10], caption_lines=9,
    name="figS_dissociation",
)


# ---------------------------------------------------------------------------
# Panel a -- entropy-collapse gate
# ---------------------------------------------------------------------------

def fig_entropy_sweep(hs, stem, size_in):
    """Left: shifts vs gate threshold. Right: P of each contrast vs threshold."""
    th = [float(t) for t in hs["thresholds"]]
    sw = {fam: hs[fam]["sweep"] for fam in ("paired", "reloc", "extinction")}
    fig, axes = plt.subplots(1, 2, figsize=size_in)

    effects = [
        ("paired", "nh_frz", "Replay, without teaching", C["nohead"]),
        ("paired", "hd_frz", "Replay, with teaching", C["head"]),
        ("paired", "nh_beh", "Goal-directed, without", OI_PURPLE),
        ("reloc", "toward_A", "Relocation, toward A", C["reward_a"]),
        ("reloc", "toward_B", "Relocation, toward B", C["reward_b"]),
    ]
    ax = axes[0]
    for fam, key, lbl, col in effects:
        m = [sw[fam][str(t)]["peak"][key]["mean"] for t in hs["thresholds"]]
        e = [sw[fam][str(t)]["peak"][key]["sem"] for t in hs["thresholds"]]
        ax.errorbar(th, m, yerr=e, fmt="o-", color=col, ms=2.2, lw=0.8,
                    capsize=1.2, elinewidth=0.6, label=lbl)
    zero_line(ax)
    ax.set_ylabel(r"$\Delta$ peak distance to reward")
    title(ax, "Shifts vs gate threshold")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 1.1 * (hi - lo), hi)        # room below the data for the legend
    ax.legend(loc="lower left", fontsize=FS_SMALL, handlelength=1.0)

    contrasts = [
        ("paired", "frz_hd_minus_nh", "Teaching effect, replay", C["head"]),
        ("paired", "interaction", "Condition \u00d7 readout", "#555555"),
        ("paired", "beh_hd_minus_nh", "Teaching effect, goal-dir.", OI_PURPLE),
        ("reloc", "A_minus_B", "Relocation A vs B", C["reward_a"]),
    ]
    ax = axes[1]
    for fam, key, lbl, col in contrasts:
        p = [sw[fam][str(t)]["peak"][key]["p_two_sided"] for t in hs["thresholds"]]
        ax.plot(th, p, "o-", color=col, ms=2.2, lw=0.8, label=lbl)
    ax.set_yscale("log")
    ax.axhline(0.05, color=C["muted"], lw=0.5, ls=(0, (2.5, 2)))
    ax.text(th[-1], 0.04, "P = 0.05", fontsize=FS_SMALL, color=C["muted"], va="top", ha="right")
    plain_log_ticks(ax, "y")
    ax.set_ylabel("Two-sided P")
    title(ax, "Contrasts vs gate threshold")
    ax.legend(loc="lower left", fontsize=FS_SMALL, handlelength=1.0)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo / 1e3, 1.0)

    n_pairs = [sw["paired"][str(t)]["n"] for t in hs["thresholds"]]
    for ax in axes:
        ax.axvline(GATE, color=C["muted"], lw=0.5, ls=":", zorder=0)
        ax.set_xlabel("Entropy gate (bits)")
        ax.set_xticks(th)
        ax.set_xticklabels([f"{t:g}" for t in th])
        top = ax.secondary_xaxis("top")
        top.set_xticks(th)
        top.set_xticklabels([str(n) for n in n_pairs], fontsize=FS_SMALL, color=C["muted"])
        top.tick_params(length=1.5, pad=1)
        top.set_xlabel("Paired networks retained", fontsize=FS_SMALL, color=C["muted"], labelpad=2)
    fig.tight_layout(pad=0.35, w_pad=0.8)
    return save_figure(fig, stem, exact_size=True)


# ---------------------------------------------------------------------------
# Panel b -- Skaggs place-cell threshold
# ---------------------------------------------------------------------------

def skaggs_rows(seeds):
    """Per-network fixed-trajectory shift at each Skaggs threshold, both conditions."""
    rows = []
    for s in seeds:
        r = {"seed": s}
        for arm, d in (("nh", RES / "multi_seed" / f"nohead_seed{s}" / "data"),
                       ("hd", RES / "multi_seed_head" / f"head_seed{s}" / "data")):
            sk0, com0, pk0 = load_period_centers_cached(str(d), FRZ[0])
            sk1, com1, pk1 = load_period_centers_cached(str(d), FRZ[1])
            for metric, (a, b) in (("peak", (pk0, pk1)), ("com", (com0, com1))):
                for t in SKAGGS_SWEEP:
                    g = (sk0 >= t) & (sk1 >= t)
                    dd = np.linalg.norm(b[g] - REWARD, axis=1) - np.linalg.norm(a[g] - REWARD, axis=1)
                    r[f"{arm}_{metric}_{t}"] = float(dd.mean()) if g.sum() >= 10 else np.nan
        rows.append(r)
    return rows


def fig_skaggs_sweep(rows, stem, size_in):
    fig, ax = plt.subplots(figsize=size_in)
    ps = []
    for arm, col, lbl in (("nh", C["nohead"], "Without teaching"), ("hd", C["head"], "With teaching")):
        for metric in ("peak", "com"):
            v = np.array([[r[f"{arm}_{metric}_{t}"] for t in SKAGGS_SWEEP] for r in rows])
            ax.errorbar(SKAGGS_SWEEP, np.nanmean(v, 0), yerr=[sem(c) for c in v.T],
                        fmt="o-" if metric == "peak" else "s--", color=col,
                        alpha=METRIC_ALPHA[metric], ms=2.2, lw=0.8, capsize=1.2,
                        elinewidth=0.6, label=f"{lbl}, {METRIC_LABEL[metric].lower()}")
    for t in SKAGGS_SWEEP:
        a = np.array([r[f"nh_peak_{t}"] for r in rows]); b = np.array([r[f"hd_peak_{t}"] for r in rows])
        ok = np.isfinite(a) & np.isfinite(b)
        ps.append(wilcoxon(a[ok], b[ok]).pvalue)
    zero_line(ax)
    ax.set_xscale("log")
    ax.set_xticks(SKAGGS_SWEEP)
    ax.set_xticklabels([f"{t:g}" for t in SKAGGS_SWEEP])
    ax.minorticks_off()
    ax.axvline(0.3, color=C["muted"], lw=0.5, ls=":", zorder=0)
    ax.set_xlabel("Skaggs threshold")
    ax.set_ylabel(r"$\Delta$ distance to reward (replay)")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.75 * (hi - lo), hi + 0.1 * (hi - lo))
    ax.legend(loc="lower left", fontsize=FS_SMALL, handlelength=1.6)
    title(ax, f"Place-cell threshold (n = {len(rows)} pairs)")
    stat_annot(ax, f"paired P (peak) \u2264 {max(ps):.3f}\nat every threshold", loc="upper right")
    fig.tight_layout(pad=0.35)
    return save_figure(fig, stem, exact_size=True), ps


# ---------------------------------------------------------------------------
# Panel c -- example-network field distributions
# ---------------------------------------------------------------------------

def fig_example_distributions(stats, stem, size_in):
    """Field-distance distributions for the proximity-teaching example."""
    fig, axes = plt.subplots(1, 2, figsize=size_in)
    for ax, key, panel_title in (
        (axes[0], "peak", "Peak"),
        (axes[1], "com", "Centre of mass"),
    ):
        pre = np.asarray(stats[key]["C_pre"], dtype=float)
        post = np.asarray(stats[key]["C"], dtype=float)
        hi = max(float(pre.max()), float(post.max()), 0.15) * 1.05
        edges = np.linspace(0.0, hi, 21)
        for vals, role, label in (
            (pre, "C_pre", "Before"),
            (post, "C_post", "After"),
        ):
            hist_outline(ax, vals, edges, PHASE_COLORS[role], label=label)
            ax.axvline(
                float(vals.mean()), ymax=1 / 1.8, color=PHASE_COLORS[role],
                ls=(0, (2.5, 2)), lw=0.7, zorder=1,
            )
        ax.set_ylim(0, ax.get_ylim()[1] * 1.8)
        delta = post - pre
        p_value = (
            wilcoxon(post, pre).pvalue
            if len(pre) >= 5 and not np.all(delta == 0)
            else None
        )
        ax.legend(loc="upper left", handlelength=1.0, borderaxespad=0.1)
        ax.text(
            0.02, 0.76,
            f"$\\Delta$d = {delta.mean():+.3f}, "
            f"{100 * (delta < 0).mean():.0f}% toward\n"
            f"{fmt_p(p_value)}, n = {len(delta)}",
            transform=ax.transAxes, fontsize=FS_SMALL, color=C["muted"],
            ha="left", va="top", linespacing=1.35,
        )
        ax.set_xlabel("Distance to reward")
        title(ax, panel_title)
    axes[0].set_ylabel("Cells")
    fig.tight_layout(pad=0.35, w_pad=0.5)
    return save_figure(fig, stem, exact_size=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output_dir", default=str(SUPP_FIGURES_DIR))
    p.add_argument("--panels", default="a,b,c")
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
    hs = load_headline_stats()
    if args.plot_only:
        sk_rows = require_keys(load_figure_data(cache_path), "skaggs_rows",
                               path=cache_path)["skaggs_rows"]
    else:
        sk_rows = skaggs_rows(sorted(campaign_passers("paired")))
        update_figure_data(cache_path, {"skaggs_rows": sk_rows})

    S = LAYOUT.slot_size_in
    panel_pdfs = {}
    if "a" in args.panels:
        _, panel_pdfs["a"] = fig_entropy_sweep(hs, panels_dir / "panel_a", S("a"))
    if "b" in args.panels:
        (_, panel_pdfs["b"]), ps = fig_skaggs_sweep(sk_rows, panels_dir / "panel_b", S("b"))
        print("  Skaggs sweep paired P (peak):", ", ".join(f"{p:.2g}" for p in ps))
    if "c" in args.panels:
        example_stats = require_keys(
            load_figure_data(PROXIMITY_CACHE), "stats", path=PROXIMITY_CACHE)["stats"]
        _, panel_pdfs["c"] = fig_example_distributions(
            example_stats, panels_dir / "panel_c", S("c"))
    if not args.no_assemble and panel_pdfs:
        figlayout.compose(LAYOUT, panel_pdfs, out / "figure5_supp_raw")
    print(f"\nDone. Outputs: {out}")


if __name__ == "__main__":
    main()
