#!/usr/bin/env python3
"""
Figure 5 (main text) — confronting the model with two experimental datasets.

Row 1 (Krishnan et al. 2022, reward-expectation extinction; Tier-1 published
source data, never reanalysed raw recordings):
  a  Cartoon: the experiment (VR track, CA1 + VTA-DA axon imaging, reward
     omitted at extinction) and its model analogue (reward + head gradient
     switched off; frozen-replay triple through baseline/learned/extinguished
     weights).
  b  Model result: per-network mean peak distance to reward A across the
     frozen-replay triple — learning concentrates, extinction reverses with
     overshoot (n=13 gate-passing seeds).
Row 2:
  c  Quantitative confrontation, three metrics side by side:
     (i) split-half reliability change (model per-seed, n=13, vs Krishnan
     per-animal, n=3); (ii) fraction of near-reward enrichment lost;
     (iii) reward-proximity signal contrast collapse (model reward-head
     near/far contrast per seed vs Krishnan DA-axon ramp end/start per axon,
     log scale).

Row 3 (Sosa, Plitt & Giocomo 2025, reward relocation; full reanalysis of the
public DANDI 001361 recordings under the model's own criteria):
  d  Cartoon: the experiment (1D VR track, reward zone switched mid-session,
     licking + CA1 imaging) and its model analogue (reward relocated A->B,
     frozen-replay triple).
  e  Reward-relative and place-anchored fractions under the identical
     classification criterion: per-session data (n=7) vs per-network model
     (n=32).
Row 4:
  f  Retargeting latency after the switch: licking (behaviour) vs place-field
     density (map), per session; the model predicts behaviour-referenced
     codes retarget faster than the stored map.
  g  Test i on the data: raw vs occupancy-matched peak shift per session —
     on a stereotyped 1D track occupancy is matched by construction, so the
     model pre-registers ~no attenuation here (a scoped negative control,
     not evidence against occupancy confounds in 2D).

Inputs (all cached, no recomputation):
  Repro/aggregate/extinction_summary.json
  external_data/krishnan2022/cache/krishnan_data.json
  external_data/sosa2025/cache/sosa_tests.json
  Repro/multi_seed_reloc/aggregate/reward_relative_summary.json

Usage:
    python "code_analysis/figure 5/generate_figure5_data.py" --no_show
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from shutil import copy2

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Rectangle
from pypdf import PdfReader, PdfWriter, Transformation

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from utils import (  # noqa: E402
    MM_TO_PT, PAGE_W, apply_nature_style, ensure_dir, save_figure,
)

apply_nature_style()

FIGURES_DIR = HERE / "figures"
PANELS_DIR = FIGURES_DIR / "panels_data"

EXT_CACHE = REPO / "Repro" / "aggregate" / "extinction_summary.json"
KRISHNAN_CACHE = REPO / "external_data" / "krishnan2022" / "cache" / "krishnan_data.json"
SOSA_CACHE = REPO / "external_data" / "sosa2025" / "cache" / "sosa_tests.json"
RR_CACHE = REPO / "Repro" / "multi_seed_reloc" / "aggregate" / "reward_relative_summary.json"

# palette — model is always blue, external data always orange/black,
# reward A teal / reward B orange match Figure 4
COLOR_MODEL = "#4c72b0"
COLOR_DATA = "#e76f51"
COLOR_PRED = "#4c72b0"
COLOR_POSTLEARN = "#8B4513"
COLOR_POSTEXT = "#2a9d8f"
COLOR_A = "#2a9d8f"
COLOR_B = "#e76f51"
COLOR_RAW = "#555555"
COLOR_MATCHED = "#4c72b0"

PERIOD_LABELS = ["baseline\n(pre)", "learned", "extinguished"]


def _fmt_p(p):
    if p is None:
        return "p=n/a"
    return f"p={p:.1e}" if p < 0.001 else f"p={p:.3f}"


def _letter(fig, letter):
    fig.text(0.012, 0.975, letter, fontsize=8, fontweight="bold",
             va="top", ha="left")


def load():
    d = json.loads(EXT_CACHE.read_text())
    kr = json.loads(KRISHNAN_CACHE.read_text())
    so = json.loads(SOSA_CACHE.read_text())
    rr = json.loads(RR_CACHE.read_text())
    return d, kr, so, rr


# ------------------------------------------------------------------ cartoons
def _mouse(ax, x, y, s=1.0, color="0.25"):
    """Minimal side-view mouse glyph at (x, y)."""
    ax.add_patch(Ellipse((x, y), 0.075 * s, 0.038 * s, facecolor=color,
                         edgecolor="none", zorder=6))
    ax.add_patch(Circle((x + 0.045 * s, y + 0.008 * s), 0.020 * s,
                        facecolor=color, edgecolor="none", zorder=6))
    ax.add_patch(Circle((x + 0.041 * s, y + 0.024 * s), 0.009 * s,
                        facecolor=color, edgecolor="none", zorder=6))
    ax.plot([x - 0.036 * s, x - 0.075 * s], [y, y - 0.014 * s],
            color=color, lw=0.7, zorder=6, solid_capstyle="round")


def _star(ax, x, y, color, s=55, crossed=False, ec="black"):
    ax.scatter([x], [y], marker="*", s=s, color=color, edgecolor=ec,
               linewidths=0.4, zorder=7)
    if crossed:
        r = 0.028
        for sx in (-1, 1):
            ax.plot([x - r, x + r], [y - sx * r, y + sx * r], color="crimson",
                    lw=1.1, zorder=8, solid_capstyle="round")


def _track(ax, x0, x1, y, tick_h=0.012):
    ax.plot([x0, x1], [y, y], color="0.25", lw=1.4, solid_capstyle="round",
            zorder=4)
    for xt in np.linspace(x0, x1, 6):
        ax.plot([xt, xt], [y - tick_h, y], color="0.25", lw=0.5, zorder=4)


def _arena(ax, x0, y0, w, star_xy, crossed=False, star_color="k"):
    ax.add_patch(Rectangle((x0, y0), w, w, facecolor="#eef3fa",
                           edgecolor="0.3", lw=0.7, zorder=3))
    sx, sy = star_xy
    _star(ax, x0 + sx * w, y0 + sy * w, star_color, s=42, crossed=crossed)


def _arrow(ax, x0, y0, x1, y1, label=None, color="0.3", fs=5.2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=7, color=color, lw=0.9,
                                 zorder=5))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.030, label, ha="center",
                va="bottom", fontsize=fs, color=color)


def _cartoon_axes():
    fig, ax = plt.subplots(figsize=(3.35, 2.24))
    ax.set_xlim(0, 1.50)
    ax.set_ylim(0, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def fig_cartoon_krishnan(stem):
    fig, ax = _cartoon_axes()

    # ---- experiment (top half) ----
    ax.text(0.085, 0.985, "EXPERIMENT — Krishnan et al. 2022", fontsize=6.0,
            fontweight="bold", va="top", color="0.15")
    ax.text(0.085, 0.90, "head-fixed VR track; CA1 place cells and "
            "VTA dopamine axons imaged", fontsize=5.2, va="top", color="0.3")

    _track(ax, 0.06, 0.56, 0.68)
    _mouse(ax, 0.13, 0.715)
    _star(ax, 0.56, 0.715, COLOR_DATA, s=55)
    ax.text(0.31, 0.60, "rewarded laps", fontsize=5.2, ha="center",
            color="0.3")
    ax.text(0.56, 0.78, "reward", fontsize=5.0, ha="center", color=COLOR_DATA)

    _arrow(ax, 0.65, 0.68, 0.82, 0.68, label="reward\nomitted", fs=5.0)

    _track(ax, 0.90, 1.40, 0.68)
    _mouse(ax, 0.97, 0.715)
    _star(ax, 1.40, 0.715, "#f4c7b8", s=55, crossed=True)
    ax.text(1.15, 0.60, "extinction laps (running continues)", fontsize=5.2,
            ha="center", va="top", color="0.3")

    ax.plot([0.02, 1.48], [0.52, 0.52], color="0.8", lw=0.6)

    # ---- model analogue (bottom half) ----
    ax.text(0.02, 0.475, "MODEL ANALOGUE", fontsize=6.0, fontweight="bold",
            va="top", color=COLOR_MODEL)
    _arena(ax, 0.06, 0.09, 0.27, (0.3, 0.3), star_color=COLOR_A)
    ax.text(0.195, 0.045, "cycle 1: reward A, head ON", fontsize=5.2,
            ha="center", va="top", color="0.3")
    _arrow(ax, 0.39, 0.23, 0.55, 0.23,
           label="$R\\to0$,\n$\\lambda_{\\rm rew}\\to0$", fs=5.0)
    _arena(ax, 0.59, 0.09, 0.27, (0.3, 0.3), crossed=True,
           star_color="#bfe0db")
    ax.text(0.725, 0.045, "cycle 2: extinction\n(behaviour continues)",
            fontsize=5.2, ha="center", va="top", color="0.3")

    ax.text(0.95, 0.44, "frozen-replay triple:\nidentical trajectory through",
            fontsize=5.2, va="top", color="0.3")
    ax.text(0.95, 0.30, "baseline $\\to$ learned $\\to$\nextinguished weights",
            fontsize=5.2, va="top", color="0.15", fontweight="bold")
    ax.text(0.95, 0.15, "occupancy fixed;\nonly weights differ", fontsize=5.0,
            va="top", color="0.45", style="italic")

    _letter(fig, "a")
    fig.tight_layout(pad=0.2)
    return save_figure(fig, stem)


def fig_cartoon_sosa(stem):
    fig, ax = _cartoon_axes()

    # ---- experiment (top half) ----
    ax.text(0.085, 0.985, "EXPERIMENT — Sosa, Plitt & Giocomo 2025",
            fontsize=6.0, fontweight="bold", va="top", color="0.15")
    ax.text(0.085, 0.90, "VR track; reward zone relocated mid-session; "
            "licking + CA1 imaging (DANDI 001361)", fontsize=5.2, va="top",
            color="0.3")

    _track(ax, 0.06, 0.56, 0.68)
    _mouse(ax, 0.12, 0.715)
    ax.add_patch(Rectangle((0.33, 0.665), 0.10, 0.030, facecolor=COLOR_A,
                           alpha=0.55, edgecolor="none", zorder=5))
    ax.text(0.38, 0.735, "zone A", fontsize=5.0, ha="center", color=COLOR_A)
    ax.text(0.31, 0.60, "pre-switch trials", fontsize=5.2, ha="center",
            color="0.3")

    _arrow(ax, 0.65, 0.68, 0.82, 0.68, label="zone\nmoved", fs=5.0)

    _track(ax, 0.90, 1.40, 0.68)
    _mouse(ax, 0.96, 0.715)
    ax.add_patch(Rectangle((1.22, 0.665), 0.10, 0.030, facecolor=COLOR_B,
                           alpha=0.55, edgecolor="none", zorder=5))
    ax.text(1.27, 0.735, "zone B", fontsize=5.0, ha="center", color=COLOR_B)
    ax.text(1.15, 0.60, "post-switch trials (licking tracked)", fontsize=5.2,
            ha="center", va="top", color="0.3")

    ax.plot([0.02, 1.48], [0.52, 0.52], color="0.8", lw=0.6)

    # ---- model analogue (bottom half) ----
    ax.text(0.02, 0.475, "MODEL ANALOGUE", fontsize=6.0, fontweight="bold",
            va="top", color=COLOR_MODEL)
    _arena(ax, 0.06, 0.09, 0.27, (0.3, 0.3), star_color=COLOR_A)
    ax.text(0.195, 0.045, "phase 1: reward A", fontsize=5.2, ha="center",
            va="top", color="0.3")
    _arrow(ax, 0.39, 0.23, 0.55, 0.23, label="reward\nrelocated", fs=5.0)
    _arena(ax, 0.59, 0.09, 0.27, (0.7, 0.7), star_color=COLOR_B)
    ax.text(0.725, 0.045, "phase 2: reward B", fontsize=5.2, ha="center",
            va="top", color="0.3")

    ax.text(0.95, 0.44, "same per-cell criterion\non model and data:",
            fontsize=5.2, va="top", color="0.3")
    ax.text(0.95, 0.30, "reward-relative vs\nplace-anchored cells",
            fontsize=5.2, va="top", color="0.15", fontweight="bold")
    ax.text(0.95, 0.15, "frozen replay: baseline\n$\\to$ after A $\\to$ after B",
            fontsize=5.0, va="top", color="0.45", style="italic")

    _letter(fig, "d")
    fig.tight_layout(pad=0.2)
    return save_figure(fig, stem)


# ------------------------------------------------------------- data panels
def fig_model_extinction(d, stem):
    """Panel b: per-seed frozen-replay mean peak distance to A (triple)."""
    rows = d["per_seed"]
    k1 = d["k1"]
    fig, ax = plt.subplots(figsize=(2.45, 2.24))
    x = np.arange(3)
    for r in rows:
        md = r["k1"]["mean_dist"]
        ax.plot(x, md, color="0.75", lw=0.7, alpha=0.7, zorder=2)
    mp = k1["mean_dist_by_period"]
    means = [mp[p]["mean"] for p in ("pred", "postLearn", "postExt")]
    sds = [mp[p]["sd"] for p in ("pred", "postLearn", "postExt")]
    for xi, m, s, c in zip(x, means, sds,
                           [COLOR_PRED, COLOR_POSTLEARN, COLOR_POSTEXT]):
        ax.errorbar([xi], [m], yerr=[s], fmt="D", color=c, markersize=4.5,
                    capsize=2.5, lw=1.2, zorder=5, markeredgecolor="black",
                    markeredgewidth=0.4)
    ax.plot(x, means, color="black", lw=1.1, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(PERIOD_LABELS, fontsize=5.4)
    ax.set_ylabel("mean peak distance to reward A")
    dl, de, rec = k1["delta_learn"], k1["delta_ext"], k1["recovery_pooled"]
    ax.set_title(
        f"Model: extinction reverses the shift\n"
        f"learn {dl['mean']:+.3f} ({_fmt_p(dl['p'])}); "
        f"exting. {de['mean']:+.3f} ({_fmt_p(de['p'])})\n"
        f"recovery {rec['point']:.2f} [{rec['ci_lo']:.2f},{rec['ci_hi']:.2f}]"
        f" (>1 = overshoot); n={k1['n_seeds']}", fontsize=5.3)
    _letter(fig, "b")
    fig.tight_layout(pad=0.35)
    return save_figure(fig, stem)


def fig_confront_krishnan(d, kr, stem):
    """Panel c: three model-vs-data metrics side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(6.0, 1.95),
                             gridspec_kw={"width_ratios": [1, 1, 1.35]})

    # (i) split-half reliability change (%)
    ax = axes[0]
    per_seed_drop = [
        100.0 * (r["k2"]["postExt"]["all"] - r["k2"]["postLearn"]["all"])
        / r["k2"]["postLearn"]["all"] for r in d["per_seed"]]
    kr_drop = [100.0 * v for v in
               kr["reliability"]["per_animal"]["frac_drop"]]
    for i, (vals, color) in enumerate([(per_seed_drop, COLOR_MODEL),
                                       (kr_drop, COLOR_DATA)]):
        ax.bar([i], [np.mean(vals)], color=color, width=0.58,
               edgecolor="black", linewidth=0.4, zorder=2)
        rng = np.random.default_rng(7 + i)
        ax.plot(i + rng.uniform(-0.13, 0.13, len(vals)), vals, "o", ms=2.6,
                color="black", mec="white", mew=0.3, alpha=0.8, zorder=3)
    ax.axhline(0, color="0.3", lw=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"model\n(n={len(per_seed_drop)} nets)",
                        f"Krishnan\n(n={len(kr_drop)} mice)"], fontsize=5.4)
    ax.set_ylabel("split-half reliability change (%)")
    ax.set_title("Reliability drop under extinction\n"
                 "(dots: networks / animals)", fontsize=5.7)

    # (ii) fraction of near-reward enrichment lost (%)
    ax = axes[1]
    nf = d["k1"]["near_frac_by_period"]
    pred_, learn_, ext_ = (nf["pred"]["mean"], nf["postLearn"]["mean"],
                           nf["postExt"]["mean"])
    model_lost = 100.0 * (learn_ - ext_) / (learn_ - pred_)
    fd = kr["field_density"]["by_condition_counts"]
    e_rew = fd["rewarded"]["enrichment_vs_uniform"]["end"]
    e_low = fd["re_low"]["enrichment_vs_uniform"]["end"]
    kr_lost = 100.0 * (e_rew - e_low) / (e_rew - 1.0)
    ax.bar([0, 1], [model_lost, kr_lost], color=[COLOR_MODEL, COLOR_DATA],
           width=0.58, edgecolor="black", linewidth=0.4)
    ax.axhline(100, color="0.3", lw=0.6, ls="--")
    ax.text(1.42, 100, "full\nloss", fontsize=5.0, va="center", color="0.4")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"model\n({model_lost:.0f}%)",
                        f"Krishnan\n({kr_lost:.0f}%)"], fontsize=5.4)
    ax.set_ylabel("near-reward enrichment lost (%)")
    ax.set_title("Enrichment loss\n(model overshoots; data partial)",
                 fontsize=5.7)
    ax.set_xlim(-0.6, 1.75)

    # (iii) proximity-signal contrast collapse (log scale)
    ax = axes[2]
    m_learn, m_ext = [], []
    for r in d["per_seed"]:
        pl = np.array(r["k3"]["postLearn"], dtype=float)
        pe = np.array(r["k3"]["postExt"], dtype=float)
        eps = 1e-6
        m_learn.append(max(pl[0] / max(pl[-1], eps), eps))
        m_ext.append(max(pe[0] / max(pe[-1], eps), eps))
    for a, b in zip(m_learn, m_ext):
        ax.plot([0, 1], [a, b], color=COLOR_MODEL, lw=0.6, alpha=0.45,
                zorder=2)
    ax.plot([0, 1], [np.median(m_learn), np.median(m_ext)], color=COLOR_MODEL,
            lw=1.6, marker="D", ms=4, zorder=4, markeredgecolor="black",
            markeredgewidth=0.4)
    da = kr["da_ramp"]
    rew_contrast = [e / s for e, s in zip(da["per_axon"]["end"],
                                          da["per_axon"]["start"])]
    ext_ctx = da["extinction_context"]["re_low"]["mean"]
    ext_contrast_mean = ext_ctx["end"] / ext_ctx["start"]
    rng = np.random.default_rng(11)
    ax.plot(2.2 + rng.uniform(-0.09, 0.09, len(rew_contrast)), rew_contrast,
            "o", ms=2.6, color=COLOR_DATA, mec="white", mew=0.3, zorder=3)
    # median per-axon contrast (the mean is distorted by one outlier axon,
    # 6.6x; median 2.4x matches the ratio-of-means convention in the text)
    ax.plot([2.2, 3.2], [np.median(rew_contrast), ext_contrast_mean],
            color=COLOR_DATA, lw=1.6, marker="D", ms=4, zorder=4,
            markeredgecolor="black", markeredgewidth=0.4)
    ax.axhline(1.0, color="0.3", lw=0.6, ls=":")
    ax.text(3.55, 1.0, "no\nsignal", fontsize=5.0, va="center", color="0.4")
    ax.set_yscale("log")
    ax.set_xticks([0, 1, 2.2, 3.2])
    ax.set_xticklabels(["learned", "exting.", "rewarded", "exting."],
                       fontsize=5.2)
    for tick, col in zip(ax.get_xticklabels(),
                         [COLOR_MODEL, COLOR_MODEL, COLOR_DATA, COLOR_DATA]):
        tick.set_color(col)
    cc = d["k3"]["contrast_near_over_far"]
    ax.set_title(
        f"Proximity signal collapses (log; medians)\nmodel head "
        f"{cc['postLearn']['median']:.1f}x$\\to${cc['postExt']['median']:.1f}x"
        f" (n=13); DA ramp {np.median(rew_contrast):.1f}x$\\to$"
        f"{ext_contrast_mean:.1f}x (n=7 axons)", fontsize=5.5)
    ax.set_ylabel("near/far output ratio")

    _letter(fig, "c")
    fig.tight_layout(pad=0.35, w_pad=1.0)
    return save_figure(fig, stem)


def fig_rr_fractions(so, rr, stem):
    """Panel e: reward-relative / place-anchored fractions, model vs data."""
    rows = [r for r in so["per_session"] if "skipped" not in r]
    data_rr = [r["rr_frac"] for r in rows]
    data_pa = [r["place_anchored_frac"] for r in rows]
    per_seed = rr["per_seed"]
    model_rr = [s["controlled"]["frac_reward_relative"] for s in per_seed]
    model_pa = [s["controlled"]["frac_place_anchored"] for s in per_seed]

    fig, ax = plt.subplots(figsize=(2.45, 2.24))
    rng = np.random.default_rng(3)
    for x0, mvals, dvals in [(0.0, model_rr, data_rr),
                             (1.0, model_pa, data_pa)]:
        xm = x0 - 0.18 + rng.uniform(-0.07, 0.07, len(mvals))
        ax.plot(xm, mvals, "o", ms=2.2, color=COLOR_MODEL, alpha=0.55,
                mec="none", zorder=2)
        ax.plot([x0 - 0.30, x0 - 0.06], [np.mean(mvals)] * 2,
                color=COLOR_MODEL, lw=1.4, zorder=4)
        xd = x0 + 0.18 + rng.uniform(-0.07, 0.07, len(dvals))
        ax.plot(xd, dvals, "o", ms=3.2, color="black", mec="white", mew=0.4,
                zorder=3)
        ax.plot([x0 + 0.06, x0 + 0.30], [np.mean(dvals)] * 2, color="black",
                lw=1.4, zorder=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["reward-\nrelative", "place-\nanchored"],
                       fontsize=5.6)
    ax.set_ylabel("fraction of cells")
    ax.set_ylim(0, max(max(model_pa), max(data_pa)) * 1.22)
    p = so["summary"]["wilcoxon_rrfrac_vs_model"]["p"]
    ax.text(0.03, 0.98,
            f"RR: model {np.mean(model_rr)*100:.1f}$\\pm$"
            f"{np.std(model_rr, ddof=1)*100:.1f}% vs\ndata "
            f"{np.mean(data_rr)*100:.1f}$\\pm$"
            f"{np.std(data_rr, ddof=1)*100:.1f}% (p={p:.2f})",
            transform=ax.transAxes, fontsize=5.2, va="top")
    handles = [plt.Line2D([], [], marker="o", ls="none", ms=3,
                          color=COLOR_MODEL,
                          label=f"model (n={len(model_rr)} nets)"),
               plt.Line2D([], [], marker="o", ls="none", ms=3, color="black",
                          label=f"data (n={len(data_rr)} sessions)")]
    ax.legend(handles=handles, fontsize=5.0, frameon=False,
              loc=(0.02, 0.60))
    ax.set_title("Identical per-cell criterion,\nmodel and reanalysed data",
                 fontsize=5.8)
    _letter(fig, "e")
    fig.tight_layout(pad=0.35)
    return save_figure(fig, stem)


def fig_latency(so, stem):
    """Panel f: lick vs map retargeting latency per session."""
    rows = [r for r in so["per_session"] if "skipped" not in r]
    pairs = [(r["lick_latency"], r["map_latency"]) for r in rows
             if r["lick_latency"] is not None]
    lim = 1 + max([p[1] or 0 for p in pairs] + [p[0] for p in pairs] + [15])

    fig, ax = plt.subplots(figsize=(3.0, 1.95))
    ax.plot([0, lim], [0, lim], color="#999999", lw=0.6, ls="--", zorder=1)
    ax.text(lim * 0.80, lim * 0.72, "unity", fontsize=5.4, color="#777777",
            rotation=38)
    ax.axhspan(0, 5, color="0.92", zorder=0)
    ax.text(lim * 0.03, 2.5, "map latency lower bound (5-trial window)",
            fontsize=4.8, ha="left", va="center", color="0.45")
    n_undef = 0
    rng = np.random.default_rng(5)
    for lick, mapl in pairs:
        if mapl is None:
            n_undef += 1
            ax.plot(lick, lim - 0.5, marker="^", ms=3.5, color="black",
                    mec="white", mew=0.4, zorder=3)
            continue
        jx, jy = rng.uniform(-0.15, 0.15, 2)
        ax.plot(lick + jx, mapl + jy, "o", ms=3.5, color="black",
                mec="white", mew=0.4, zorder=3)
    if n_undef:
        ax.text(0.03, 0.96, f"$\\blacktriangle$ map never reached\n"
                f"pre-switch level (n={n_undef})", transform=ax.transAxes,
                fontsize=5.2, va="top")
    p = so["summary"]["wilcoxon_lick_vs_map_latency"]["p"]
    win = so["summary"]["prediction4_lick_before_map"]
    ax.text(0.97, 0.35,
            f"lick faster: {win['n_lick_faster']}/{win['n_pairs']} sessions\n"
            f"Wilcoxon p={p:.4f}\n(floor at n={win['n_pairs']})",
            transform=ax.transAxes, fontsize=5.2, ha="right", va="bottom")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("lick retargeting latency (trials)")
    ax.set_ylabel("map transfer latency (trials)")
    ax.set_title("Behaviour retargets before the map\n"
                 "(model: goal code re-anchors, map lags)", fontsize=5.7)
    _letter(fig, "f")
    fig.tight_layout(pad=0.35)
    return save_figure(fig, stem)


def fig_test_i(so, stem):
    """Panel g: raw vs occupancy-matched peak shift per session (Test i)."""
    rows = [r for r in so["per_session"] if "skipped" not in r]
    raw = [r["delta_raw"] for r in rows]
    mat = [r["delta_matched"] for r in rows]

    fig, ax = plt.subplots(figsize=(3.0, 1.95))
    x0, x1 = 0.0, 1.0
    for rv, mv in zip(raw, mat):
        ax.plot([x0, x1], [rv, mv], color="#bbbbbb", lw=0.5, zorder=1)
    ax.plot([x0] * len(raw), raw, "o", ms=3.5, color=COLOR_RAW, mec="white",
            mew=0.4, zorder=2)
    ax.plot([x1] * len(mat), mat, "o", ms=3.5, color=COLOR_MATCHED,
            mec="white", mew=0.4, zorder=2)
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_xticks([x0, x1])
    ax.set_xticklabels(["raw", "occupancy-\nmatched"], fontsize=5.6)
    ax.set_xlim(-0.6, 1.6)
    lo, hi = min(min(raw), min(mat)), max(max(raw), max(mat))
    ax.set_ylim(lo - 0.05, hi + 0.015)
    ax.set_ylabel("peak shift toward new zone\n($-$ = toward, normalised)")
    p = so["summary"]["wilcoxon_raw_vs_matched"]["p"]
    ax.text(0.97, 0.05,
            f"raw vs matched p = {p:.2f}\n(n={len(raw)} sessions)",
            transform=ax.transAxes, fontsize=5.2, ha="right", va="bottom")
    ax.set_title("Occupancy matching on a 1D track:\npre-registered "
                 "$\\approx$no attenuation (laps repeat)", fontsize=5.7)
    _letter(fig, "g")
    fig.tight_layout(pad=0.35)
    return save_figure(fig, stem)


# ---------------------------------------------------------------- assembly
def _pdf_to_png(pdf, png, dpi=300):
    with tempfile.TemporaryDirectory() as tmp:
        pre = Path(tmp) / "p"
        subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-singlefile",
                        str(pdf), str(pre)], check=True)
        copy2(pre.with_suffix(".png"), png)


def assemble(pdfs, out_stem, dpi=300):
    """Rows of panels; column widths proportional to native panel widths, so
    every panel in a row is placed at the same scale (~1.0)."""
    out_stem = Path(out_stem).with_suffix("")
    M_, G = 8 * MM_TO_PT, 3.5 * MM_TO_PT
    rows = [("a", "b"), ("c",), ("d", "e"), ("f", "g")]
    sizes = {k: (lambda b: (float(b.width), float(b.height)))(
        PdfReader(str(pdfs[k])).pages[0].mediabox) for r in rows for k in r}
    heights, scales = [], []
    for r in rows:
        avail = PAGE_W - 2 * M_ - (len(r) - 1) * G
        sc = avail / sum(sizes[k][0] for k in r)
        scales.append(sc)
        heights.append(max(sizes[k][1] * sc for k in r))
    page_h = 2 * M_ + sum(heights) + (len(rows) - 1) * G
    writer = PdfWriter()
    page = writer.add_blank_page(width=PAGE_W, height=page_h)
    y = M_
    for r, rh, sc in zip(reversed(rows), reversed(heights), reversed(scales)):
        left = M_
        for k in r:
            iw, ih = sizes[k]
            page.merge_transformed_page(
                PdfReader(str(pdfs[k])).pages[0],
                Transformation().scale(sc).translate(left, y + rh - ih * sc),
                over=True)
            left += iw * sc + G
        y += rh + G
    pdf_path = out_stem.with_suffix(".pdf")
    with open(pdf_path, "wb") as f:
        writer.write(f)
    _pdf_to_png(pdf_path, out_stem.with_suffix(".png"), dpi)
    for r, sc in zip(rows, scales):
        print(f"  row {r}: scale {sc:.2f}")
    print(f"  page: {PAGE_W:.0f} x {page_h:.0f} pt")
    print(f"  Saved: {pdf_path}\n  Saved: {out_stem.with_suffix('.png')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_show", action="store_true")
    ap.parse_args()
    ensure_dir(PANELS_DIR)
    d, kr, so, rr = load()
    print(f"extinction seeds: {d['k1']['n_seeds']}; "
          f"sosa sessions: {so['summary']['n_sessions_analyzed']}; "
          f"reloc networks: {rr['summary']['n_seeds']}")

    pdfs = {}
    print("a (Krishnan cartoon)…"); _, pdfs["a"] = fig_cartoon_krishnan(PANELS_DIR / "panel_a")
    print("b (model extinction)…"); _, pdfs["b"] = fig_model_extinction(d, PANELS_DIR / "panel_b")
    print("c (Krishnan confrontation)…"); _, pdfs["c"] = fig_confront_krishnan(d, kr, PANELS_DIR / "panel_c")
    print("d (Sosa cartoon)…"); _, pdfs["d"] = fig_cartoon_sosa(PANELS_DIR / "panel_d")
    print("e (RR fractions)…"); _, pdfs["e"] = fig_rr_fractions(so, rr, PANELS_DIR / "panel_e")
    print("f (latency)…"); _, pdfs["f"] = fig_latency(so, PANELS_DIR / "panel_f")
    print("g (test i)…"); _, pdfs["g"] = fig_test_i(so, PANELS_DIR / "panel_g")
    print("Assembling…"); assemble(pdfs, FIGURES_DIR / "figure5_data_raw")
    print("Done.")


if __name__ == "__main__":
    main()
