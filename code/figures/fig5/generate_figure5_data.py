#!/usr/bin/env python3
"""
Figures 6 and 7 (main text) -- the model against two experimental datasets.
Model quantities are blue, experimental quantities black, and every comparison
shows its dispersion (networks, animals, axons or sessions).

Figure 6, reward omission (Krishnan et al. 2022; published source data only):
  a  Experiment and model protocols.
  b  Model, fixed-trajectory replay through baseline, learned and withdrawn
     weights: field-peak distance to the former reward, and the fraction of
     field peaks near it (50 networks).
  c  Proximity readout: model decoder near/far ratio per network vs VTA-CA1
     axon end/start activity per axon, rewarded and extinction.
  d  Fields near the reward before and after withdrawal: model per network,
     data per animal (reward region = end of track).
  e  Map reliability change: per network vs per animal (n = 12).

Figure 7, reward relocation (Sosa, Plitt & Giocomo 2025; DANDI 001361
reanalysed with the model's own criteria):
  a  Experiment and model protocols.
  b  Field-peak displacement across relocation in units of the reward
     displacement (0 = stayed, 1 = moved with the reward), model vs mice.
  c  Reward-relative and place-anchored fractions: data sessions vs the model
     read out on the fixed replay trajectory and on goal-directed trajectories.
  d  Behavioural retargeting: model success per episode after relocation vs
     first learning; mouse anticipatory-lick preference for the new zone.

Inputs (all cached, no recomputation):
  results/aggregate/extinction_summary.json
  data/krishnan2022/cache/krishnan_data.json  (key "extended")
  data/sosa2025/cache/sosa_tests.json
  results/multi_seed_reloc/aggregate/reward_relative_summary.json

Usage:
    python code/figures/fig5/generate_figure5_data.py --no_show
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

from utils import (  # noqa: E402
    fmt_p, title,
    C, FS_SMALL, FS_TICK, plain_log_ticks,
    COLOR_MODEL, COLOR_DATA, COLOR_A, COLOR_B,
    apply_nature_style, ensure_dir, save_figure,
)
import figlayout  # noqa: E402
from figure_cache import add_plot_only  # noqa: E402

sys.path.insert(0, str(REPO / "code" / "lib"))
from analysis_helpers import campaign_passers, load_period_centers_cached  # noqa: E402
import cartoons as ct  # noqa: E402
from pubstyle import mean_sem_marker, sem, zero_line  # noqa: E402

apply_nature_style()

FIGURES_DIR = HERE / "figures"
PANELS_DIR = FIGURES_DIR / "panels_data"

EXT_CACHE = REPO / "results" / "aggregate" / "extinction_summary.json"
KRISHNAN_CACHE = REPO / "data" / "krishnan2022" / "cache" / "krishnan_data.json"
SOSA_CACHE = REPO / "data" / "sosa2025" / "cache" / "sosa_tests.json"
RR_CACHE = REPO / "results" / "multi_seed_reloc" / "aggregate" / "reward_relative_summary.json"

MODEL_LIGHT = "#8DB8DA"          # model, secondary readout
PERIODS = ["Baseline", "Learned", "Withdrawn"]

OMISSION_LAYOUT = figlayout.Layout(
    [("a", 0, 0, 4), ("b", 0, 4, 4),
     ("c", 1, 0, 3), ("d", 1, 3, 3), ("e", 1, 6, 2)],
    ncols=8, row_h_in=[2.05, 1.95], caption_lines=10, name="fig_omission",
)
RELOC_LAYOUT = figlayout.Layout(
    [("a", 0, 0, 1), ("b", 0, 1, 1),
     ("c", 1, 0, 1), ("d", 1, 1, 1)],
    ncols=2, row_h_in=[2.05, 2.05], caption_lines=10, name="fig_relocdata",
)


def load():
    """The four cached JSONs these figures need."""
    d = json.loads(EXT_CACHE.read_text())
    kr = json.loads(KRISHNAN_CACHE.read_text())
    so = json.loads(SOSA_CACHE.read_text())
    rr = json.loads(RR_CACHE.read_text())
    return d, kr, so, rr


def _strip(ax, x, vals, color, size=4.0, jitter=0.09, alpha=0.6, seed=0, marker="o"):
    """Points for one group plus a mean +/- s.e.m. diamond in black."""
    vals = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    ax.scatter(x + rng.uniform(-jitter, jitter, len(vals)), vals, s=size, color=color,
               alpha=alpha, lw=0, zorder=2, marker=marker)
    mean_sem_marker(ax, x, vals)


def _paired(ax, x0, x1, a, b, color, size=4.0, lw=0.35, alpha=0.5):
    """Paired values joined per unit, with mean +/- s.e.m. at each end."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    for u, v in zip(a, b):
        ax.plot([x0, x1], [u, v], color=color, lw=lw, alpha=alpha, zorder=1)
    ax.scatter([x0] * len(a), a, s=size, color=color, lw=0, alpha=0.8, zorder=2)
    ax.scatter([x1] * len(b), b, s=size, color=color, lw=0, alpha=0.8, zorder=2)
    mean_sem_marker(ax, x0, a)
    mean_sem_marker(ax, x1, b)


# ------------------------------------------------------------------ cartoons
# Shared glyphs (code/lib/cartoons.py): the same mouse, track, arena and arrows
# in both protocol panels. Each panel is two aligned rows -- the experiment on
# top, the model below -- with the two states of each row centred on the same
# x positions, so the eye reads each column as one moment of the protocol.
XL, XR = 0.31, 1.19          # column centres of the two states


def _cartoon_axes(size_in):
    fig, ax = plt.subplots(figsize=size_in)
    ax.set_xlim(0, 1.50)
    ax.set_ylim(0, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def _row_labels(ax):
    ax.text(0.0, 0.985, "Mouse, linear VR track", fontsize=FS_TICK, fontweight="bold",
            va="top", color=COLOR_DATA)
    ax.text(0.0, 0.505, "Model, open arena", fontsize=FS_TICK, fontweight="bold",
            va="top", color=COLOR_MODEL)
    ax.plot([0.0, 1.50], [0.535, 0.535], color="0.85", lw=0.5)


def _mouse_track(ax, xc, zone_x=None, zone_color=None, reward=None, crossed=False):
    """One track state: a mouse running toward the right end of a track."""
    x0, x1, y = xc - 0.27, xc + 0.27, 0.70
    ct.track(ax, x0, x1, y, h=0.035)
    if zone_x is not None:
        ct.zone(ax, zone_x - 0.05, zone_x + 0.05, y, zone_color, h=0.035)
    ct.mouse(ax, x0 + 0.10, y, length=0.17)
    if reward is not None:
        ct.star(ax, x1 - 0.035, y + 0.055, reward, size=48, crossed=crossed)


def _model_arena(ax, xc, star_xy, star_color, crossed=False, w=0.30):
    x0, y0 = xc - w / 2, 0.11
    ct.arena(ax, x0, y0, w)
    ct.path_in_arena(ax, x0, y0, w, seed=3)
    ct.star(ax, x0 + star_xy[0] * w, y0 + star_xy[1] * w, star_color, size=42, crossed=crossed)


def _state_label(ax, x, y, text):
    ax.text(x, y, text, fontsize=FS_SMALL, ha="center", va="top", color="0.25")


def _transition(ax, y, label, note=None):
    ct.arrow(ax, XL + 0.36, y, XR - 0.36, y, color="0.3")
    ax.text((XL + XR) / 2, y + 0.03, label, fontsize=FS_SMALL, ha="center", va="bottom",
            color="0.25", linespacing=1.1)
    if note:
        ax.text((XL + XR) / 2, y - 0.03, note, fontsize=FS_SMALL, ha="center", va="top",
                color=C["muted"], style="italic", linespacing=1.1)


def fig_cartoon_omission(stem, size_in):
    fig, ax = _cartoon_axes(size_in)
    _row_labels(ax)
    _mouse_track(ax, XL, reward=COLOR_DATA)
    _mouse_track(ax, XR, reward="#E9C4B8", crossed=True)
    _state_label(ax, XL, 0.635, "Rewarded laps")
    _state_label(ax, XR, 0.635, "Extinction laps")
    _transition(ax, 0.71, "reward\nomitted")
    _model_arena(ax, XL, (0.3, 0.3), COLOR_A)
    _model_arena(ax, XR, (0.3, 0.3), "#BFDCEB", crossed=True)
    _state_label(ax, XL, 0.085, "Learning")
    _state_label(ax, XR, 0.085, "Withdrawal")
    _transition(ax, 0.26, "reward and\nteaching withdrawn", "same trajectory\nreplayed")
    fig.tight_layout(pad=0.2)
    return save_figure(fig, stem, exact_size=True)


def fig_cartoon_reloc(stem, size_in):
    fig, ax = _cartoon_axes(size_in)
    _row_labels(ax)
    _mouse_track(ax, XL, zone_x=XL + 0.08, zone_color=COLOR_A)
    _mouse_track(ax, XR, zone_x=XR + 0.19, zone_color=COLOR_B)
    _state_label(ax, XL, 0.635, "Reward zone 1")
    _state_label(ax, XR, 0.635, "Reward zone 2")
    _transition(ax, 0.71, "zone moved\nwithin session")
    _model_arena(ax, XL, (0.3, 0.3), COLOR_A)
    _model_arena(ax, XR, (0.7, 0.7), COLOR_B)
    _state_label(ax, XL, 0.085, "Reward at A")
    _state_label(ax, XR, 0.085, "Reward at B")
    _transition(ax, 0.26, "reward\nrelocated", "same trajectory\nreplayed")
    fig.tight_layout(pad=0.2)
    return save_figure(fig, stem, exact_size=True)


# ------------------------------------------------------------- Figure 6
def fig_omission_model(d, stem, size_in):
    """b: distance of field peaks to the former reward, and fields near it."""
    rows = d["per_seed"]
    fig, axes = plt.subplots(1, 2, figsize=size_in)
    x = np.arange(3)
    series = [
        (np.array([r["k1"]["mean_dist"] for r in rows]), "Peak distance to reward", "Distance to reward"),
        (100 * np.array([r["k1"]["near_frac"] for r in rows]), "Fields near reward", "Field peaks within 0.15 (%)"),
    ]
    for ax, (v, ttl, ylab) in zip(axes, series):
        for row in v:
            ax.plot(x, row, color="#C9C9C9", lw=0.35, zorder=1)
        m = v.mean(0)
        ax.errorbar(x, m, yerr=[sem(c) for c in v.T], color=COLOR_MODEL, fmt="o-", ms=3.5,
                    lw=1.1, capsize=2, elinewidth=0.9, zorder=4, mec="white", mew=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(PERIODS, fontsize=FS_SMALL)
        ax.set_xlim(-0.3, 2.3)
        ax.set_ylabel(ylab)
        title(ax, ttl)
    p_dist = (d["k1"]["delta_learn"]["p"], d["k1"]["delta_ext"]["p"])
    nf = series[1][0]
    p_near = (wilcoxon(nf[:, 0], nf[:, 1]).pvalue, wilcoxon(nf[:, 1], nf[:, 2]).pvalue)
    for ax, (p1, p2) in zip(axes, (p_dist, p_near)):
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.16 * (hi - lo))
        for xa, xb, p in ((0, 1, p1), (1, 2, p2)):
            y = hi + 0.03 * (hi - lo)
            ax.plot([xa + 0.08, xb - 0.08], [y, y], color="#555555", lw=0.5)
            ax.text((xa + xb) / 2, y + 0.01 * (hi - lo), fmt_p(p), ha="center", va="bottom",
                    fontsize=FS_SMALL, color="#333333")
    fig.tight_layout(pad=0.35, w_pad=0.8)
    return save_figure(fig, stem, exact_size=True)


# Model-vs-data row (c-e): fixed margins in inches, so the three panels'
# axes share one top and one bottom edge on the page (tight_layout would
# size each axes to its own tick labels).
ROW_TOP_IN, ROW_BOTTOM_IN = 0.26, 0.50


def _row_frame(fig, size_in, left_in=0.48, right_in=0.06, wspace=None):
    w, h = size_in
    fig.subplots_adjust(left=left_in / w, right=1 - right_in / w,
                        bottom=ROW_BOTTOM_IN / h, top=1 - ROW_TOP_IN / h,
                        **({"wspace": wspace} if wspace is not None else {}))


def fig_reliability_change(d, kr, stem, size_in):
    """e: % change of split-half map reliability, model per network vs data per animal."""
    rel = np.array([[r["k2"][p]["all"] for p in ("postLearn", "postExt")] for r in d["per_seed"]])
    model = 100 * (rel[:, 1] - rel[:, 0]) / rel[:, 0]
    data = 100 * np.asarray(kr["extended"]["reliability_n12"]["frac_change"])
    fig, ax = plt.subplots(figsize=size_in)
    _strip(ax, 0, model, COLOR_MODEL, seed=1)
    _strip(ax, 1, data, COLOR_DATA, size=7, alpha=0.75, seed=2)
    zero_line(ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"Model\n(n = {len(model)})", f"Mice\n(n = {len(data)})"])
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylabel("Reliability change (%)")
    p_m = d["k2"]["all"]["postLearn_vs_postExt"]["p"]
    p_d = kr["extended"]["reliability_n12"]["test"]["wilcoxon_p"]
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.18 * (hi - lo))
    for x, p in ((0, p_m), (1, p_d)):
        ax.text(x, hi + 0.04 * (hi - lo), fmt_p(p), ha="center", va="bottom", fontsize=FS_SMALL,
                color="#333333")
    title(ax, "Map reliability")
    _row_frame(fig, size_in)
    return save_figure(fig, stem, exact_size=True)


def fig_enrichment(d, kr, stem, size_in):
    """d: fields near the reward before vs after withdrawal, model and data."""
    nf = 100 * np.array([r["k1"]["near_frac"] for r in d["per_seed"]])
    ef = kr["extended"]["end_fraction_n12"]
    fig, axes = plt.subplots(1, 2, figsize=size_in)
    _paired(axes[0], 0, 1, nf[:, 1], nf[:, 2], COLOR_MODEL)
    axes[0].axhline(nf[:, 0].mean(), color=C["muted"], lw=0.5, ls=(0, (2.5, 2)), zorder=0)
    axes[0].set_ylabel("Fields near reward (%)")
    _paired(axes[1], 0, 1, ef["rewarded"], ef["re_low"], COLOR_DATA, size=6)
    axes[1].axhline(ef["chance_pct"], color=C["muted"], lw=0.5, ls=(0, (2.5, 2)), zorder=0)
    p_m = wilcoxon(nf[:, 1], nf[:, 2]).pvalue
    for ax, ttl, labels, p in ((axes[0], "Model", ("Learn.", "Withdr."), p_m),
                               (axes[1], "Mice", ("Rew.", "Ext."), ef["test"]["wilcoxon_p"])):
        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels, fontsize=FS_SMALL)
        ax.set_xlim(-0.45, 1.45)
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(lo, 0), hi + 0.15 * (hi - lo))
        ax.text(0.5, 0.98, fmt_p(p), transform=ax.transAxes, ha="center", va="top",
                fontsize=FS_SMALL, color="#333333")
        title(ax, ttl)
    _row_frame(fig, size_in, wspace=0.45)
    return save_figure(fig, stem, exact_size=True)


def fig_proximity(d, kr, stem, size_in):
    """c: model decoder near/far ratio vs VTA-CA1 axon end/start, both conditions."""
    k3 = [(r["k3"]["postLearn"][0] / r["k3"]["postLearn"][-1],
           r["k3"]["postExt"][0] / r["k3"]["postExt"][-1]) for r in d["per_seed"] if r.get("k3")]
    k3 = np.array(k3)
    ax_d = kr["extended"]["axon_ramp"]
    data = np.array([ax_d["rewarded"]["end_over_start"], ax_d["re_low"]["end_over_start"]]).T
    fig, ax = plt.subplots(figsize=size_in)
    for (x0, x1), v, col in (((0, 1), k3, COLOR_MODEL), ((2.2, 3.2), data, COLOR_DATA)):
        for u, w in v:
            ax.plot([x0, x1], [u, w], color=col, lw=0.35, alpha=0.5, zorder=1)
        ax.scatter([x0] * len(v), v[:, 0], s=4, color=col, lw=0, alpha=0.8, zorder=2)
        ax.scatter([x1] * len(v), v[:, 1], s=4, color=col, lw=0, alpha=0.8, zorder=2)
        med = np.median(v, 0)
        ax.plot([x0, x1], med, color=col, lw=1.4, zorder=3)
        ax.plot([x0, x1], med, "D", color=col, ms=3.5, mec="white", mew=0.4, zorder=4)
    ax.set_yscale("log")
    plain_log_ticks(ax, "y")
    ax.axhline(1, color=C["muted"], lw=0.5, ls=(0, (2.5, 2)), zorder=0)
    ax.set_xticks([0, 1, 2.2, 3.2])
    ax.set_xticklabels(["Learn.", "Withdr.", "Rew.", "Extinct."], fontsize=FS_SMALL)
    for x, lbl in ((0.5, f"Model decoder\n(n = {len(k3)})"), (2.7, f"VTA–CA1 axons\n(n = {len(data)})")):
        ax.text(x, -0.26, lbl, transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=FS_SMALL, linespacing=1.1)
    ax.set_xlim(-0.4, 3.6)
    ax.set_ylabel("Near/far signal ratio")
    title(ax, "Proximity signal")
    _row_frame(fig, size_in)
    return save_figure(fig, stem, exact_size=True)


# ------------------------------------------------------------- Figure 7
def fig_cell_classes(so, rr, stem, size_in):
    """c: reward-relative / place-anchored fractions, data vs model under two readouts."""
    rows = [r for r in so["per_session"] if "skipped" not in r]
    keep = campaign_passers("reloc")
    ps = [s for s in rr["per_seed"] if int(s["seed"]) in keep]
    fig, ax = plt.subplots(figsize=size_in)
    groups = [("frac_reward_relative", "rr_frac", 0.0), ("frac_place_anchored", "place_anchored_frac", 1.6)]
    for mkey, dkey, x0 in groups:
        fixed = 100 * np.array([s["controlled"][mkey] for s in ps])
        goal = 100 * np.array([s["behavioural"][mkey] for s in ps])
        data = 100 * np.array([r[dkey] for r in rows])
        _strip(ax, x0 - 0.45, fixed, COLOR_MODEL, size=3, jitter=0.1, seed=1)
        _strip(ax, x0, goal, MODEL_LIGHT, size=3, jitter=0.1, seed=2)
        _strip(ax, x0 + 0.45, data, COLOR_DATA, size=7, jitter=0.06, alpha=0.8, seed=3)
        p = wilcoxon(data - fixed.mean()).pvalue
        top = max(fixed.max(), data.max())
        ax.text(x0, top + 6, f"mice vs fixed-traj.\n{fmt_p(p)}", ha="center", va="bottom",
                fontsize=FS_SMALL, color="#333333", linespacing=1.2)
    ax.set_xticks([0, 1.6])
    ax.set_xticklabels(["Reward-relative", "Place-anchored"])
    ax.set_xlim(-0.8, 2.4)
    ax.set_ylim(0, 92)
    ax.set_ylabel("Cells (%)")
    handles = [plt.Line2D([], [], marker="o", ls="none", ms=3, color=COLOR_MODEL,
                          label=f"Model, fixed trajectory (n = {len(ps)})"),
               plt.Line2D([], [], marker="o", ls="none", ms=3, color=MODEL_LIGHT,
                          label="Model, goal-directed trajectories"),
               plt.Line2D([], [], marker="o", ls="none", ms=3.5, color=COLOR_DATA,
                          label=f"Mice (n = {len(rows)} sessions)")]
    ax.legend(handles=handles, loc="upper left", fontsize=FS_SMALL, handletextpad=0.2)
    title(ax, "Cell classes after relocation")
    fig.tight_layout(pad=0.35)
    return save_figure(fig, stem, exact_size=True)


RELOC_A, RELOC_B = np.array([0.3, 0.3]), np.array([0.7, 0.7])
TRACK_CM = 450.0


def _wrap_cm(x):
    """Signed displacement on the circular 450-cm track, in (-225, 225]."""
    return (np.asarray(x, float) + TRACK_CM / 2) % TRACK_CM - TRACK_CM / 2


def displacement_ratios(so):
    """Per-cell field-peak displacement across relocation, in units of the reward
    displacement (0 = stayed, 1 = moved with the reward).

    Model: fixed-trajectory replay through the post-A and post-B weights, cells
    gated at Skaggs >= 0.15 in both, displacement projected on the A->B axis;
    pooled over the retained networks. Data: every place cell of every session.
    """
    u = (RELOC_B - RELOC_A) / np.linalg.norm(RELOC_B - RELOC_A)
    model = []
    for s in sorted(campaign_passers("reloc")):
        d = REPO / "results" / "multi_seed_reloc" / f"reloc_seed{s}" / "data"
        sk_a, _, pk_a = load_period_centers_cached(str(d), "replay_reloc_postA")
        sk_b, _, pk_b = load_period_centers_cached(str(d), "replay_reloc_postB")
        g = (sk_a >= 0.15) & (sk_b >= 0.15)
        model.append((pk_b[g] - pk_a[g]) @ u / np.linalg.norm(RELOC_B - RELOC_A))
    data = []
    for r in (r for r in so["per_session"] if "skipped" not in r):
        c = r["cells"]
        shift = _wrap_cm(c["zone_post_start"] - c["zone_pre_start"])
        data.append(_wrap_cm(np.array(c["pk_post"]) - np.array(c["pk_pre"])) / shift)
    return model, data


def fig_field_displacement(so, stem, size_in):
    """b: where fields go when the reward moves, model vs mice."""
    model, data = displacement_ratios(so)
    bins = np.arange(-2.0, 2.0001, 0.125)       # wrapped track displacements span (-1.9, 1.9)
    fig, ax = plt.subplots(figsize=size_in)
    for vals, col, lbl, fill in ((np.concatenate(model), COLOR_MODEL,
                                  f"Model ({sum(map(len, model))} cells, {len(model)} networks)", True),
                                 (np.concatenate(data), COLOR_DATA,
                                  f"Mice ({sum(map(len, data))} cells, {len(data)} sessions)", False)):
        h = 100 * np.histogram(np.clip(vals, bins[0], bins[-1]), bins)[0] / len(vals)
        if fill:
            ax.stairs(h, bins, color=col, alpha=0.25, fill=True, lw=0)
        ax.stairs(h, bins, color=col, lw=1.0, label=lbl)
    for x in (0, 1):
        ax.axvline(x, color=C["muted"], lw=0.5, ls=(0, (2.5, 2)), zorder=0)
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(0, 40)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_xticklabels(["-2", "-1", "0\nstayed", "1\nmoved with reward", "2"])
    ax.set_xlabel("Field displacement / reward displacement")
    ax.set_ylabel("Cells (%)")
    ax.legend(loc="upper right", fontsize=FS_SMALL, handlelength=1.2, borderaxespad=0.2)
    fig.subplots_adjust(left=0.48 / size_in[0], right=1 - 0.08 / size_in[0],
                        bottom=0.50 / size_in[1], top=1 - 0.30 / size_in[1])
    title(ax, "Field displacement")
    return save_figure(fig, stem, exact_size=True)


def _success_curves():
    """Per-network success per episode (rows) for the retained relocation networks."""
    return np.array([np.load(REPO / "results" / "multi_seed_reloc" / f"reloc_seed{s}" / "learning_curves.npz")
                     ["rl_success"] for s in sorted(campaign_passers("reloc"))], float)


def _lick_preference(so, max_t=30):
    """Per-session (new - old)/(new + old) anticipatory lick rate by trial since
    the switch, as a centred 3-trial running mean."""
    rows = [r for r in so["per_session"] if "skipped" not in r]
    lick = np.full((len(rows), max_t), np.nan)
    for i, r in enumerate(rows):
        sw = r["lick_rates_post"][0]["trial"]
        for e in r["lick_rates_post"]:
            t = e["trial"] - sw
            tot = e["rate_new"] + e["rate_old"]
            if 0 <= t < max_t and tot > 0:
                lick[i, t] = (e["rate_new"] - e["rate_old"]) / tot
    out = np.full_like(lick, np.nan)
    with np.errstate(all="ignore"):
        for j in range(max_t):
            out[:, j] = np.nanmean(lick[:, max(0, j - 1):j + 2], axis=1)
    return out, rows


def _band(ax, x, v, color, min_n=3, **kw):
    m = np.nanmean(v, 0)
    n = np.sum(np.isfinite(v), 0)
    e = np.nanstd(v, 0, ddof=1) / np.sqrt(np.maximum(n, 1))
    ok = n >= min_n
    ax.fill_between(x[ok], (m - e)[ok], (m + e)[ok], color=color, alpha=0.2, lw=0)
    ax.plot(x[ok], m[ok], color=color, lw=1.0, **kw)


def fig_retargeting(so, stem, size_in, n_ep=150, smooth=10):
    """d: behaviour retargets to the new reward at once, model vs mice."""
    curves = _success_curves()
    k = np.ones(smooth) / smooth
    sm = np.array([np.convolve(c, k, mode="valid") for c in curves])
    x = np.arange(n_ep) + smooth
    fig, (axm, axd) = plt.subplots(1, 2, figsize=size_in)
    _band(axm, x, sm[:, :n_ep], "#9A9A9A", label="First learning (A)")
    _band(axm, x, sm[:, 1000:1000 + n_ep], COLOR_MODEL, label="After relocation (B)")
    axm.set_xlim(0, n_ep + smooth)
    axm.set_ylim(0, 0.55)
    axm.set_xlabel("Episodes since reward onset")
    axm.set_ylabel("Success rate")
    axm.legend(loc="upper left", fontsize=FS_SMALL, handlelength=1.2, borderaxespad=0.2)
    title(axm, f"Model (n = {len(curves)})")

    lick, rows = _lick_preference(so)
    t = np.arange(lick.shape[1])
    _band(axd, t, lick, COLOR_DATA)
    axd.axhline(0, color=C["muted"], lw=0.5, ls=(0, (2.5, 2)))
    lat = np.mean([r["lick_latency"] for r in rows if r["lick_latency"] is not None])
    axd.axvline(lat, color=COLOR_DATA, lw=0.6, ls=":")
    axd.text(lat + 0.6, -0.9, f"{lat:.1f} trials", fontsize=FS_SMALL, color="#333333", va="bottom")
    axd.set_xlim(0, t[-1])
    axd.set_ylim(-1, 1)
    axd.set_xlabel("Trials since reward switch")
    axd.set_ylabel("Lick preference, new zone")
    title(axd, f"Mice (n = {len(rows)} sessions)")
    fig.subplots_adjust(left=0.42 / size_in[0], right=1 - 0.06 / size_in[0],
                        bottom=0.50 / size_in[1], top=1 - 0.30 / size_in[1], wspace=0.55)
    return save_figure(fig, stem, exact_size=True)


# ---------------------------------------------------------------- assembly
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_show", action="store_true")
    ap.add_argument("--assemble_only", action="store_true",
                    help="recompose from the existing panel PDFs")
    add_plot_only(ap)
    args = ap.parse_args()
    ensure_dir(PANELS_DIR)
    d, kr, so, rr = load()
    om_dir = ensure_dir(PANELS_DIR / "omission")
    rl_dir = ensure_dir(PANELS_DIR / "reloc")
    OL, RL = OMISSION_LAYOUT, RELOC_LAYOUT
    if args.assemble_only:
        figlayout.compose(OL, figlayout.existing_panels(om_dir, OL.keys()), FIGURES_DIR / "figure_omission_raw")
        figlayout.compose(RL, figlayout.existing_panels(rl_dir, RL.keys()), FIGURES_DIR / "figure_relocdata_raw")
        return

    S = OL.slot_size_in
    om = {
        "a": fig_cartoon_omission(om_dir / "panel_a", S("a"))[1],
        "b": fig_omission_model(d, om_dir / "panel_b", S("b"))[1],
        "c": fig_proximity(d, kr, om_dir / "panel_c", S("c"))[1],
        "d": fig_enrichment(d, kr, om_dir / "panel_d", S("d"))[1],
        "e": fig_reliability_change(d, kr, om_dir / "panel_e", S("e"))[1],
    }
    figlayout.compose(OL, om, FIGURES_DIR / "figure_omission_raw")

    S = RL.slot_size_in
    rl = {
        "a": fig_cartoon_reloc(rl_dir / "panel_a", S("a"))[1],
        "b": fig_field_displacement(so, rl_dir / "panel_b", S("b"))[1],
        "c": fig_cell_classes(so, rr, rl_dir / "panel_c", S("c"))[1],
        "d": fig_retargeting(so, rl_dir / "panel_d", S("d"))[1],
    }
    figlayout.compose(RL, rl, FIGURES_DIR / "figure_relocdata_raw")
    print("Done.")


if __name__ == "__main__":
    main()
