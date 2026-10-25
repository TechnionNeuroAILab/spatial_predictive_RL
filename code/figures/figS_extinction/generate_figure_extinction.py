#!/usr/bin/env python3
"""
Supplementary figure: reward omission, network by network, and the time course
of behaviour and map in model and mice.

Panels (51 retained omission networks, analysis_helpers.campaign_passers("extinction")):
  a  output of the proximity decoder (weights fixed after learning) against
     distance to the former reward, learned vs withdrawn, with its target
  b  split-half map reliability on the replayed trajectory for all cells, cells
     near the former reward and cells far from it, through baseline, learned
     and withdrawn weights (mean +/- s.e.m.)
  c  model behaviour across withdrawal: arrivals at the former reward per
     episode
  d  mice across extinction laps: anticipatory licking and position decoding
     (Krishnan et al. 2022, Fig. 1E source data)

Usage:
    python code/figures/figS_extinction/generate_figure_extinction.py --no_show
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

from utils import (  # noqa: E402
    COLOR_DATA, COLOR_MODEL, COLOR_POSTEXT, COLOR_POSTLEARN, COLOR_PRED,
    FS_SMALL, title, apply_nature_style, ensure_dir, save_figure,
)
import figlayout  # noqa: E402
from figure_cache import add_plot_only  # noqa: E402

sys.path.insert(0, str(REPO / "code" / "lib"))
from analysis_helpers import campaign_passers  # noqa: E402
from pubstyle import C, sem  # noqa: E402

apply_nature_style()

LAYOUT = figlayout.Layout(
    [("a", 0, 0, 3), ("b", 0, 3, 3),
     ("c", 1, 0, 3), ("d", 1, 3, 3)],
    ncols=6, row_h_in=[1.95, 1.85], caption_lines=9,
    name="figS_extinction",
)
CACHE = REPO / "results" / "aggregate" / "extinction_summary.json"
KRISHNAN_CACHE = REPO / "data" / "krishnan2022" / "cache" / "krishnan_data.json"
FIGURES_DIR = HERE / "figures_supp"
PERIODS = ["Baseline", "Learned", "Withdrawn"]


def load():
    return json.loads(CACHE.read_text()), json.loads(KRISHNAN_CACHE.read_text())


def fig_reliability(d, stem, size_in):
    """a: reliability for all / near / far cells across the replay triple."""
    fig, ax = plt.subplots(figsize=size_in)
    x = np.arange(3)
    for stratum, col, lbl in (("all", "#1A1A1A", "All cells"),
                              ("near", C["reward_a"], "Near former reward"),
                              ("far", "#9A9A9A", "Far from reward")):
        v = np.array([[r["k2"][p][stratum] for p in ("pred", "postLearn", "postExt")]
                      for r in d["per_seed"]])
        ax.errorbar(x, v.mean(0), yerr=[sem(c) for c in v.T], fmt="o-", color=col, ms=3,
                    lw=1.0, capsize=2, elinewidth=0.8, label=lbl)
    ax.set_xticks(x)
    ax.set_xticklabels(PERIODS)
    ax.set_xlim(-0.3, 2.3)
    ax.set_ylabel("Split-half reliability")
    ax.legend(loc="lower left", fontsize=FS_SMALL)
    p = d["k2"]["far_drop_vs_near_drop"]["p"]
    from pubstyle import fmt_p
    ax.text(0.98, 0.97, f"far vs near drop\n{fmt_p(p)}", transform=ax.transAxes, ha="right",
            va="top", fontsize=FS_SMALL, color="#333333")
    title(ax, f"Map reliability (n = {d['k1']['n_seeds']})")
    fig.tight_layout(pad=0.35)
    return save_figure(fig, stem, exact_size=True)


def fig_decoder(d, stem, size_in):
    """b: proximity decoder output vs distance to the former reward."""
    k3 = d["k3"]
    xs = np.asarray(k3["dist_bin_centres"])
    rows = [r for r in d["per_seed"] if r.get("k3")]
    fig, ax = plt.subplots(figsize=size_in)
    for key, col, lbl in (("postLearn", COLOR_POSTLEARN, "Learned"),
                          ("postExt", COLOR_POSTEXT, "Withdrawn")):
        v = np.array([r["k3"][key] for r in rows])
        m, e = v.mean(0), np.array([sem(c) for c in v.T])
        ax.fill_between(xs, m - e, m + e, color=col, alpha=0.2, lw=0)
        ax.plot(xs, m, "o-", color=col, ms=2.5, lw=1.0, label=lbl)
    ax.plot(xs, k3["target_profile"], color="#555555", lw=0.8, ls=(0, (3, 2)), label="Teaching target")
    ax.set_xlabel("Distance to former reward")
    ax.set_ylabel("Decoder output")
    ax.legend(loc="upper right", fontsize=FS_SMALL)
    title(ax, f"Proximity decoder, weights fixed (n = {len(rows)})")
    fig.tight_layout(pad=0.35)
    return save_figure(fig, stem, exact_size=True)


def fig_model_behaviour(stem, size_in, before=100, after=300, smooth=10):
    """c: arrivals at the former reward per episode around withdrawal."""
    curves = []
    for s in sorted(campaign_passers("extinction")):
        z = np.load(REPO / "results" / "multi_seed_ext" / f"ext_seed{s}" / "learning_curves.npz")
        curves.append(z["rl_success"][1000 - before:1000 + after])
    v = np.array(curves)
    k = np.ones(smooth) / smooth
    v = np.array([np.convolve(c, k, mode="same") for c in v])
    x = np.arange(-before, after)
    keep = (x >= -before + smooth) & (x < after - smooth)
    m, e = v.mean(0), v.std(0, ddof=1) / np.sqrt(len(v))
    fig, ax = plt.subplots(figsize=size_in)
    ax.fill_between(x[keep], (m - e)[keep], (m + e)[keep], color=COLOR_MODEL, alpha=0.2, lw=0)
    ax.plot(x[keep], m[keep], color=COLOR_MODEL, lw=1.0)
    ax.axvline(0, color="#333333", lw=0.6, ls="--")
    ax.text(3, 0.97, "reward and teaching\nwithdrawn", transform=ax.get_xaxis_transform(),
            fontsize=FS_SMALL, va="top", color="#333333")
    ax.set_xlabel("Episode relative to withdrawal")
    ax.set_ylabel("Arrivals at former reward")
    ax.set_ylim(0, None)
    title(ax, f"Model behaviour (n = {len(v)})")
    fig.tight_layout(pad=0.35)
    return save_figure(fig, stem, exact_size=True)


def fig_mouse_laps(kr, stem, size_in):
    """d: licking and position decoding across extinction laps (source data)."""
    lc = kr["extended"]["lap_course"]
    lap = np.asarray(lc["lap"])
    fig, ax = plt.subplots(figsize=size_in)
    for key, col, lbl in (("lick", COLOR_DATA, "Anticipatory licking (norm.)"),
                          ("decoder_r2", "#8C8C8C", "Position decoding, R²")):
        m, e = np.asarray(lc[key]), np.asarray(lc[f"{key}_sem"])
        ax.fill_between(lap, m - e, m + e, color=col, alpha=0.18, lw=0)
        ax.plot(lap, m, "o-", color=col, ms=2.5, lw=1.0, label=lbl)
    ax.set_xlabel("Extinction lap")
    ax.set_ylabel("Value")
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_ylim(0, 1.08)
    ax.legend(loc="lower left", bbox_to_anchor=(0.28, 0.22), fontsize=FS_SMALL)
    title(ax, "Mice (Krishnan et al. 2022)")
    fig.tight_layout(pad=0.35)
    return save_figure(fig, stem, exact_size=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_show", action="store_true")
    add_plot_only(ap)
    ap.parse_args()
    d, kr = load()
    pan = ensure_dir(FIGURES_DIR / "panels")
    S = LAYOUT.slot_size_in
    pdfs = {
        "a": fig_decoder(d, pan / "panel_a", S("a"))[1],
        "b": fig_reliability(d, pan / "panel_b", S("b"))[1],
        "c": fig_model_behaviour(pan / "panel_c", S("c"))[1],
        "d": fig_mouse_laps(kr, pan / "panel_d", S("d"))[1],
    }
    figlayout.compose(LAYOUT, pdfs, FIGURES_DIR / "figure_extinction_supp_raw")
    print("Done.")


if __name__ == "__main__":
    main()
