#!/usr/bin/env python3
"""Supplementary figure: CA1 reward-relocation reanalysis (Sosa, Plitt &
Giocomo 2025, DANDI 001361).

Reads data/sosa2025/cache/sosa_tests.json (code/external/sosa2025/analysis_sosa.py);
writes
figures_supp/figure_sosa_supp_raw.{pdf,png}.

Panels:
  a  example session: lick rate by position and trial, reward zones marked
  b  field peak before vs after the switch for one session: place-anchored
     cells lie on the identity line, reward-relative cells on the line offset
     by the zone shift
  c  reward-relative fraction under the full criterion and under the
     distance-only criterion, per session, with the model mean +/- s.e.m.
  d  per-session lick preference and map fraction after the switch
  e  lick vs map retargeting latency per session (map latency cannot be
     detected below five trials)
  f  raw vs occupancy-matched peak shift per session (negative control: laps
     on a 1D track already repeat the same route)

Usage:
    python code/figures/figS_sosa/generate_figure_sosa.py --no_show
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CACHE = REPO / "data" / "sosa2025" / "cache" / "sosa_tests.json"
RR_CACHE = REPO / "results" / "multi_seed_reloc" / "aggregate" / "reward_relative_summary.json"
OUT = HERE / "figures_supp"
sys.path.insert(0, str(REPO / "code" / "lib"))

import figlayout  # noqa: E402
from analysis_helpers import campaign_passers  # noqa: E402
from pubstyle import (  # noqa: E402
    C, FS_SMALL, FS_TICK, apply_nature_style, ensure_dir, fmt_p, mean_sem_marker,
    save_figure, sem, title, zero_line,
)

apply_nature_style()

COLOR_PRE_ZONE, COLOR_POST_ZONE = C["reward_a"], C["reward_b"]
TRACK = 450.0
SCATTER_SESSION = ("m11", 12)

LAYOUT = figlayout.Layout(
    [("a", 0, 0, 2), ("b", 0, 2, 2), ("c", 0, 4, 2),
     ("d", 1, 0, 2), ("e", 1, 2, 2), ("f", 1, 4, 2)],
    ncols=6, row_h_in=[1.95, 1.95], caption_lines=11, name="figS_sosa",
)


def _rows(so):
    return [r for r in so["per_session"] if "skipped" not in r]


def panel_raster(ex, stem, size_in):
    lm = np.array(ex["lick_matrix"], dtype=float)
    lm[lm < 0] = np.nan
    n_trials = lm.shape[0]
    bw = ex["bin_centers"][1] - ex["bin_centers"][0]
    track_len = ex["bin_centers"][-1] + bw / 2.0
    vmax = np.nanpercentile(lm, 98)
    cmap = plt.get_cmap("Greys").copy()
    cmap.set_bad("#f2f2f2")
    fig, ax = plt.subplots(figsize=size_in)
    im = ax.imshow(lm, aspect="auto", origin="upper", cmap=cmap, vmin=0, vmax=vmax,
                   extent=(0, track_len, n_trials, 0), interpolation="nearest")
    sw = ex["switch_ordinal"]
    for z, col, y0, y1 in ((ex["zone_pre"], COLOR_PRE_ZONE, 0, sw),
                           (ex["zone_post"], COLOR_POST_ZONE, sw, n_trials)):
        ax.add_patch(Rectangle((z[0], y0), z[1] - z[0], y1 - y0, facecolor=col,
                               edgecolor="none", alpha=0.35, zorder=3))
    ax.axhline(sw, color="k", lw=0.6, ls="--")
    ax.text(track_len * 0.99, sw - 1.0, "switch", ha="right", va="bottom", fontsize=FS_SMALL)
    ax.set_xlabel("Position (cm)")
    ax.set_ylabel("Trial")
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_ticks([0, int(vmax)])
    cb.set_label("Lick rate (Hz)", fontsize=FS_SMALL)
    cb.ax.tick_params(labelsize=FS_SMALL, width=0.4, length=1.5)
    cb.outline.set_visible(False)
    title(ax, f"Licking, {ex['subject']} session {ex['session']}")
    fig.tight_layout(pad=0.35)
    return save_figure(fig, stem, exact_size=True)


def panel_field_scatter(so, stem, size_in):
    r = next(r for r in _rows(so) if (r["subject"], r["session"]) == SCATTER_SESSION)
    c = r["cells"]
    pre, post = np.array(c["pk_pre"]), np.array(c["pk_post"])
    rr, pa = np.array(c["is_rr"], bool), np.array(c["is_anchored"], bool)
    shift = (c["zone_post_start"] - c["zone_pre_start"]) % TRACK
    fig, ax = plt.subplots(figsize=size_in)
    other = ~(rr | pa)
    ax.scatter(pre[other], post[other], s=4, color="#C8C8C8", lw=0, label="Other")
    ax.scatter(pre[pa], post[pa], s=5, color=C["data"], lw=0, label="Place-anchored")
    ax.scatter(pre[rr], post[rr], s=7, color=COLOR_POST_ZONE, lw=0, label="Reward-relative")
    xs = np.linspace(0, TRACK, 2)
    ax.plot(xs, xs, color="#777777", lw=0.5, ls="--")
    for k in (0, -1):                           # the offset line wraps around the track
        ax.plot(xs, xs + shift + k * TRACK, color=COLOR_POST_ZONE, lw=0.5, ls="--")
    for z, col, axis in ((c["zone_pre_start"], COLOR_PRE_ZONE, "x"), (c["zone_post_start"], COLOR_POST_ZONE, "y")):
        span = (z, z + 50)
        (ax.axvspan if axis == "x" else ax.axhspan)(*span, color=col, alpha=0.18, lw=0, zorder=0)
    ax.set_xlim(0, TRACK); ax.set_ylim(0, TRACK)
    ax.set_aspect("equal")
    ax.set_xticks([0, 150, 300, 450]); ax.set_yticks([0, 150, 300, 450])
    ax.set_xlabel("Field peak before switch (cm)")
    ax.set_ylabel("Field peak after switch (cm)")
    ax.legend(loc="upper left", fontsize=FS_SMALL, handletextpad=0.1, borderaxespad=0.1,
              markerscale=1.5, framealpha=0.85, frameon=True, edgecolor="none")
    title(ax, f"Fields across the switch (n = {len(pre)})")
    fig.tight_layout(pad=0.35)
    return save_figure(fig, stem, exact_size=True)


def panel_classifier(so, rr_cache, stem, size_in):
    rows = _rows(so)
    full = 100 * np.array([r["rr_frac"] for r in rows])
    dist = 100 * np.array([r["rr_frac_50cm_only"] for r in rows])
    keep = campaign_passers("reloc")
    model = 100 * np.array([s["controlled"]["frac_reward_relative"] for s in rr_cache["per_seed"]
                            if int(s["seed"]) in keep])
    fig, ax = plt.subplots(figsize=size_in)
    m, e = model.mean(), sem(model)
    ax.axhspan(m - e, m + e, color=C["model"], alpha=0.2, lw=0)
    ax.axhline(m, color=C["model"], lw=0.8)
    ax.text(1.55, m + e + 0.5, f"model {m:.1f} ± {e:.1f}%", color=C["model"],
            fontsize=FS_SMALL, ha="right", va="bottom")
    for u, v in zip(full, dist):
        ax.plot([0, 1], [u, v], color="#BBBBBB", lw=0.4, zorder=1)
    for x, v in ((0, full), (1, dist)):
        ax.scatter([x] * len(v), v, s=8, color=C["data"], lw=0, zorder=2)
        mean_sem_marker(ax, x, v)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Distance and\nshuffle criteria", "Distance\nonly"])
    ax.set_xlim(-0.5, 1.6)
    ax.set_ylim(0, None)
    ax.set_ylabel("Reward-relative cells (%)")
    title(ax, f"Classifier dependence (n = {len(rows)})")
    fig.tight_layout(pad=0.35)
    return save_figure(fig, stem, exact_size=True)


def panel_session_traces(so, stem, size_in, max_t=40):
    rows = _rows(so)
    fig, axes = plt.subplots(2, 1, figsize=size_in, sharex=True)
    for r in rows:
        sw = r["lick_rates_post"][0]["trial"]
        t, v = [], []
        for e in r["lick_rates_post"]:
            tot = e["rate_new"] + e["rate_old"]
            if tot > 0 and e["trial"] - sw < max_t:
                t.append(e["trial"] - sw); v.append((e["rate_new"] - e["rate_old"]) / tot)
        axes[0].plot(t, v, color=C["data"], lw=0.4, alpha=0.45)
        ref = r.get("map_pre_ref_frac") or np.nan
        tt = [e["trial_end"] - sw for e in r["map_fracs_post"] if e["trial_end"] - sw < max_t]
        vv = [e["frac"] / ref for e in r["map_fracs_post"] if e["trial_end"] - sw < max_t]
        axes[1].plot(tt, vv, color=C["data"], lw=0.5, alpha=0.6)
        if r["lick_latency"] is not None:
            axes[0].plot(r["lick_latency"], 1.12, "v", color=C["data"], ms=2.5)
        if r["map_latency"] is not None:
            axes[1].plot(r["map_latency"], 1.0, "o", color=C["data"], ms=2.5, mfc="white", mew=0.6, zorder=4)
    axes[0].axhline(0, color=C["muted"], lw=0.5, ls=(0, (2.5, 2)))
    axes[1].axhline(1, color=C["muted"], lw=0.5, ls=(0, (2.5, 2)))
    axes[0].set_ylabel("Lick preference", fontsize=FS_SMALL)
    axes[1].set_ylabel("Map fraction\n(rel. pre-switch)", fontsize=FS_SMALL)
    axes[1].set_xlabel("Trials since switch")
    axes[0].set_xlim(0, max_t)
    axes[0].set_ylim(-1.15, 1.45)
    title(axes[0], "Each session")
    fig.tight_layout(pad=0.35, h_pad=0.3)
    return save_figure(fig, stem, exact_size=True)


def panel_occupancy(so, stem, size_in):
    rows = _rows(so)
    raw = np.array([r["delta_raw"] for r in rows])
    mat = np.array([r["delta_matched"] for r in rows])
    fig, ax = plt.subplots(figsize=size_in)
    for u, v in zip(raw, mat):
        ax.plot([0, 1], [u, v], color="#BBBBBB", lw=0.5, zorder=1)
    for x, v, col in ((0, raw, C["muted"]), (1, mat, C["data"])):
        ax.scatter([x] * len(v), v, s=10, color=col, lw=0, zorder=2)
        mean_sem_marker(ax, x, v)
    zero_line(ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Raw", "Occupancy-\nmatched"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("Peak shift toward new zone")
    p = so["summary"]["wilcoxon_raw_vs_matched"]["p"]
    ax.text(0.5, 0.97, f"paired {fmt_p(p)}", transform=ax.transAxes, ha="center", va="top",
            fontsize=FS_SMALL, color="#333333")
    title(ax, f"Occupancy control (n = {len(rows)})")
    fig.tight_layout(pad=0.35)
    return save_figure(fig, stem, exact_size=True)


def panel_latency(so, stem, size_in):
    """Lick vs map retargeting latency per session."""
    pairs = [(r["lick_latency"], r["map_latency"]) for r in _rows(so) if r["lick_latency"] is not None]
    lim = 1 + max([p[1] or 0 for p in pairs] + [p[0] for p in pairs] + [15])
    fig, ax = plt.subplots(figsize=size_in)
    ax.plot([0, lim], [0, lim], color="#999999", lw=0.6, ls="--", zorder=1)
    ax.axhspan(0, 5, color="0.92", zorder=0)
    ax.text(lim * 0.97, 2.5, "below detection floor", fontsize=FS_SMALL, ha="right",
            va="center", color="0.45")
    rng = np.random.default_rng(5)
    for lick, mapl in pairs:
        if mapl is None:
            ax.plot(lick, lim - 0.6, marker="^", ms=4, color=C["data"], mec="white", mew=0.4, zorder=3)
            continue
        jx, jy = rng.uniform(-0.15, 0.15, 2)
        ax.plot(lick + jx, mapl + jy, "o", ms=4, color=C["data"], mec="white", mew=0.4, zorder=3)
    p = so["summary"]["wilcoxon_lick_vs_map_latency"]["p"]
    win = so["summary"]["prediction4_lick_before_map"]
    ax.text(0.97, 0.30, f"licking earlier in {win['n_lick_faster']}/{win['n_pairs']}\n{fmt_p(p)}",
            transform=ax.transAxes, fontsize=FS_SMALL, ha="right", va="bottom", color="#333333")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Lick latency (trials)")
    ax.set_ylabel("Map latency (trials)")
    title(ax, "Latency per session")
    fig.tight_layout(pad=0.35)
    return save_figure(fig, stem, exact_size=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_show", action="store_true")
    ap.add_argument("--plot-only", action="store_true", help="accepted for regenerate_all")
    ap.parse_args()
    so = json.loads(CACHE.read_text())
    rr_cache = json.loads(RR_CACHE.read_text())
    panels = ensure_dir(OUT / "panels")
    S = LAYOUT.slot_size_in
    pdfs = {
        "a": panel_raster(so["example_session"], panels / "panel_a", S("a"))[1],
        "b": panel_field_scatter(so, panels / "panel_b", S("b"))[1],
        "c": panel_classifier(so, rr_cache, panels / "panel_c", S("c"))[1],
        "d": panel_session_traces(so, panels / "panel_d", S("d"))[1],
        "e": panel_latency(so, panels / "panel_e", S("e"))[1],
        "f": panel_occupancy(so, panels / "panel_f", S("f"))[1],
    }
    figlayout.compose(LAYOUT, pdfs, OUT / "figure_sosa_supp_raw")
    print("Done.")


if __name__ == "__main__":
    main()
