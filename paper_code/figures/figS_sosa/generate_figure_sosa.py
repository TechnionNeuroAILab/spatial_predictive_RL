"""Supplementary figure: Sosa/Plitt/Giocomo 2025 reanalysis (raw data tests).

Reads ONLY cache/sosa_tests.json (produced by analysis_sosa.py) and writes
figures/figure_sosa_supp_raw.{pdf,png}.

Panels:
  a  example switch session: lick raster (lick rate by position x trial) with
     pre/post reward zones and the switch trial marked.
  b  TEST i: per-session raw vs occupancy-matched shift toward the new zone.
  c  TEST ii: per-session reward-relative fraction vs the model band.
  d  TEST iii: behavioural (lick) vs map-transfer latency, unity line.

Styling follows code_analysis/figure_sr/utils.py (apply_nature_style /
save_figure pattern).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
CACHE = HERE.parents[2] / "external_data" / "sosa2025" / "cache" / "sosa_tests.json"
FIG_STEM = HERE / "figures_supp" / "figure_sosa_supp_raw"

MM_TO_PT = 72.0 / 25.4
PAGE_W_IN = 183.0 / 25.4          # Nature double column, inches

# fixed colour assignments (project palette; teal/orange pair is CVD-safe)
COLOR_PRE_ZONE = "#2a9d8f"        # pre-switch reward zone
COLOR_POST_ZONE = "#e76f51"       # post-switch (new) reward zone
COLOR_RAW = "#555555"
COLOR_MATCHED = "#4c72b0"
COLOR_MODEL_BAND = "#4c72b0"


def apply_nature_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 0.75,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_figure(fig: plt.Figure, stem: Path, dpi: int = 300):
    stem = Path(stem).with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    fig.savefig(png_path, format="png", dpi=dpi, facecolor="white",
                edgecolor="none", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(pdf_path, format="pdf", dpi=dpi, facecolor="white",
                edgecolor="none", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")
    return png_path, pdf_path


def panel_label(ax, letter):
    ax.text(-0.28, 1.08, letter, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top", ha="left")


def session_tag(r):
    return f"{r['subject']} s{r['session']:02d}"


# --------------------------------------------------------------------- panels
def panel_a(ax, ex):
    lm = np.array(ex["lick_matrix"], dtype=float)
    lm[lm < 0] = np.nan                      # -1 encodes missing/flagged
    n_trials, nbins = lm.shape
    track_len = ex["bin_centers"][-1] + (ex["bin_centers"][1]
                                         - ex["bin_centers"][0]) / 2.0
    vmax = np.nanpercentile(lm, 98)
    assert 0 < vmax < 50, "lick rate out of sane range"
    cmap = plt.get_cmap("Greys").copy()
    cmap.set_bad("#f2f2f2")
    im = ax.imshow(lm, aspect="auto", origin="upper", cmap=cmap,
                   vmin=0, vmax=vmax, extent=(0, track_len, n_trials, 0),
                   interpolation="nearest")
    sw = ex["switch_ordinal"]
    for (z, color, y0, y1) in ((ex["zone_pre"], COLOR_PRE_ZONE, 0, sw),
                               (ex["zone_post"], COLOR_POST_ZONE, sw,
                                n_trials)):
        ax.add_patch(Rectangle((z[0], y0), z[1] - z[0], y1 - y0,
                               facecolor=color, edgecolor="none", alpha=0.35,
                               zorder=3))
    ax.axhline(sw, color="k", lw=0.6, ls="--")
    ax.text(track_len * 0.99, sw - 1.0, "switch", ha="right", va="bottom",
            fontsize=5.5)
    ax.text(np.mean(ex["zone_pre"]), 2.2, ex["zone_pre_label"],
            color="#1a6e63", ha="center", va="top", fontsize=6,
            fontweight="bold", zorder=5)
    ax.text(np.mean(ex["zone_post"]), sw + 2.0, ex["zone_post_label"],
            color="#b3401f", ha="center", va="top", fontsize=6,
            fontweight="bold", zorder=5)
    ax.set_xlabel("position (cm)")
    ax.set_ylabel("trial")
    ax.set_title(f"{ex['subject']} ses-{ex['session']:02d} licking", pad=5)
    cax = ax.inset_axes([0.60, 0.865, 0.30, 0.035])
    cb = plt.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([0, int(vmax)])
    cax.set_title("lick rate (Hz)", fontsize=5.5, pad=1.5)
    cb.ax.tick_params(labelsize=5.5, width=0.5, length=1.5)
    cb.outline.set_linewidth(0.5)


def panel_b(ax, rows, summ):
    raw = [r["delta_raw"] for r in rows]
    mat = [r["delta_matched"] for r in rows]
    assert all(abs(v) <= 1 for v in raw + mat)
    x0, x1 = 0.0, 1.0
    for r_v, m_v in zip(raw, mat):
        ax.plot([x0, x1], [r_v, m_v], color="#bbbbbb", lw=0.5, zorder=1)
    ax.plot([x0] * len(raw), raw, "o", ms=3.5, color=COLOR_RAW,
            mec="white", mew=0.4, zorder=2)
    ax.plot([x1] * len(mat), mat, "o", ms=3.5, color=COLOR_MATCHED,
            mec="white", mew=0.4, zorder=2)
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_xticks([x0, x1])
    ax.set_xticklabels(["raw", "occup.-\nmatched"])
    ax.set_xlim(-0.5, 1.5)
    lo = min(min(raw), min(mat))
    hi = max(max(raw), max(mat))
    ax.set_ylim(lo - 0.045, hi + 0.012)
    ax.set_ylabel("peak shift toward new zone\n($-$ = toward, norm.)")
    ap = summ.get("attenuation_pooled")
    p = summ["wilcoxon_raw_vs_matched"]["p"]
    txt = (f"pooled A = {ap:.2f}" if ap is not None
           else "A ill-defined ($\\delta_{raw}\\approx$0)")
    if p is not None:
        txt += f"\nraw vs matched p = {p:.2f}"
    ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=5.5, va="bottom")
    ax.set_title("Test i: peak shift", pad=8)


def panel_c(ax, rows, summ):
    rr = [r["rr_frac"] for r in rows]
    rr50 = [r["rr_frac_50cm_only"] for r in rows]
    assert all(0 <= v <= 1 for v in rr + rr50)
    band_lo = summ["model_rr_frac"] - summ["model_rr_sd"]
    band_hi = summ["model_rr_frac"] + summ["model_rr_sd"]
    ax.axhspan(band_lo, band_hi, color=COLOR_MODEL_BAND, alpha=0.15, lw=0)
    ax.axhline(summ["model_rr_frac"], color=COLOR_MODEL_BAND, lw=0.7, ls="--")
    ax.text(1.42, band_lo - 0.006, "model 6.4$\\pm$3.1%",
            fontsize=5.5, color=COLOR_MODEL_BAND, va="top", ha="right")
    rng = np.random.default_rng(3)
    for x0, vals, filled in ((0.0, rr, True), (1.0, rr50, False)):
        xj = x0 + rng.uniform(-0.09, 0.09, len(vals))
        ax.plot(xj, vals, "o", ms=3.5,
                color="#222222" if filled else "none",
                mec="#222222", mew=0.6, zorder=3)
        ax.plot([x0 - 0.18, x0 + 0.18], [np.mean(vals)] * 2, color="#222222",
                lw=1.1, zorder=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["RR\ncriterion", "50-cm\nonly"])
    ax.set_xlim(-0.5, 1.45)
    ax.set_ylim(0, max(0.32, max(rr50) * 1.15))
    ax.set_ylabel("reward-relative fraction")
    ax.set_title("Test ii: RR fraction", pad=8)


def panel_d(ax, rows, summ):
    pairs = [(r["lick_latency"], r["map_latency"], session_tag(r))
             for r in rows if r["lick_latency"] is not None]
    lim = 1 + max([p[1] or 0 for p in pairs] + [p[0] for p in pairs] + [15])
    ax.plot([0, lim], [0, lim], color="#999999", lw=0.6, ls="--", zorder=1)
    ax.text(lim * 0.62, lim * 0.55, "unity", fontsize=5.5, color="#777777",
            rotation=45, ha="left", va="top")
    n_undef = 0
    rng = np.random.default_rng(5)
    for lick, mapl, tag in pairs:
        if mapl is None:
            n_undef += 1
            ax.plot(lick, lim - 0.5, marker="^", ms=3.5, color="#222222",
                    mec="white", mew=0.4, zorder=3)
            continue
        jx, jy = rng.uniform(-0.15, 0.15, 2)
        ax.plot(lick + jx, mapl + jy, "o", ms=3.5, color="#222222",
                mec="white", mew=0.4, zorder=3)
    if n_undef:
        ax.text(0.35, 0.90, f"$\\blacktriangle$ map never\nreached pre level "
                f"(n={n_undef})", transform=ax.transAxes,
                fontsize=5.5, va="top", ha="left")
    p = summ["wilcoxon_lick_vs_map_latency"]["p"]
    win = summ["prediction4_lick_before_map"]
    ax.text(0.97, 0.04,
            f"lick < map: {win['n_lick_faster']}/{win['n_pairs']}\n"
            + (f"Wilcoxon p = {p:.3f}" if p is not None else ""),
            transform=ax.transAxes, fontsize=5.5, ha="right", va="bottom")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("lick latency\n(trials after switch)")
    ax.set_ylabel("map latency (trials)")
    ax.set_title("Test iii: lick vs map", pad=8)


def main():
    apply_nature_style()
    data = json.loads(CACHE.read_text())
    rows = [r for r in data["per_session"] if "skipped" not in r]
    summ = data["summary"]
    ex = data["example_session"]
    assert ex is not None and len(rows) >= 5

    fig, axes = plt.subplots(1, 4, figsize=(PAGE_W_IN, 2.05),
                             gridspec_kw=dict(wspace=0.75,
                                              width_ratios=[1.55, 1, 1, 1]))
    panel_a(axes[0], ex)
    panel_b(axes[1], rows, summ)
    panel_c(axes[2], rows, summ)
    panel_d(axes[3], rows, summ)
    for ax, letter in zip(axes, "abcd"):
        panel_label(ax, letter)
    save_figure(fig, FIG_STEM)


if __name__ == "__main__":
    main()
