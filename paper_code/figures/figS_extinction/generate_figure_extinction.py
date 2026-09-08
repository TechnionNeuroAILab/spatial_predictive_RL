#!/usr/bin/env python3
"""
Supplementary figure S7 — extinction simulation: reward over-representation
collapses (and overshoots) when the reward and the reward-prediction-head
objective are switched off, tested on the frozen-replay triple; confronted
against Krishnan et al. 2022's published source data.

Reads Repro/aggregate/extinction_summary.json (paper_code/aggregate/run_extinction_analysis.py;
13 entropy-gated seeds x frozen replay triple) and
external_data/krishnan2022/cache/krishnan_data.json (published source data).

Panels:
  a  K1 over-representation: per-seed mean distance-to-A across the triple
     (pred -> postLearn -> postExt), paired lines; delta_learn/delta_ext/
     recovery-fraction annotated. Model predicts NOT JUST recovery but
     overshoot past the pre-learning baseline (recovery fraction > 1).
  b  K2 reliability analogue: split-half rate-map correlation, postLearn vs
     postExt, split by near-A/far-A strata (n=13 paired). The model's drop is
     significant overall but concentrated in FAR cells, not near cells --
     flagged explicitly as a divergence from Krishnan's zone-specific pattern.
  c  K3 dopamine proxy: same (postLearn) head evaluated on postLearn vs
     postExt replay activity -- the readout-vs-map-drift analogue of an
     unchanged DA circuit reporting a degraded map. Group-mean profile +-SD,
     n=13, median near/far contrast collapse annotated (mean is heavy-tailed).
  d  Side-by-side model vs Krishnan: reliability drop magnitude and DA/proxy
     contrast collapse, so the confrontation is not just qualitative.

Usage:
    python code_analysis/figure_extinction/generate_figure_extinction.py --no_show
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
from pypdf import PdfReader, PdfWriter, Transformation

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

from utils import (  # noqa: E402
    COLOR_DATA, COLOR_MODEL, COLOR_POSTEXT, COLOR_POSTLEARN, COLOR_PRED,
    MM_TO_PT, PAGE_W, apply_nature_style, ensure_dir, save_figure,
)

apply_nature_style()
CACHE = REPO / "Repro" / "aggregate" / "extinction_summary.json"
KRISHNAN_CACHE = REPO / "external_data" / "krishnan2022" / "cache" / "krishnan_data.json"
FIGURES_DIR = HERE / "figures_supp"
PERIOD_LABELS = ["pre-learning\n(pred)", "learned\n(postLearn)", "extinguished\n(postExt)"]


def _fmt_p(p):
    if p is None:
        return "p=n/a"
    return f"p={p:.1e}" if p < 0.001 else f"p={p:.3f}"


def load():
    with open(CACHE) as f:
        d = json.load(f)
    with open(KRISHNAN_CACHE) as f:
        kr = json.load(f)
    return d, kr


def fig_k1(d, stem):
    rows = d["per_seed"]
    k1 = d["k1"]
    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    x = np.arange(3)
    for r in rows:
        md = r["k1"]["mean_dist"]
        ax.plot(x, md, color="0.7", lw=0.7, alpha=0.6, zorder=2)
        ax.scatter(x, md, color="0.7", s=6, alpha=0.6, zorder=2)
    mp = k1["mean_dist_by_period"]
    means = [mp[p]["mean"] for p in ("pred", "postLearn", "postExt")]
    sds = [mp[p]["sd"] for p in ("pred", "postLearn", "postExt")]
    colors = [COLOR_PRED, COLOR_POSTLEARN, COLOR_POSTEXT]
    for xi, m, s, c in zip(x, means, sds, colors):
        ax.errorbar([xi], [m], yerr=[s], fmt="D", color=c, markersize=5, capsize=3,
                    lw=1.3, zorder=5, markeredgecolor="black", markeredgewidth=0.4)
    ax.plot(x, means, color="black", lw=1.2, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(PERIOD_LABELS, fontsize=5.5)
    ax.set_ylabel("mean peak distance to reward A")
    dl, de, rec = k1["delta_learn"], k1["delta_ext"], k1["recovery_pooled"]
    ax.set_title(
        f"n={k1['n_seeds']} seeds: learn {dl['mean']:+.3f} ({_fmt_p(dl['p'])}), "
        f"extinguish {de['mean']:+.3f} ({_fmt_p(de['p'])})\n"
        f"recovery frac. {rec['point']:.2f} [{rec['ci_lo']:.2f},{rec['ci_hi']:.2f}] "
        f"(>1 = overshoot past baseline)", fontsize=5.2)
    fig.tight_layout(pad=0.4)
    return save_figure(fig, stem)


def fig_k2(d, stem):
    k2 = d["k2"]
    fig, ax = plt.subplots(figsize=(2.9, 2.6))
    strata = ["all", "near", "far"]
    x = np.arange(len(strata))
    w = 0.32
    for off, period, c in [(-w / 2, "postLearn", COLOR_POSTLEARN), (w / 2, "postExt", COLOR_POSTEXT)]:
        means = [k2[s][period]["mean"] for s in strata]
        sds = [k2[s][period]["sd"] or 0 for s in strata]
        ax.bar(x + off, means, width=w, yerr=sds, capsize=2.5, color=c,
               label=period, edgecolor="black", linewidth=0.4)
    for i, s in enumerate(strata):
        p = k2[s]["postLearn_vs_postExt"]["p"]
        ymax = max(k2[s]["postLearn"]["mean"], k2[s]["postExt"]["mean"]) + 0.13
        ax.annotate(_fmt_p(p), (i, ymax), ha="center", fontsize=5,
                    color="crimson" if (p is not None and p < 0.05) else "0.4")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n(n={k2['n_seeds']})" for s in strata], fontsize=5.5)
    ax.set_ylabel("split-half reliability (r)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=5.5, frameon=False, loc="lower left")
    ax.set_title("Reliability drop is concentrated in FAR cells\n(divergence from Krishnan's zone-specific drop)",
                 fontsize=5.6)
    fig.tight_layout(pad=0.4)
    return save_figure(fig, stem)


def fig_k3(d, stem):
    k3 = d["k3"]
    fig, ax = plt.subplots(figsize=(2.9, 2.6))
    c = np.array(k3["dist_bin_centres"])
    mp = k3["mean_profile"]
    for label, col in [("postLearn", COLOR_POSTLEARN), ("postExt", COLOR_POSTEXT)]:
        m = np.array(mp[label])
        s = np.array(mp[f"{label}_sd"])
        ax.plot(c, m, color=col, lw=1.3, marker="o", markersize=3, label=label)
        ax.fill_between(c, m - s, m + s, color=col, alpha=0.15, linewidth=0)
    ax.plot(c, k3["target_profile"], color="0.4", lw=0.9, ls="--", label="analytic target\n(same both periods)")
    ax.set_xlabel("distance to reward A")
    ax.set_ylabel("head output (proximity readout)")
    cc = k3["contrast_near_over_far"]
    ax.set_title(
        f"n={k3['n_seeds']} seeds: near/far contrast collapses\n"
        f"median {cc['postLearn']['median']:.1f}x -> {cc['postExt']['median']:.1f}x "
        f"({_fmt_p(cc['postLearn_vs_postExt']['p'])})", fontsize=5.6)
    ax.legend(fontsize=5, frameon=False)
    fig.tight_layout(pad=0.4)
    return save_figure(fig, stem)


def fig_confrontation(d, kr, stem):
    """Model (n=13 seeds) vs Krishnan (n=3 imaged mice, published source data)
    side by side, on two INDEPENDENTLY-derived fractional-change metrics:
    (1) split-half reliability drop; (2) fraction of the near-reward
    field-density enrichment above its uniform/pre-learning baseline that is
    lost under extinction (model: postLearn->postExt near-A fraction relative
    to the pred baseline; Krishnan: their own enrichment-vs-uniform ratio,
    rewarded->re_low, sheet 'Figure3'). Krishnan values are Tier-1 published
    source data, never reanalysed raw data."""
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.4))

    ax = axes[0]
    model_all = d["k2"]["all"]
    model_drop = (model_all["postExt"]["mean"] - model_all["postLearn"]["mean"]) / model_all["postLearn"]["mean"]
    kr_drop = kr["reliability"]["frac_drop_group_mean"]
    ax.bar([0, 1], [model_drop * 100, kr_drop * 100], color=[COLOR_MODEL, COLOR_DATA],
           edgecolor="black", linewidth=0.4, width=0.55)
    ax.axhline(0, color="0.3", lw=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"model\n(n={d['k2']['n_seeds']} seeds)",
                        f"Krishnan 2022\n(n={kr['reliability']['n_animals']} mice)"], fontsize=5.6)
    ax.set_ylabel("reliability change (%)")
    ax.set_title("Reliability drop under extinction", fontsize=5.8)

    ax = axes[1]
    nf = d["k1"]["near_frac_by_period"]
    pred_, learn_, ext_ = nf["pred"]["mean"], nf["postLearn"]["mean"], nf["postExt"]["mean"]
    model_frac_lost = (learn_ - ext_) / (learn_ - pred_)
    fd = kr["field_density"]["by_condition_counts"]
    e_rew = fd["rewarded"]["enrichment_vs_uniform"]["end"]
    e_low = fd["re_low"]["enrichment_vs_uniform"]["end"]
    kr_frac_lost = (e_rew - e_low) / (e_rew - 1.0)
    ax.bar([0, 1], [model_frac_lost * 100, kr_frac_lost * 100], color=[COLOR_MODEL, COLOR_DATA],
           edgecolor="black", linewidth=0.4, width=0.55)
    ax.axhline(100, color="0.3", lw=0.6, ls="--")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["model\n(near-A fraction)", "Krishnan 2022\n(field-density\nenrichment)"], fontsize=5.6)
    ax.set_ylabel("% of reward-related\nenrichment lost")
    ax.set_title(f"Fraction of enrichment lost\n(model {model_frac_lost*100:.0f}%: full loss + overshoot;\n"
                 f"Krishnan {kr_frac_lost*100:.0f}%: substantial, partial)", fontsize=5.4)
    fig.tight_layout(pad=0.4, w_pad=1.2)
    return save_figure(fig, stem)


def _pdf_to_png(pdf, png, dpi=300):
    with tempfile.TemporaryDirectory() as tmp:
        pre = Path(tmp) / "p"
        subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-singlefile", str(pdf), str(pre)], check=True)
        copy2(pre.with_suffix(".png"), png)


def assemble(pdfs, out_stem, dpi=300):
    out_stem = Path(out_stem).with_suffix("")
    M_, G = 10 * MM_TO_PT, 5 * MM_TO_PT
    rows = [[k for k in ("a", "b") if k in pdfs], [k for k in ("c", "d") if k in pdfs]]
    rows = [r for r in rows if r]
    sizes = {k: (lambda b: (float(b.width), float(b.height)))(PdfReader(str(pdfs[k])).pages[0].mediabox)
             for r in rows for k in r}
    heights = []
    for r in rows:
        cw = (PAGE_W - 2 * M_ - (len(r) - 1) * G) / len(r)
        heights.append(max(sizes[k][1] * cw / sizes[k][0] for k in r))
    page_h = 2 * M_ + sum(heights) + (len(rows) - 1) * G
    writer = PdfWriter()
    page = writer.add_blank_page(width=PAGE_W, height=page_h)
    y = M_
    for r, rh in zip(reversed(rows), reversed(heights)):
        cw = (PAGE_W - 2 * M_ - (len(r) - 1) * G) / len(r)
        for i, k in enumerate(r):
            iw, ih = sizes[k]
            sc = min(cw / iw, rh / ih)
            left = M_ + i * (cw + G) + (cw - iw * sc) / 2
            page.merge_transformed_page(PdfReader(str(pdfs[k])).pages[0],
                                        Transformation().scale(sc).translate(left, y + (rh - ih * sc) / 2),
                                        over=True)
        y += rh + G
    pdf_path = out_stem.with_suffix(".pdf")
    with open(pdf_path, "wb") as f:
        writer.write(f)
    _pdf_to_png(pdf_path, out_stem.with_suffix(".png"), dpi)
    print(f"  Saved: {pdf_path}\n  Saved: {out_stem.with_suffix('.png')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_show", action="store_true")
    args = ap.parse_args()
    ensure_dir(FIGURES_DIR / "panels")
    d, kr = load()
    print(f"loaded {d['k1']['n_seeds']} entropy-gated extinction seeds")

    pdfs = {}
    print("Panel a (K1 over-representation)…"); _, pdfs["a"] = fig_k1(d, FIGURES_DIR / "panels" / "panel_a")
    print("Panel b (K2 reliability)…");          _, pdfs["b"] = fig_k2(d, FIGURES_DIR / "panels" / "panel_b")
    print("Panel c (K3 dopamine proxy)…");        _, pdfs["c"] = fig_k3(d, FIGURES_DIR / "panels" / "panel_c")
    print("Panel d (model vs Krishnan)…");        _, pdfs["d"] = fig_confrontation(d, kr, FIGURES_DIR / "panels" / "panel_d")
    print("Assembling…"); assemble(pdfs, FIGURES_DIR / "figure_extinction_supp_raw")
    print("Done.")


if __name__ == "__main__":
    main()
