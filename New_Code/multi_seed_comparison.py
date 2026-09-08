"""
multi_seed_comparison.py
Per-seed head vs. no-head field shift comparison.

Loads matched pairs of head / nohead runs:
  <head-root>/matched_head_{lr_tag}_seed{S}/
  <nohead-root>/matched_nohead_{lr_tag}_seed{S}/

Produces:
  <output-dir>/paired_shift_seeds.{png,pdf}   — violin + strip plots per condition
  <output-dir>/per_seed_delta.{png,pdf}        — paired dots (one line per seed)
  <output-dir>/multi_seed_summary.json

Usage:
  python New_Code/multi_seed_comparison.py \\
      --head-root   New_Results_v3 \\
      --nohead-root New_Results_v3 \\
      --lr-tag 1e-4 \\
      --seeds 1 2 7 42 99 123 \\
      --reward-x 0.3 --reward-y 0.3 \\
      --output-dir Analysis_Plots/multi_seed
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon, mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis_helpers import dist_to_reward, has_period, load_period_centers  # noqa: E402
from common import bootstrap_mean_ci, cohen_d_independent, cohen_d_paired, ensure_dir, fdr_bh, save_json  # noqa: E402


# ── data loading ──────────────────────────────────────────────────────────────

def _shift_for_seed(run_dir: str, reward: np.ndarray, thr: float, N: int):
    """Load replay period and return per-cell Δd arrays, or None if unavailable."""
    data_dir = os.path.join(run_dir, "data")
    pre, post = f"replay_c{N}_pred", f"replay_c{N}_postrl"
    if not (has_period(data_dir, pre) and has_period(data_dir, post)):
        return None
    sk_pre, com_pre, peak_pre = load_period_centers(data_dir, pre)
    sk_post, com_post, peak_post = load_period_centers(data_dir, post)
    active = np.where((sk_pre >= thr) & (sk_post >= thr))[0]
    if len(active) == 0:
        return None
    dcom = dist_to_reward(com_post[active], reward) - dist_to_reward(com_pre[active], reward)
    dpeak = dist_to_reward(peak_post[active], reward) - dist_to_reward(peak_pre[active], reward)
    return {
        "n": int(len(active)),
        "dcom": dcom,
        "dpeak": dpeak,
        "com_mean": float(dcom.mean()),
        "peak_mean": float(dpeak.mean()),
        "com_frac_toward": float((dcom < 0).mean()),
        "peak_frac_toward": float((dpeak < 0).mean()),
    }


def load_pairs(
    head_root: str,
    nohead_root: str,
    lr_tag: str,
    seeds: list[int],
    reward: np.ndarray,
    thr: float,
    N: int,
) -> list[dict]:
    records = []
    for seed in seeds:
        h_dir = os.path.join(head_root, f"matched_head_{lr_tag}_seed{seed}")
        nh_dir = os.path.join(nohead_root, f"matched_nohead_{lr_tag}_seed{seed}")

        h_res = _shift_for_seed(h_dir, reward, thr, N) if os.path.isdir(h_dir) else None
        nh_res = _shift_for_seed(nh_dir, reward, thr, N) if os.path.isdir(nh_dir) else None

        if h_res is None:
            print(f"  seed {seed}: SKIP head ({h_dir} missing or no data)")
        if nh_res is None:
            print(f"  seed {seed}: SKIP nohead ({nh_dir} missing or no data)")

        if h_res is not None and nh_res is not None:
            records.append({"seed": seed, "head": h_res, "nohead": nh_res})
            print(
                f"  seed {seed}: "
                f"head  PEAK Δ={h_res['peak_mean']:+.4f} ({100*h_res['peak_frac_toward']:.0f}% toward, n={h_res['n']})  |  "
                f"nohead PEAK Δ={nh_res['peak_mean']:+.4f} ({100*nh_res['peak_frac_toward']:.0f}% toward, n={nh_res['n']})"
            )
    return records


# ── plotting ──────────────────────────────────────────────────────────────────

HEAD_C = "#8055c8"
NOHEAD_C = "#55a8c8"


def _violin_strip(ax, data_head: np.ndarray, data_nohead: np.ndarray,
                  label_head: str, label_nohead: str, ylabel: str, title: str):
    """Violin + strip plot for head vs nohead distributions."""
    parts = ax.violinplot(
        [data_head, data_nohead], positions=[1, 2],
        showmeans=True, showmedians=False, showextrema=True,
    )
    colors = [HEAD_C, NOHEAD_C]
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(colors[i])
        body.set_alpha(0.5)
        body.set_edgecolor("0.25")
    for key in ("cbars", "cmins", "cmaxes", "cmeans"):
        if key in parts:
            parts[key].set_color("0.3")
            parts[key].set_linewidth(1.0)

    rng = np.random.default_rng(0)
    ax.scatter(
        rng.uniform(0.82, 1.18, len(data_head)), data_head,
        s=14, alpha=0.75, color=HEAD_C, zorder=3, edgecolors="none",
    )
    ax.scatter(
        rng.uniform(1.82, 2.18, len(data_nohead)), data_nohead,
        s=14, alpha=0.75, color=NOHEAD_C, zorder=3, edgecolors="none",
    )
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xticks([1, 2])
    ax.set_xticklabels([label_head, label_nohead])
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Annotate with Wilcoxon (within each condition vs 0) and Mann-Whitney (between)
    def _fmt_p(p):
        return f"p={p:.1e}" if p < 0.001 else f"p={p:.3f}"

    def _wil_p(arr):
        arr = arr[np.isfinite(arr)]
        if len(arr) < 5 or np.all(arr == 0):
            return float("nan")
        try:
            return float(wilcoxon(arr)[1])
        except Exception:
            return float("nan")

    ph = _wil_p(data_head)
    pnh = _wil_p(data_nohead)

    ax.text(1, ax.get_ylim()[0] + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
            _fmt_p(ph) if np.isfinite(ph) else "", ha="center", va="bottom",
            fontsize=7, color=HEAD_C)
    ax.text(2, ax.get_ylim()[0] + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
            _fmt_p(pnh) if np.isfinite(pnh) else "", ha="center", va="bottom",
            fontsize=7, color=NOHEAD_C)

    # Between-condition test
    if len(data_head) >= 3 and len(data_nohead) >= 3:
        try:
            _, p_mw = mannwhitneyu(data_head, data_nohead, alternative="two-sided")
            ax.text(1.5, ax.get_ylim()[1] * 0.97, f"MW: {_fmt_p(float(p_mw))}",
                    ha="center", va="top", fontsize=7, color="0.3")
        except Exception:
            pass


def plot_violin_figure(records: list[dict], out_dir: str, lr_tag: str):
    """Violin + strip plots for all seeds pooled."""
    head_peaks = np.concatenate([r["head"]["dpeak"] for r in records])
    nohead_peaks = np.concatenate([r["nohead"]["dpeak"] for r in records])
    head_coms = np.concatenate([r["head"]["dcom"] for r in records])
    nohead_coms = np.concatenate([r["nohead"]["dcom"] for r in records])

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    _violin_strip(
        axes[0], head_peaks, nohead_peaks,
        "Head", "No head",
        "Δ peak distance to reward (post − pre)\n< 0 = toward reward",
        f"Peak shift  (all seeds, lr={lr_tag})",
    )
    _violin_strip(
        axes[1], head_coms, nohead_coms,
        "Head", "No head",
        "Δ COM distance to reward (post − pre)\n< 0 = toward reward",
        f"COM shift  (all seeds, lr={lr_tag})",
    )
    fig.suptitle(
        f"Head vs. No-Head field shift  |  {len(records)} seeds  |  LR = {lr_tag}",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    for ext in (".png", ".pdf"):
        p = os.path.join(out_dir, f"paired_shift_seeds{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


def plot_per_seed_figure(records: list[dict], out_dir: str, lr_tag: str):
    """One dot per seed, connected lines between head and no-head means."""
    seeds = [r["seed"] for r in records]
    head_peak_means = [r["head"]["peak_mean"] for r in records]
    nohead_peak_means = [r["nohead"]["peak_mean"] for r in records]
    head_com_means = [r["head"]["com_mean"] for r in records]
    nohead_com_means = [r["nohead"]["com_mean"] for r in records]

    # Cohen's d (paired, across seeds)
    d_peak = cohen_d_paired(
        np.array(head_peak_means, dtype=float),
        np.array(nohead_peak_means, dtype=float),
    )
    d_com = cohen_d_paired(
        np.array(head_com_means, dtype=float),
        np.array(nohead_com_means, dtype=float),
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, h_vals, nh_vals, ylabel, metric, d in [
        (axes[0], head_peak_means, nohead_peak_means,
         "Mean peak Δd (post − pre)\n< 0 = toward reward", "Peak", d_peak),
        (axes[1], head_com_means, nohead_com_means,
         "Mean COM Δd (post − pre)\n< 0 = toward reward", "COM", d_com),
    ]:
        for seed, hv, nv in zip(seeds, h_vals, nh_vals):
            ax.plot([1, 2], [hv, nv], "-", color="0.7", lw=1.0, zorder=1)
        ax.scatter([1] * len(h_vals), h_vals, s=60, color=HEAD_C, zorder=3,
                   label="Head", edgecolors="white", linewidths=0.5)
        ax.scatter([2] * len(nh_vals), nh_vals, s=60, color=NOHEAD_C, zorder=3,
                   label="No head", edgecolors="white", linewidths=0.5)

        # Bootstrap CI on the mean across seeds
        _m_h, _lo_h, _hi_h = bootstrap_mean_ci(np.array(h_vals))
        _m_nh, _lo_nh, _hi_nh = bootstrap_mean_ci(np.array(nh_vals))
        for xpos, m, lo, hi, c in [
            (1, _m_h, _lo_h, _hi_h, HEAD_C),
            (2, _m_nh, _lo_nh, _hi_nh, NOHEAD_C),
        ]:
            ax.errorbar([xpos + 0.12], [m], yerr=[[m - lo], [hi - m]],
                        fmt="D", color=c, capsize=5, lw=1.8, ms=7, zorder=4)

        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Head", "No head"])
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"{metric} Δd per seed  (lr={lr_tag})\n"
            f"Cohen's d = {d:.3f}" if np.isfinite(d) else f"{metric} Δd per seed  (lr={lr_tag})",
        )
        ax.legend(frameon=False, fontsize=9)

        # Paired Wilcoxon across seed means
        arr = np.array(h_vals, dtype=float) - np.array(nh_vals, dtype=float)
        if len(arr) >= 5:
            try:
                _, p_wil = wilcoxon(arr)
                ax.text(1.5, ax.get_ylim()[0] * 0.92 if ax.get_ylim()[0] < 0 else ax.get_ylim()[1] * 0.97,
                        f"paired Wilcoxon: p={p_wil:.3f}" if p_wil >= 0.001 else f"paired Wilcoxon: p={p_wil:.1e}",
                        ha="center", fontsize=8, color="0.35")
            except Exception:
                pass

    fig.suptitle(
        f"Per-seed head vs. no-head comparison  ({len(records)} seeds, lr={lr_tag})",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    for ext in (".png", ".pdf"):
        p = os.path.join(out_dir, f"per_seed_delta{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--head-root", default="New_Results_v3",
                    help="Directory containing matched_head_{lr_tag}_seed* subdirs")
    ap.add_argument("--nohead-root", default="New_Results_v3",
                    help="Directory containing matched_nohead_{lr_tag}_seed* subdirs")
    ap.add_argument("--lr-tag", default="1e-4",
                    help="LR tag used in directory names, e.g. '1e-4'")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 7, 42, 99, 123])
    ap.add_argument("--reward-x", type=float, default=0.3)
    ap.add_argument("--reward-y", type=float, default=0.3)
    ap.add_argument("--replay-cycle", type=int, default=1)
    ap.add_argument("--skaggs-threshold", type=float, default=0.15)
    ap.add_argument("--output-dir", default="Analysis_Plots/multi_seed")
    return ap.parse_args()


def main():
    args = parse_args()
    reward = np.array([args.reward_x, args.reward_y], dtype=float)
    out_dir = ensure_dir(args.output_dir)

    print(f"\n[multi_seed_comparison] lr={args.lr_tag}  seeds={args.seeds}")
    records = load_pairs(
        args.head_root, args.nohead_root, args.lr_tag,
        args.seeds, reward, args.skaggs_threshold, args.replay_cycle,
    )

    if not records:
        print("No complete seed pairs found. Check --head-root / --nohead-root / --lr-tag.")
        return

    print(f"\n  {len(records)} seed pairs loaded. Generating figures...")
    plot_violin_figure(records, str(out_dir), args.lr_tag)
    plot_per_seed_figure(records, str(out_dir), args.lr_tag)

    # Summary JSON
    per_seed = []
    for r in records:
        _m_h, _lo_h, _hi_h = bootstrap_mean_ci(r["head"]["dpeak"])
        _m_nh, _lo_nh, _hi_nh = bootstrap_mean_ci(r["nohead"]["dpeak"])
        per_seed.append({
            "seed": r["seed"],
            "head": {
                "n": r["head"]["n"],
                "peak_mean": r["head"]["peak_mean"],
                "peak_ci95": [float(_lo_h), float(_hi_h)],
                "com_mean": r["head"]["com_mean"],
                "peak_frac_toward": r["head"]["peak_frac_toward"],
                "com_frac_toward": r["head"]["com_frac_toward"],
            },
            "nohead": {
                "n": r["nohead"]["n"],
                "peak_mean": r["nohead"]["peak_mean"],
                "peak_ci95": [float(_lo_nh), float(_hi_nh)],
                "com_mean": r["nohead"]["com_mean"],
                "peak_frac_toward": r["nohead"]["peak_frac_toward"],
                "com_frac_toward": r["nohead"]["com_frac_toward"],
            },
        })

    all_head_peak = np.concatenate([r["head"]["dpeak"] for r in records])
    all_nohead_peak = np.concatenate([r["nohead"]["dpeak"] for r in records])
    all_head_com = np.concatenate([r["head"]["dcom"] for r in records])
    all_nohead_com = np.concatenate([r["nohead"]["dcom"] for r in records])

    try:
        _, p_peak_mw = mannwhitneyu(all_head_peak, all_nohead_peak, alternative="two-sided")
        _, p_com_mw = mannwhitneyu(all_head_com, all_nohead_com, alternative="two-sided")
    except Exception:
        p_peak_mw = p_com_mw = float("nan")

    fdr_ps = fdr_bh([p for p in [p_peak_mw, p_com_mw] if np.isfinite(p)])

    save_json(
        out_dir / "multi_seed_summary.json",
        {
            "script": "multi_seed_comparison.py",
            "lr_tag": args.lr_tag,
            "seeds": args.seeds,
            "n_seeds_loaded": len(records),
            "reward_pos": [args.reward_x, args.reward_y],
            "skaggs_threshold": args.skaggs_threshold,
            "pooled_peak_mw_p": float(p_peak_mw) if np.isfinite(p_peak_mw) else None,
            "pooled_com_mw_p": float(p_com_mw) if np.isfinite(p_com_mw) else None,
            "pooled_peak_cohen_d": float(cohen_d_independent(all_head_peak, all_nohead_peak)),
            "pooled_com_cohen_d": float(cohen_d_independent(all_head_com, all_nohead_com)),
            "per_seed": per_seed,
        },
    )
    print(f"  Saved summary -> {out_dir}/multi_seed_summary.json")
    print(f"\n  Pooled peak MW p={p_peak_mw:.3e}  |  COM MW p={p_com_mw:.3e}")


if __name__ == "__main__":
    main()
