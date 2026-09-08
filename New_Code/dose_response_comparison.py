"""
dose_response_comparison.py
Overlay head vs. no-head dose-response curves across the RNN learning-rate sweep.

Loads both controlled LR sweeps:
  --head-root    New_Results_v3/dose_sweep_lr        (reward_pred_head=True)
  --nohead-root  New_Results_v3/dose_sweep_lr_nohead (reward_pred_head=False)

Each subdirectory must have config.json and replay data for cycle 1
(df_activations_replay_c1_{pred,postrl}.pkl).

Produces:
  <output-dir>/dose_response_comparison.{png,pdf}
  <output-dir>/dose_response_comparison_summary.json

The figure has three panels (rows):
  1. Mean Δ distance to reward (peak and COM) vs LR — head (solid) vs no-head (dashed)
  2. Mann-Whitney U p-value at each LR (head vs no-head Δd distributions; peak + COM)
  3. n_active place cells and mean Skaggs (post) vs LR — head vs no-head

Usage:
  python New_Code/dose_response_comparison.py \\
      --head-root    New_Results_v3/dose_sweep_lr \\
      --nohead-root  New_Results_v3/dose_sweep_lr_nohead \\
      --reward-x 0.3 --reward-y 0.3 \\
      --output-dir   Analysis_Plots/dose_response_comparison
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis_helpers import dist_to_reward, has_period, load_period_centers  # noqa: E402
from common import ensure_dir, fdr_bh, save_json  # noqa: E402


# ── helpers ──────────────────────────────────────────────────────────────────

def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return f"{p:.1e}"
    return f"{p:.3f}"


def _run_reward(cfg: dict, fallback_x: float, fallback_y: float) -> np.ndarray:
    rp = cfg.get("reward_pos")
    if rp and len(rp) == 2:
        return np.array([float(rp[0]), float(rp[1])], dtype=float)
    return np.array([fallback_x, fallback_y], dtype=float)


def _shift_arrays(run_dir: str, reward: np.ndarray, thr: float, N: int):
    """Return (dcom, dpeak) arrays for all place-cell-gated units, or None."""
    data_dir = os.path.join(run_dir, "data")
    pre_period = f"replay_c{N}_pred"
    post_period = f"replay_c{N}_postrl"
    if not (has_period(data_dir, pre_period) and has_period(data_dir, post_period)):
        return None
    sk_pre, com_pre, peak_pre = load_period_centers(data_dir, pre_period)
    sk_post, com_post, peak_post = load_period_centers(data_dir, post_period)
    active = np.where((sk_pre >= thr) & (sk_post >= thr))[0]
    if len(active) == 0:
        return None
    dcom = dist_to_reward(com_post[active], reward) - dist_to_reward(com_pre[active], reward)
    dpeak = dist_to_reward(peak_post[active], reward) - dist_to_reward(peak_pre[active], reward)
    sk_post_active = sk_post[active]
    return {
        "n_active": int(len(active)),
        "dcom": dcom,
        "dpeak": dpeak,
        "skaggs_mean_post": float(sk_post_active.mean()),
        "com_mean_delta": float(dcom.mean()),
        "peak_mean_delta": float(dpeak.mean()),
        "com_frac_toward": float((dcom < 0).mean()),
        "peak_frac_toward": float((dpeak < 0).mean()),
    }


def _load_sweep(root: str, reward_x: float, reward_y: float, thr: float, N: int) -> list[dict]:
    """Load all subdirs of root that have config.json + replay data."""
    rows = []
    if not os.path.isdir(root):
        return rows
    for name in sorted(os.listdir(root)):
        run_dir = os.path.join(root, name)
        cfg_path = os.path.join(run_dir, "config.json")
        if not os.path.isfile(cfg_path):
            continue
        cfg = json.loads(open(cfg_path).read())
        lr = cfg.get("rnn_rl_learning_rate")
        if lr is None:
            continue
        reward = _run_reward(cfg, reward_x, reward_y)
        res = _shift_arrays(run_dir, reward, thr, N)
        if res is None:
            print(f"  SKIP {name} (no replay data or no qualifying cells)")
            continue
        rows.append({
            "run": name,
            "lr": float(lr),
            "reward_pred_head": bool(cfg.get("reward_pred_head", False)),
            **res,
        })
        print(f"  {name}: lr={lr:.0e}  n={res['n_active']}  "
              f"COM Δ={res['com_mean_delta']:+.4f}  PEAK Δ={res['peak_mean_delta']:+.4f}")
    rows.sort(key=lambda r: r["lr"])
    return rows


# ── plotting ──────────────────────────────────────────────────────────────────

def _mw_p(arr_head: np.ndarray, arr_nohead: np.ndarray) -> float:
    if len(arr_head) < 3 or len(arr_nohead) < 3:
        return float("nan")
    try:
        stat, p = mannwhitneyu(arr_head, arr_nohead, alternative="two-sided")
        return float(p)
    except Exception:
        return float("nan")


def plot_comparison(head_rows: list[dict], nohead_rows: list[dict], out_dir: str,
                    skaggs_threshold: float) -> dict:
    # Align by LR — only plot LRs present in both conditions
    lrs_head = {r["lr"]: r for r in head_rows}
    lrs_nohead = {r["lr"]: r for r in nohead_rows}
    shared_lrs = sorted(set(lrs_head) & set(lrs_nohead))

    if not shared_lrs:
        print("  WARNING: no shared LRs between head and nohead sweeps. "
              "Cannot produce comparison figure.")
        return {}

    x = np.array(shared_lrs)
    h = [lrs_head[lr] for lr in shared_lrs]
    n = [lrs_nohead[lr] for lr in shared_lrs]

    # p-values at each LR (Mann-Whitney; FDR-correct across both metrics)
    raw_p_com = [_mw_p(lrs_head[lr]["dcom"], lrs_nohead[lr]["dcom"]) for lr in shared_lrs]
    raw_p_peak = [_mw_p(lrs_head[lr]["dpeak"], lrs_nohead[lr]["dpeak"]) for lr in shared_lrs]
    all_p = raw_p_com + raw_p_peak
    fdr_p = fdr_bh([p for p in all_p if np.isfinite(p)])
    # Re-insert NaN into FDR-corrected array in original positions
    fdr_all = []
    fdr_iter = iter(fdr_p)
    for p in all_p:
        fdr_all.append(next(fdr_iter) if np.isfinite(p) else float("nan"))
    fdr_com = fdr_all[:len(shared_lrs)]
    fdr_peak = fdr_all[len(shared_lrs):]

    # ── Figure ────────────────────────────────────────────────────────────────
    HEAD_COLOR_SOLID = "#8055c8"
    HEAD_COLOR_DASHED = "#c85580"
    NOHEAD_COLOR_SOLID = "#55a8c8"
    NOHEAD_COLOR_DASHED = "#c8a055"

    fig, axes = plt.subplots(3, 1, figsize=(9, 12), sharex=True)

    # Panel 1: mean Δd
    ax = axes[0]
    ax.plot(x, [r["peak_mean_delta"] for r in h], "o-", color=HEAD_COLOR_SOLID,
            label="head  peak Δ", lw=1.8)
    ax.plot(x, [r["com_mean_delta"] for r in h], "s--", color=HEAD_COLOR_DASHED,
            label="head  COM Δ", lw=1.5)
    ax.plot(x, [r["peak_mean_delta"] for r in n], "o-", color=NOHEAD_COLOR_SOLID,
            label="nohead peak Δ", lw=1.8, alpha=0.8)
    ax.plot(x, [r["com_mean_delta"] for r in n], "s--", color=NOHEAD_COLOR_DASHED,
            label="nohead COM Δ", lw=1.5, alpha=0.8)
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_ylabel("mean Δ distance to reward\n(< 0 = toward)")
    ax.set_title(
        f"Head vs. No-Head dose-response  (Skaggs ≥ {skaggs_threshold:g} in both replay states)",
        fontsize=11,
    )
    ax.legend(frameon=False, ncol=2, fontsize=9)

    # Panel 2: significance (FDR-corrected p-values)
    ax = axes[1]
    ax.plot(x, [-np.log10(p) if np.isfinite(p) else np.nan for p in fdr_com],
            "s--", color=HEAD_COLOR_DASHED, label="COM  FDR-p (head vs nohead)", lw=1.5)
    ax.plot(x, [-np.log10(p) if np.isfinite(p) else np.nan for p in fdr_peak],
            "o-", color=HEAD_COLOR_SOLID, label="Peak FDR-p (head vs nohead)", lw=1.8)
    ax.axhline(-np.log10(0.05), color="red", ls=":", lw=1.0, label="FDR p=0.05")
    ax.set_ylabel("-log₁₀(FDR-corrected p)\nMann-Whitney U (head vs nohead)")
    ax.set_xscale("log")
    ax.legend(frameon=False, fontsize=9)
    # Annotate raw p-values above each point
    for i, (lr, p_com, p_pk) in enumerate(zip(shared_lrs, fdr_com, fdr_peak)):
        ax.annotate(_fmt_p(p_pk), (lr, -np.log10(p_pk) if np.isfinite(p_pk) else 0),
                    textcoords="offset points", xytext=(0, 6), fontsize=7,
                    ha="center", color=HEAD_COLOR_SOLID)

    # Panel 3: n_active + Skaggs
    ax3 = axes[2]
    ax3.plot(x, [r["n_active"] for r in h], "o-", color=HEAD_COLOR_SOLID, lw=1.8,
             label="head n_active")
    ax3.plot(x, [r["n_active"] for r in n], "o-", color=NOHEAD_COLOR_SOLID, lw=1.8,
             alpha=0.8, label="nohead n_active")
    ax3.set_ylabel("# active place cells", color="black")
    ax3b = ax3.twinx()
    ax3b.plot(x, [r["skaggs_mean_post"] for r in h], "s--", color=HEAD_COLOR_DASHED, lw=1.5,
              label="head Skaggs")
    ax3b.plot(x, [r["skaggs_mean_post"] for r in n], "s--", color=NOHEAD_COLOR_DASHED, lw=1.5,
              alpha=0.8, label="nohead Skaggs")
    ax3b.set_ylabel("mean Skaggs (post)", color="#c88055")
    ax3b.tick_params(axis="y", labelcolor="#c88055")
    ax3.set_xscale("log")
    ax3.set_xlabel("RNN learning rate (plasticity)")

    # Combined legend for panel 3
    h3a, l3a = ax3.get_legend_handles_labels()
    h3b, l3b = ax3b.get_legend_handles_labels()
    ax3.legend(h3a + h3b, l3a + l3b, frameon=False, ncol=2, fontsize=9)

    fig.tight_layout(h_pad=0.5)
    for ext in (".png", ".pdf"):
        out = os.path.join(out_dir, f"dose_response_comparison{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  Saved: {out}")
    plt.close(fig)

    summary = {
        "shared_lrs": shared_lrs,
        "head": [
            {
                "lr": r["lr"],
                "n_active": r["n_active"],
                "com_mean_delta": r["com_mean_delta"],
                "peak_mean_delta": r["peak_mean_delta"],
                "com_frac_toward": r["com_frac_toward"],
                "peak_frac_toward": r["peak_frac_toward"],
                "skaggs_mean_post": r["skaggs_mean_post"],
            }
            for r in h
        ],
        "nohead": [
            {
                "lr": r["lr"],
                "n_active": r["n_active"],
                "com_mean_delta": r["com_mean_delta"],
                "peak_mean_delta": r["peak_mean_delta"],
                "com_frac_toward": r["com_frac_toward"],
                "peak_frac_toward": r["peak_frac_toward"],
                "skaggs_mean_post": r["skaggs_mean_post"],
            }
            for r in n
        ],
        "significance": [
            {
                "lr": float(lr),
                "fdr_p_com": float(fp) if np.isfinite(fp) else None,
                "fdr_p_peak": float(fpk) if np.isfinite(fpk) else None,
                "raw_p_com": float(rp) if np.isfinite(rp) else None,
                "raw_p_peak": float(rpk) if np.isfinite(rpk) else None,
            }
            for lr, fp, fpk, rp, rpk in zip(
                shared_lrs, fdr_com, fdr_peak, raw_p_com, raw_p_peak
            )
        ],
    }
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--head-root", default="New_Results_v3/dose_sweep_lr",
                    help="Root dir of LR sweep WITH reward-prediction head")
    ap.add_argument("--nohead-root", default="New_Results_v3/dose_sweep_lr_nohead",
                    help="Root dir of LR sweep WITHOUT reward-prediction head")
    ap.add_argument("--reward-x", type=float, default=0.3)
    ap.add_argument("--reward-y", type=float, default=0.3)
    ap.add_argument("--replay-cycle", type=int, default=1)
    ap.add_argument("--skaggs-threshold", type=float, default=0.15,
                    help="Skaggs gate for place-cell qualification (both replay states)")
    ap.add_argument("--output-dir", default="Analysis_Plots/dose_response_comparison")
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)

    print(f"\n[dose_response_comparison] Loading head sweep from: {args.head_root}")
    head_rows = _load_sweep(
        args.head_root, args.reward_x, args.reward_y, args.skaggs_threshold, args.replay_cycle
    )

    print(f"\n[dose_response_comparison] Loading nohead sweep from: {args.nohead_root}")
    nohead_rows = _load_sweep(
        args.nohead_root, args.reward_x, args.reward_y, args.skaggs_threshold, args.replay_cycle
    )

    if not head_rows:
        print("WARNING: no head runs loaded — check --head-root path.")
    if not nohead_rows:
        print("WARNING: no nohead runs loaded — check --nohead-root path.")

    print(f"\n[dose_response_comparison] Plotting overlay ({len(head_rows)} head, "
          f"{len(nohead_rows)} nohead runs)...")
    summary = plot_comparison(head_rows, nohead_rows, str(out_dir), args.skaggs_threshold)

    save_json(
        out_dir / "dose_response_comparison_summary.json",
        {
            "script": "dose_response_comparison.py",
            "head_root": args.head_root,
            "nohead_root": args.nohead_root,
            "reward_pos": [args.reward_x, args.reward_y],
            "skaggs_threshold": args.skaggs_threshold,
            **summary,
        },
    )
    print(f"  Saved summary -> {out_dir}/dose_response_comparison_summary.json")


if __name__ == "__main__":
    main()
