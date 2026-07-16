"""
replay_field_analysis.py
Frozen-weight replay analyses for one run (pre = replay_c{N}_pred, post =
replay_c{N}_postrl). Because the trajectory is identical in both passes, any
change in the fields is representational, not occupancy-driven.

Produces three analyses, all from the SAME verified field-centre pipeline used
for the paper figures (see analysis_helpers):

  1. shift             -- per-cell COM and PEAK distance-to-reward change (Delta d),
                          means, fraction toward reward, Wilcoxon. Reproduces the
                          paper's Fig-3 numbers as a self-consistency check.
  2. shift_vs_gate     -- Delta d as the place-cell Skaggs gate is tightened
                          (require Skaggs >= g in BOTH pre and post). A degenerate-
                          cell artifact would VANISH as the gate tightens; a
                          genuine shift persists/grows.
  3. recruitment_fate  -- classify cells as shared / lost / recruited / neither by
                          place-cell status pre vs post; report per-fate Skaggs
                          change (sharpening vs diffusing) and distance to reward.

Usage:
  python New_Code/replay_field_analysis.py \
      --run-dir New_Results_v2/reward_pred_head_offcenter \
      --output-dir Analysis_Plots/reward_head/replay \
      --reward-x 0.3 --reward-y 0.3
"""
from __future__ import annotations
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis_helpers import load_period_centers, dist_to_reward  # noqa: E402

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from common import ensure_dir, save_json  # noqa: E402
except Exception:  # pragma: no cover
    import json

    def ensure_dir(p):
        os.makedirs(p, exist_ok=True)
        return p

    def save_json(p, payload):
        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)


def _wilcoxon_p(a: np.ndarray, b: np.ndarray) -> float:
    """Paired Wilcoxon p-value; NaN if not computable."""
    if len(a) < 5 or np.all(a == b):
        return float("nan")
    try:
        _, p = wilcoxon(a, b)
        return float(p)
    except Exception:
        return float("nan")

def analyse_shift(sk_pre, com_pre, peak_pre, sk_post, com_post, peak_post, reward, thr):
    active = np.where(np.maximum(sk_pre, sk_post) >= thr)[0]
    dcom_pre = dist_to_reward(com_pre[active], reward)
    dcom_post = dist_to_reward(com_post[active], reward)
    dpk_pre = dist_to_reward(peak_pre[active], reward)
    dpk_post = dist_to_reward(peak_post[active], reward)
    dcom = dcom_post - dcom_pre
    dpk = dpk_post - dpk_pre
    return {
        "active": active,
        "n_active": int(len(active)),
        "com": {
            "pre_mean": float(dcom_pre.mean()),
            "post_mean": float(dcom_post.mean()),
            "mean_delta": float(dcom.mean()),
            "frac_toward": float((dcom < 0).mean()),
            "wilcoxon_p": _wilcoxon_p(dcom_post, dcom_pre),
            "delta": dcom,
        },
        "peak": {
            "pre_mean": float(dpk_pre.mean()),
            "post_mean": float(dpk_post.mean()),
            "mean_delta": float(dpk.mean()),
            "frac_toward": float((dpk < 0).mean()),
            "wilcoxon_p": _wilcoxon_p(dpk_post, dpk_pre),
            "delta": dpk,
        },
    }


def plot_shift(shift, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, key, color, title in [
        (axes[0], "com", "slateblue", "Centre-of-mass"),
        (axes[1], "peak", "seagreen", "Peak (de-confounded)"),
    ]:
        d = shift[key]["delta"]
        lim = np.nanpercentile(np.abs(d), 99) if len(d) else 0.3
        ax.hist(d, bins=np.linspace(-lim, lim, 31), color=color, alpha=0.75)
        ax.axvline(0, color="black", lw=1.2)
        ax.axvline(shift[key]["mean_delta"], color="crimson", ls="--", lw=1.8,
                   label=f"mean $\\Delta$ = {shift[key]['mean_delta']:+.3f}")
        ax.set_title(f"{title}\n{100*shift[key]['frac_toward']:.0f}% toward reward, "
                     f"p = {shift[key]['wilcoxon_p']:.1e}")
        ax.set_xlabel("$\\Delta$ distance to reward (post $-$ pre); < 0 = toward")
        ax.set_ylabel("Neurons")
        ax.legend()
    fig.suptitle(f"Frozen-replay field shift ({shift['n_active']} paired place cells)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def analyse_gate(sk_pre, com_pre, peak_pre, sk_post, com_post, peak_post, reward, gates):
    rows = []
    for g in gates:
        sel = np.where((sk_pre >= g) & (sk_post >= g))[0]  # place cell in BOTH passes
        if len(sel) < 5:
            rows.append({"gate": float(g), "n": int(len(sel)),
                         "com_mean_delta": float("nan"), "peak_mean_delta": float("nan"),
                         "com_p": float("nan"), "peak_p": float("nan")})
            continue
        dcom = dist_to_reward(com_post[sel], reward) - dist_to_reward(com_pre[sel], reward)
        dpk = dist_to_reward(peak_post[sel], reward) - dist_to_reward(peak_pre[sel], reward)
        rows.append({
            "gate": float(g),
            "n": int(len(sel)),
            "com_mean_delta": float(dcom.mean()),
            "peak_mean_delta": float(dpk.mean()),
            "com_p": _wilcoxon_p(dist_to_reward(com_post[sel], reward), dist_to_reward(com_pre[sel], reward)),
            "peak_p": _wilcoxon_p(dist_to_reward(peak_post[sel], reward), dist_to_reward(peak_pre[sel], reward)),
        })
    return rows


def plot_gate(rows, out_path):
    g = [r["gate"] for r in rows]
    com = [r["com_mean_delta"] for r in rows]
    pk = [r["peak_mean_delta"] for r in rows]
    ns = [r["n"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(g, com, "-o", color="slateblue", label="COM mean $\\Delta$")
    ax.plot(g, pk, "-s", color="seagreen", label="Peak mean $\\Delta$")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    for gi, ci, ni in zip(g, com, ns):
        if np.isfinite(ci):
            ax.annotate(f"n={ni}", (gi, ci), textcoords="offset points",
                        xytext=(0, -14), ha="center", fontsize=8, color="gray")
    ax.set_xlabel("Place-cell gate: require Skaggs $\\geq g$ in BOTH pre and post")
    ax.set_ylabel("Mean $\\Delta$ distance to reward (< 0 = toward)")
    ax.set_title("Toward-reward shift vs. place-cell gate\n"
                 "(an artifact would vanish as the gate tightens)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def analyse_fate(sk_pre, com_pre, sk_post, com_post, reward, thr):
    place_pre = sk_pre >= thr
    place_post = sk_post >= thr
    fates = {
        "shared": place_pre & place_post,
        "lost": place_pre & ~place_post,
        "recruited": ~place_pre & place_post,
        "neither": ~place_pre & ~place_post,
    }
    dpre = dist_to_reward(com_pre, reward)
    dpost = dist_to_reward(com_post, reward)
    out = {}
    for name, mask in fates.items():
        idx = np.where(mask)[0]
        out[name] = {
            "n": int(len(idx)),
            "mean_skaggs_pre": float(sk_pre[idx].mean()) if len(idx) else float("nan"),
            "mean_skaggs_post": float(sk_post[idx].mean()) if len(idx) else float("nan"),
            "mean_dist_pre": float(dpre[idx].mean()) if len(idx) else float("nan"),
            "mean_dist_post": float(dpost[idx].mean()) if len(idx) else float("nan"),
        }
    return out, fates, sk_pre, sk_post, dpre, dpost


def plot_fate(fate_stats, fates, sk_pre, sk_post, dpre, dpost, thr, out_path):
    colors = {"shared": "#8055c8", "lost": "#e76f51", "recruited": "#2a9d8f", "neither": "#bbbbbb"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    # (a) skaggs pre vs post, colored by fate
    ax = axes[0]
    for name, mask in fates.items():
        idx = np.where(mask)[0]
        ax.scatter(sk_pre[idx], sk_post[idx], s=8, alpha=0.5, color=colors[name],
                   label=f"{name} (n={len(idx)})", edgecolors="none")
    ax.axvline(thr, color="k", lw=0.8, ls=":")
    ax.axhline(thr, color="k", lw=0.8, ls=":")
    lim = max(sk_pre.max(), sk_post.max())
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Skaggs pre-RL")
    ax.set_ylabel("Skaggs post-RL replay")
    ax.set_title("a) Place-cell status migration")
    ax.legend(fontsize=8)
    # (b) mean skaggs change per fate (sharpening vs diffusing)
    ax = axes[1]
    order = ["lost", "recruited", "shared"]
    pre_vals = [fate_stats[k]["mean_skaggs_pre"] for k in order]
    post_vals = [fate_stats[k]["mean_skaggs_post"] for k in order]
    xpos = np.arange(len(order))
    ax.bar(xpos - 0.18, pre_vals, width=0.36, label="pre-RL", color="#4c78a8")
    ax.bar(xpos + 0.18, post_vals, width=0.36, label="post-RL replay", color="#f58518")
    ax.set_xticks(xpos)
    ax.set_xticklabels(order)
    ax.set_ylabel("Mean Skaggs (bits)")
    ax.set_title("b) Sharpness change by fate")
    ax.legend()
    # (c) mean distance to reward pre vs post per fate
    ax = axes[2]
    pre_d = [fate_stats[k]["mean_dist_pre"] for k in order]
    post_d = [fate_stats[k]["mean_dist_post"] for k in order]
    ax.bar(xpos - 0.18, pre_d, width=0.36, label="pre-RL", color="#4c78a8")
    ax.bar(xpos + 0.18, post_d, width=0.36, label="post-RL replay", color="#f58518")
    ax.set_xticks(xpos)
    ax.set_xticklabels(order)
    ax.set_ylabel("Mean distance to reward")
    ax.set_title("c) Distance to reward by fate")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", "--run_dir", dest="run_dir", required=True)
    ap.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True)
    ap.add_argument("--reward-x", type=float, default=0.3)
    ap.add_argument("--reward-y", type=float, default=0.3)
    ap.add_argument("--replay-cycle", type=int, default=1)
    ap.add_argument("--skaggs-threshold", type=float, default=0.15)
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--hidden-dim", type=int, default=512)
    return ap.parse_args()


def main():
    args = parse_args()
    reward = np.array([args.reward_x, args.reward_y], dtype=float)
    thr = args.skaggs_threshold
    data_dir = os.path.join(args.run_dir, "data")
    out_dir = ensure_dir(args.output_dir)
    N = args.replay_cycle

    print(f"[replay_field_analysis] run={args.run_dir}  reward={reward}  thr={thr}")
    sk_pre, com_pre, peak_pre = load_period_centers(data_dir, f"replay_c{N}_pred", args.m, args.hidden_dim)
    sk_post, com_post, peak_post = load_period_centers(data_dir, f"replay_c{N}_postrl", args.m, args.hidden_dim)

    # 1. shift
    shift = analyse_shift(sk_pre, com_pre, peak_pre, sk_post, com_post, peak_post, reward, thr)
    print(f"  shift: n_active={shift['n_active']}  "
          f"COM mean_delta={shift['com']['mean_delta']:+.4f} "
          f"({100*shift['com']['frac_toward']:.0f}% toward, p={shift['com']['wilcoxon_p']:.1e})  |  "
          f"PEAK mean_delta={shift['peak']['mean_delta']:+.4f} "
          f"({100*shift['peak']['frac_toward']:.0f}% toward, p={shift['peak']['wilcoxon_p']:.1e})")
    plot_shift(shift, os.path.join(out_dir, "shift.png"))

    # 2. shift_vs_gate
    gates = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    gate_rows = analyse_gate(sk_pre, com_pre, peak_pre, sk_post, com_post, peak_post, reward, gates)
    plot_gate(gate_rows, os.path.join(out_dir, "shift_vs_gate.png"))
    print("  gate sweep (BOTH-pass Skaggs>=g):")
    for r in gate_rows:
        print(f"    g={r['gate']:.1f}  n={r['n']:4d}  COM Δ={r['com_mean_delta']:+.4f}  "
              f"PEAK Δ={r['peak_mean_delta']:+.4f}")

    # 3. recruitment_fate
    fate_stats, fates, skp, skq, dpre, dpost = analyse_fate(sk_pre, com_pre, sk_post, com_post, reward, thr)
    plot_fate(fate_stats, fates, skp, skq, dpre, dpost, thr, os.path.join(out_dir, "recruitment_fate.png"))
    print("  fates:", {k: v["n"] for k, v in fate_stats.items()})

    # strip arrays before JSON
    shift_json = {
        "n_active": shift["n_active"],
        "com": {k: v for k, v in shift["com"].items() if k != "delta"},
        "peak": {k: v for k, v in shift["peak"].items() if k != "delta"},
    }
    save_json(os.path.join(out_dir, "replay_field_analysis_summary.json"), {
        "script": "replay_field_analysis.py",
        "run_dir": args.run_dir,
        "reward": reward.tolist(),
        "skaggs_threshold": thr,
        "shift": shift_json,
        "shift_vs_gate": gate_rows,
        "recruitment_fate": fate_stats,
    })
    print(f"  saved -> {out_dir}")


if __name__ == "__main__":
    main()
