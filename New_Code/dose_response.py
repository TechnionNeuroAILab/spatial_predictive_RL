from __future__ import annotations
import argparse
import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis_helpers import load_period_centers, dist_to_reward, has_period  # noqa: E402
from common import ensure_dir, save_json  # noqa: E402

RUNS = [
    "reward_pred_head",
    "reward_pred_head_offcenter",
    "reward_pred_head_strong",
    "reward_pred_head_v2",
]
def run_reward(cfg):
    rp = cfg.get("reward_pos")
    if rp and len(rp) == 2:
        return np.array([float(rp[0]), float(rp[1])])
    return np.array([0.5, 0.5])


def shift_for_run(run_dir, reward, thr=0.15, N=1):
    data_dir = os.path.join(run_dir, "data")
    if not (has_period(data_dir, f"replay_c{N}_pred") and has_period(data_dir, f"replay_c{N}_postrl")):
        return None
    sk_pre, com_pre, peak_pre = load_period_centers(data_dir, f"replay_c{N}_pred")
    sk_post, com_post, peak_post = load_period_centers(data_dir, f"replay_c{N}_postrl")
    # gate: keep only genuine place cells -- Skaggs >= thr in BOTH replay states.
    active = np.where((sk_pre >= thr) & (sk_post >= thr))[0]
    dcom = dist_to_reward(com_post[active], reward) - dist_to_reward(com_pre[active], reward)
    dpk = dist_to_reward(peak_post[active], reward) - dist_to_reward(peak_pre[active], reward)
    return {
        "n_active": int(len(active)),
        "com_mean_delta": float(dcom.mean()),
        "com_frac_toward": float((dcom < 0).mean()),
        "peak_mean_delta": float(dpk.mean()),
        "peak_frac_toward": float((dpk < 0).mean()),
        "skaggs_mean_pre": float(sk_pre[active].mean()) if len(active) else float("nan"),
        "skaggs_mean_post": float(sk_post[active].mean()) if len(active) else float("nan"),
    }


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", default="New_Results_v2")
    ap.add_argument("--scan-root", default=None,
                    help="if given, aggregate ALL run subdirs under this folder (e.g. the "
                         "controlled dose_sweep) instead of the fixed ad-hoc run list")
    ap.add_argument("--reward-x", type=float, default=None,
                    help="reward x for scan-root runs whose config lacks reward_pos")
    ap.add_argument("--reward-y", type=float, default=None)
    ap.add_argument("--x-param", "--x_param", dest="x_param",
                    choices=["reward_pred_weight", "rnn_rl_lr"],
                    default="reward_pred_weight",
                    help="which swept hyperparameter is the x-axis (use rnn_rl_lr for the "
                         "plasticity sweep)")
    ap.add_argument("--skaggs-threshold", "--skaggs_threshold", dest="skaggs_threshold",
                    type=float, default=0.15,
                    help="keep only cells with Skaggs >= this in BOTH replay states. Use >=1.0 "
                         "to restrict to genuine place cells (removes the diffuse-cell artifact "
                         "that inflates the base peak shift, e.g. policy-only @1e-4).")
    ap.add_argument("--output-dir", "--output_dir", dest="output_dir", default="Analysis_Plots/dose_response")
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)

    if args.scan_root:
        names = sorted(d for d in os.listdir(args.scan_root)
                       if os.path.exists(os.path.join(args.scan_root, d, "config.json")))
        base = args.scan_root
    else:
        names, base = RUNS, args.results_root

    rows = []
    for name in names:
        run_dir = os.path.join(base, name)
        cfg_path = os.path.join(run_dir, "config.json")
        if not os.path.exists(cfg_path):
            print(f"  SKIP {name} (no config)")
            continue
        cfg = json.loads(open(cfg_path).read())
        reward = run_reward(cfg)
        if cfg.get("reward_pos") is None and args.reward_x is not None and args.reward_y is not None:
            reward = np.array([args.reward_x, args.reward_y], dtype=float)
        res = shift_for_run(run_dir, reward, thr=args.skaggs_threshold)
        if res is None:
            print(f"  SKIP {name} (no replay data)")
            continue
        row = {
            "run": name,
            "reward_pred_weight": cfg.get("reward_pred_weight"),
            "pred_loss_weight": cfg.get("pred_loss_weight"),
            "rnn_rl_lr": cfg.get("rnn_rl_learning_rate"),
            "reward_pos": reward.tolist(),
            **res,
        }
        rows.append(row)
        print(f"  {name:20s} rew_w={row['reward_pred_weight']} pred_w={row['pred_loss_weight']} "
              f"lr={row['rnn_rl_lr']}  n={res['n_active']} skaggs_post={res['skaggs_mean_post']:.2f}  "
              f"PEAK delta={res['peak_mean_delta']:+.4f} ({100*res['peak_frac_toward']:.0f}% toward)  "
              f"COM delta={res['com_mean_delta']:+.4f} ({100*res['com_frac_toward']:.0f}% toward)")

    if not rows:
        print("No runs available.")
        return

    xkey = args.x_param
    controlled = args.scan_root is not None
    rows.sort(key=lambda r: (r[xkey] if r[xkey] is not None else 0))
    x = [r[xkey] for r in rows]
    xlabel = ("RNN learning rate (plasticity)" if xkey == "rnn_rl_lr"
              else "reward_pred_weight $\\lambda_{rew}$")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    ax1.plot(x, [r["peak_mean_delta"] for r in rows], "o-", color="#8055c8", label="peak $\\Delta$")
    ax1.plot(x, [r["com_mean_delta"] for r in rows], "s--", color="#c85580", label="COM $\\Delta$")
    ax1.axhline(0, color="gray", lw=0.8, ls=":")
    ax1.set_ylabel("mean $\\Delta$ distance to reward\n(< 0 = toward)")
    ax1.legend(frameon=False)
    ax1.set_title(("Controlled " if controlled else "UNCONTROLLED ")
                  + f"sweep vs {xkey}  (place cells: Skaggs $\\geq$ {args.skaggs_threshold:g} in both states)"
                  + ("" if controlled else "\n(other hyperparameters + reward vary — see labels)"))

    ax2.plot(x, [r["n_active"] for r in rows], "o-", color="#5588c8", label="# active place cells")
    ax2.set_ylabel("# active place cells", color="#5588c8")
    ax2.tick_params(axis="y", labelcolor="#5588c8")
    ax2b = ax2.twinx()
    ax2b.plot(x, [r["skaggs_mean_post"] for r in rows], "s--", color="#c8a055",
              label="mean Skaggs (post)")
    ax2b.set_ylabel("mean Skaggs info (post)", color="#c8a055")
    ax2b.tick_params(axis="y", labelcolor="#c8a055")
    ax2.set_xlabel(xlabel)

    if xkey == "rnn_rl_lr":
        ax1.set_xscale("log")
        ax2.set_xscale("log")
    if not controlled:
        for r in rows:
            ax1.annotate(f"{r['run'].replace('reward_pred_head', 'rph')}\n"
                         f"pred_w={r['pred_loss_weight']}, lr={r['rnn_rl_lr']}",
                         (r[xkey], r["peak_mean_delta"]),
                         textcoords="offset points", xytext=(6, 0), fontsize=6, va="center")
    fig.tight_layout()
    out = os.path.join(out_dir, "dose_response.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)

    caveat = (f"Controlled sweep: only {xkey} varies across runs; seed, predictive anchor, "
              "reward-head strength (where fixed) and reward position are held constant, so "
              "the trend is causal. The map-health panel (n_active, Skaggs) flags where high "
              "plasticity destabilises the representation." if controlled else
              "NOT a controlled sweep: reward_pred_weight, pred_loss_weight, rnn_rl_lr and "
              "reward position all vary between runs; the strongest run partially collapsed "
              "the map. Interpret only as an honest tabulation, not a causal dose-response.")
    save_json(os.path.join(out_dir, "dose_response_summary.json"), {
        "script": "dose_response.py",
        "x_param": xkey,
        "controlled": controlled,
        "skaggs_threshold": args.skaggs_threshold,
        "caveat": caveat,
        "runs": rows,
    })
    print(f"  saved -> {out_dir}")


if __name__ == "__main__":
    main()
