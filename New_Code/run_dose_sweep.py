"""
run_dose_sweep.py
Controlled dose-response. Exactly ONE quantity varies across the runs; every
other hyperparameter, the seed, the predictive anchor and the reward position
are held fixed, so any change in the frozen-replay shift is causal.

Two modes:
  --mode weight : vary ONLY the reward-head strength (reward_pred_weight).
  --mode lr     : vary ONLY the RNN plasticity (rnn_rl_lr), holding the
                  reward-head strength fixed. This is the plasticity
                  dose-response -- it answers "does more plasticity remap
                  more, or does it destabilise / collapse the map?".

Each run saves the frozen-replay probe so it can be analysed with the verified
pipeline afterwards (see dose_response.py).

Usage:
  # reward-head strength sweep (original behaviour)
  python New_Code/run_dose_sweep.py --mode weight --weights 0.0 0.5 1.0 1.5 3.0 \
      --seed 42 --root New_Results_v3/dose_sweep_weight

  # RNN plasticity (learning-rate) sweep
  python New_Code/run_dose_sweep.py --mode lr --lrs 2e-6 1e-5 5e-5 1e-4 3e-4 \
      --seed 42 --root New_Results_v3/dose_sweep_lr

  # then aggregate (x-axis matches the swept quantity):
  python New_Code/dose_response.py --scan-root New_Results_v3/dose_sweep_lr \
      --x-param rnn_rl_lr --output-dir Analysis_Plots/dose_response_lr \
      --reward-x 0.3 --reward-y 0.3
"""
import argparse
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join("Code", "ratinabox_v11.py")
FIXED = [
    "--fixed_goal", "--reward_x", "0.3", "--reward_y", "0.3",
    "--n_trials", "1000", "--timesteps", "100", "--phase1_epochs", "100",
    "--n_cycles", "1", "--max_episodes", "1000", "--max_steps", "20",
    "--replay_cycle", "1",
    "--interleave_epochs", "20", "--checkpoint_every", "2000", "--no_show",
    "--reward_pred_sigma", "0.15",
]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["weight", "lr"], default="weight",
                    help="which single quantity to sweep")
    ap.add_argument("--weights", type=float, nargs="+", default=[0.0, 0.5, 1.0, 1.5, 3.0],
                    help="reward_pred_weight values (mode=weight)")
    ap.add_argument("--lrs", type=float, nargs="+", default=[2e-6, 1e-5, 5e-5, 1e-4, 3e-4],
                    help="rnn_rl_lr values (mode=lr)")
    ap.add_argument("--reward_pred_weight", type=float, default=1.5,
                    help="fixed reward-head strength when mode=lr")
    ap.add_argument("--rnn_rl_lr", default="5e-5",
                    help="fixed RNN learning rate when mode=weight")
    ap.add_argument("--pred_loss_weight", default=None,
                    help="predictive-anchor weight; default 0.1 (mode=weight) / 0.3 (mode=lr)")
    ap.add_argument("--replay_steps", default="50000")
    ap.add_argument("--no-head", dest="no_head", action="store_true",
                    help="mode=lr only: sweep plasticity WITHOUT the reward head, i.e. the "
                         "policy-induced remap. Pair with a normal --mode lr run to get the "
                         "policy-only vs policy+head curves.")
    ap.add_argument("--prune", action="store_true",
                    help="after each run, delete all activation probes except the frozen-replay "
                         "pair (saves ~1.6 GB per run)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--root", default=None)
    ap.add_argument("--skip-existing", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    pred_w = args.pred_loss_weight or ("0.3" if args.mode == "lr" else "0.1")
    if args.mode == "lr":
        default_root = "dose_sweep_lr_nohead" if args.no_head else "dose_sweep_lr"
    else:
        default_root = "dose_sweep_weight"
    root = args.root or os.path.join("New_Results_v3", default_root)
    os.makedirs(os.path.join(REPO, root), exist_ok=True)

    if args.mode == "lr":
        head_args = ([] if args.no_head else
                     ["--reward_pred_head", "--reward_pred_weight", str(args.reward_pred_weight)])
        sweep = [(f"lr_{lr:g}", ["--rnn_rl_lr", f"{lr:g}"] + head_args)
                 for lr in args.lrs]
    else:
        sweep = [(f"rew_w_{w:g}",
                  ["--rnn_rl_lr", str(args.rnn_rl_lr), "--reward_pred_head",
                   "--reward_pred_weight", f"{w:g}"])
                 for w in args.weights]

    for tag, var_args in sweep:
        out = os.path.join(root, tag)
        done = os.path.join(REPO, out, "data", "df_activations_replay_c1_postrl.pkl")
        if args.skip_existing and os.path.exists(done):
            print(f"=== SKIP {tag} (already has replay data) ===", flush=True)
            continue
        print(f"\n=== TRAIN {tag} ({args.mode} sweep) -> {out} ===", flush=True)
        cmd = ([sys.executable, TRAIN, "--seed", str(args.seed), "--output_dir", out]
               + FIXED
               + ["--pred_loss_weight", str(pred_w), "--replay_steps", str(args.replay_steps)]
               + var_args)
        print("  $ " + " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=REPO)
        if args.prune and os.path.exists(done):
            subprocess.run([sys.executable, os.path.join("New_Code", "prune_run.py"),
                            "--run-dir", out, "--drop-checkpoints"], cwd=REPO)

    xparam = "rnn_rl_lr" if args.mode == "lr" else "reward_pred_weight"
    print("\nDone. Aggregate with:")
    print(f"  python New_Code/dose_response.py --scan-root {root} "
          f"--x-param {xparam} --output-dir Analysis_Plots/dose_response_{args.mode} "
          f"--reward-x 0.3 --reward-y 0.3")


if __name__ == "__main__":
    main()
