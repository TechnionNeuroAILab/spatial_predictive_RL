"""
run_matched_seeds.py
Clean matched head-vs-no-head comparison at a fixed plasticity, across seeds.

For each seed it runs TWO trainings that are identical except for the reward head:
  matched_head_1e-4_seed<S>    (--reward_pred_head ON)
  matched_nohead_1e-4_seed<S>  (head OFF)

Usage:
  python New_Code/run_matched_seeds.py --seeds 1 2 99
  python New_Code/run_matched_seeds.py --seeds 1 2 99 --skip-existing   # resume-safe
"""
import argparse
import json
import os
import subprocess
import sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join("Code", "ratinabox_v11.py")

BASE = [
    "--n_trials", "1000", "--timesteps", "100", "--phase1_epochs", "100",
    "--n_cycles", "1", "--max_episodes", "1000", "--max_steps", "20",
    "--fixed_goal", "--reward_x", "0.3", "--reward_y", "0.3",
    "--pred_loss_weight", "0.3", "--interleave_epochs", "20",
    "--replay_cycle", "1", "--replay_steps", "50000",
    "--checkpoint_every", "2000", "--no_show",
]
HEAD_FLAGS = ["--reward_pred_head", "--reward_pred_weight", "1.5", "--reward_pred_sigma", "0.15"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--lr", default="1e-4", help="rnn_rl_lr (default 1e-4, the sweep optimum)")
    ap.add_argument("--root", default="New_Results_v3")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--no-prune", dest="no_prune", action="store_true")
    return ap.parse_args()


def run_one(seed, head, args):
    tag = f"matched_{'head' if head else 'nohead'}_1e-4_seed{seed}"
    out = os.path.join(args.root, tag)
    done = os.path.join(REPO, out, "data", "df_activations_replay_c1_postrl.pkl")
    cfg_path = os.path.join(REPO, out, "config.json")

    if args.skip_existing and os.path.exists(done) and os.path.exists(cfg_path):
        if json.load(open(cfg_path)).get("reward_pred_head") == head:
            print(f"=== SKIP {tag} (already done, config OK) ===", flush=True)
            return
        print(f"!! {tag} exists but config head != {head}; re-running", flush=True)

    print(f"\n=== TRAIN {tag}  (head={head}) ===", flush=True)
    cmd = ([sys.executable, TRAIN, "--seed", str(seed), "--output_dir", out,
            "--rnn_rl_lr", args.lr] + BASE + (HEAD_FLAGS if head else []))
    print("  $ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO, check=True)
    if not os.path.exists(cfg_path):
        sys.exit(f"ABORT: {tag} produced no config.json")
    got = json.load(open(cfg_path)).get("reward_pred_head")
    if got != head:
        sys.exit(f"ABORT: {tag} config reward_pred_head={got} but expected {head}. "
                 f"Not pruning; investigate before trusting any results.")
    if not os.path.exists(done):
        sys.exit(f"ABORT: {tag} did not produce the frozen-replay pair (df_activations_replay_c1_postrl).")
    print(f"  VERIFIED: {tag} reward_pred_head={got}, replay pair present.", flush=True)

    if not args.no_prune:
        subprocess.run([sys.executable, os.path.join("New_Code", "prune_run.py"),
                        "--run-dir", out, "--drop-checkpoints"], cwd=REPO, check=True)


def main():
    args = parse_args()
    for seed in args.seeds:
        run_one(seed, head=True, args=args)
        run_one(seed, head=False, args=args)
    print("\n=== ALL DONE. Per-run config was verified after each training. ===")
    print("Runs produced (head / no-head) for seeds:", ", ".join(str(s) for s in args.seeds))


if __name__ == "__main__":
    main()
