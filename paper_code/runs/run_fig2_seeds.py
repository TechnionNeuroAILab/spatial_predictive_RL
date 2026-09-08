#!/usr/bin/env python3
"""
Run 10 seeds with the same config as Repro/nohead_seed42.

Outputs go to Repro/multi_seed/nohead_seed{N}/.
Runs are sequential (single GPU).  Estimated ~7 min each → ~70 min total.

Usage:
    python paper_code/runs/run_fig2_seeds.py
    python paper_code/runs/run_fig2_seeds.py --seeds 0 1 2 3 4 5 6 7 8 9
    python paper_code/runs/run_fig2_seeds.py --dry_run   # print commands only
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "paper_code" / "model" / "ratinabox_v11.py"
OUT_DIR = REPO / "Repro" / "multi_seed"

# Identical to Repro/nohead_seed42 config except seed and output paths
BASE_ARGS: list[str] = [
    "--n_trials",        "1000",
    "--timesteps",       "100",
    "--phase1_epochs",   "100",
    "--max_episodes",    "1000",
    "--max_steps",       "20",
    "--checkpoint_every","2000",
    "--n_cycles",        "1",
    "--interleave_epochs","20",
    "--interleave_lr",   "1e-4",
    "--fixed_goal",
    "--rnn_rl_lr",       "5e-5",
    "--pred_loss_weight","0.3",
    "--replay_cycle",    "1",
    "--reward_pred_sigma","0.15",
    "--reward_pred_weight","0.3",
    "--reward_x",        "0.3",
    "--reward_y",        "0.3",
    "--no_show",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)),
                   help="Seeds to run (default 0–9)")
    p.add_argument("--dry_run", action="store_true", help="Print commands without running")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip seeds whose data/df_activations_cycle_1.pkl already exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ok, failed = [], []
    for seed in args.seeds:
        run_name = f"nohead_seed{seed}"
        run_dir = OUT_DIR / run_name
        log_path = OUT_DIR / f"{run_name}_runner.log"

        if args.skip_existing:
            probe_file = run_dir / "data" / "df_activations_cycle_1.pkl"
            if probe_file.exists():
                print(f"[skip] seed {seed} — already complete")
                ok.append(seed)
                continue

        cmd = [
            sys.executable, str(SCRIPT),
            "--seed", str(seed),
            "--output_dir", str(OUT_DIR),
            "--run_name", run_name,
            *BASE_ARGS,
        ]

        if args.dry_run:
            print("DRY RUN:", " ".join(str(c) for c in cmd))
            continue

        print(f"\n{'='*60}")
        print(f"Seed {seed}/{args.seeds[-1]}  →  {run_dir}")
        print(f"Log  →  {log_path}")
        print(f"{'='*60}", flush=True)

        with open(log_path, "w") as lf:
            lf.write("CMD: " + " ".join(str(c) for c in cmd) + "\n\n")
            lf.flush()
            result = subprocess.run(cmd, stdout=lf, stderr=lf)

        if result.returncode != 0:
            print(f"  ERROR: seed {seed} failed (rc={result.returncode})")
            failed.append(seed)
        else:
            print(f"  OK: seed {seed} complete")
            ok.append(seed)

    print(f"\n{'='*60}")
    print(f"Done.  OK: {ok}  |  Failed: {failed}")
    print(f"Outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
