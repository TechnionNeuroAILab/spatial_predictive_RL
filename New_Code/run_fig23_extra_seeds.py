#!/usr/bin/env python3
"""
Extra seeds for the Figure 5 double-dissociation power boost.

Runs additional no-head (Fig 2) and head-ON (Fig 3) seeds beyond the original
0-9, across 2 GPUs with a work-stealing queue (same pattern as
run_fig34_parallel.py). New seeds are paired (same seed index used in both
conditions) so the Figure 5 panel-b paired test gains independent replicates.

Usage:
    python New_Code/run_fig23_extra_seeds.py --seeds 10 11 12 13 14 15 16 17 18 19
    python New_Code/run_fig23_extra_seeds.py --dry_run
"""
from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
V11_SCRIPT = REPO / "Code" / "ratinabox_v11.py"
NOHEAD_OUT = REPO / "Repro" / "multi_seed"
HEAD_OUT = REPO / "Repro" / "multi_seed_head"

# ── No-head (Fig 2) base args — identical to run_fig2_seeds.py ─────────────
FIG2_BASE: list[str] = [
    "--n_trials",          "1000",
    "--timesteps",         "100",
    "--phase1_epochs",     "100",
    "--max_episodes",      "1000",
    "--max_steps",         "20",
    "--checkpoint_every",  "2000",
    "--n_cycles",          "1",
    "--interleave_epochs", "20",
    "--interleave_lr",     "1e-4",
    "--fixed_goal",
    "--rnn_rl_lr",         "5e-5",
    "--pred_loss_weight",  "0.3",
    "--replay_cycle",      "1",
    "--reward_pred_sigma", "0.15",
    "--reward_pred_weight","0.3",
    "--reward_x",          "0.3",
    "--reward_y",          "0.3",
    "--no_show",
]

# ── Head-ON (Fig 3) base args — identical to run_fig34_parallel.py FIG3_BASE
FIG3_BASE: list[str] = [
    "--n_trials",          "1000",
    "--timesteps",         "100",
    "--phase1_epochs",     "100",
    "--max_episodes",      "1000",
    "--max_steps",         "20",
    "--checkpoint_every",  "2000",
    "--n_cycles",          "1",
    "--interleave_epochs", "20",
    "--interleave_lr",     "1e-4",
    "--fixed_goal",
    "--rnn_rl_lr",         "5e-5",
    "--pred_loss_weight",  "0.3",
    "--replay_cycle",      "1",
    "--reward_pred_sigma", "0.15",
    "--reward_pred_weight","1.5",
    "--reward_x",          "0.3",
    "--reward_y",          "0.3",
    "--reward_pred_head",
    "--no_show",
]


def _already_done_nohead(seed: int) -> bool:
    return (NOHEAD_OUT / f"nohead_seed{seed}" / "data" / "df_activations_cycle_1.pkl").exists()


def _already_done_head(seed: int) -> bool:
    return (HEAD_OUT / f"head_seed{seed}" / "data" / "df_activations_cycle_1.pkl").exists()


def _build_cmd(cond: str, seed: int):
    if cond == "nohead":
        run_name = f"nohead_seed{seed}"
        out_dir = NOHEAD_OUT
        log = NOHEAD_OUT / f"{run_name}_runner.log"
        cmd = [sys.executable, str(V11_SCRIPT),
               "--seed", str(seed), "--output_dir", str(out_dir),
               "--run_name", run_name, *FIG2_BASE]
    else:
        run_name = f"head_seed{seed}"
        out_dir = HEAD_OUT
        log = HEAD_OUT / f"{run_name}_runner.log"
        cmd = [sys.executable, str(V11_SCRIPT),
               "--seed", str(seed), "--output_dir", str(out_dir),
               "--run_name", run_name, *FIG3_BASE]
    return cmd, out_dir, log


def worker(gpu_id: int, job_q: "queue.Queue", results: list, lock: threading.Lock, dry_run: bool):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
    while True:
        try:
            cond, seed = job_q.get_nowait()
        except queue.Empty:
            return
        cmd, out_dir, log = _build_cmd(cond, seed)
        out_dir.mkdir(parents=True, exist_ok=True)
        label = f"{cond}/seed{seed}"

        if dry_run:
            print(f"DRY RUN [{label} -> GPU{gpu_id}]: {' '.join(str(c) for c in cmd)}")
            job_q.task_done()
            continue

        with lock:
            print(f"\n{'='*60}\n  START  {label} -> GPU{gpu_id}\n  log -> {log}\n{'='*60}", flush=True)

        with open(log, "w") as lf:
            lf.write("CMD: " + " ".join(str(c) for c in cmd) + "\n\n")
            lf.flush()
            result = subprocess.run(cmd, stdout=lf, stderr=lf, env=env)

        with lock:
            status = "OK" if result.returncode == 0 else f"FAIL(rc={result.returncode})"
            print(f"\n{'='*60}\n  {status}  {label} -> GPU{gpu_id}\n{'='*60}", flush=True)
            results.append((label, result.returncode == 0))
        job_q.task_done()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", required=True, help="New seed indices to run")
    p.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    p.add_argument("--skip_existing", action="store_true", default=True)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    job_q: "queue.Queue" = queue.Queue()
    n_queued = 0
    for seed in args.seeds:
        if not (args.skip_existing and _already_done_nohead(seed)):
            job_q.put(("nohead", seed))
            n_queued += 1
        if not (args.skip_existing and _already_done_head(seed)):
            job_q.put(("head", seed))
            n_queued += 1

    print(f"Queued {n_queued} jobs across {len(args.gpus)} GPU(s): {args.gpus}")

    results: list = []
    lock = threading.Lock()
    threads = [threading.Thread(target=worker, args=(gpu, job_q, results, lock, args.dry_run))
               for gpu in args.gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if not args.dry_run:
        ok = [r for r, s in results if s]
        bad = [r for r, s in results if not s]
        print(f"\n{'='*60}\nAll jobs finished.  OK: {len(ok)}  Failed: {bad}")
        print(f"  No-head outputs: {NOHEAD_OUT}")
        print(f"  Head outputs:    {HEAD_OUT}")


if __name__ == "__main__":
    main()
