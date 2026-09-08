#!/usr/bin/env python3
"""
Unified parallel runner for Figures 3 and 4 — 10 seeds each, across 2 GPUs.

Figure 3 (reward-prediction head):
    Calls Code/ratinabox_v11.py with --reward_pred_head on, seeds 0–9.
    Outputs → Repro/multi_seed_head/head_seed{N}/

Figure 4 (reward relocation A → B):
    Calls Code/ratinabox_relocation.py with 2 cycles (A then B), seeds 0–9.
    Outputs → Repro/multi_seed_reloc/reloc_seed{N}/

Work-stealing: a shared queue of 20 jobs is drained by 2 GPU workers.
Fig 4 jobs are put first in the queue (they are ~2× longer) so both GPUs start
on the long jobs immediately, then mop up Fig 3 jobs afterward.

Usage:
    # Run everything (all 20 jobs across GPUs 0 and 1):
    python paper_code/runs/run_fig34_parallel.py

    # Dry-run to see commands:
    python paper_code/runs/run_fig34_parallel.py --dry_run

    # Only Fig 3, only Fig 4, custom seeds/GPUs:
    python paper_code/runs/run_fig34_parallel.py --only fig3 --gpus 0
    python paper_code/runs/run_fig34_parallel.py --only fig4 --gpus 0 1 --seeds 0 1 2 3 4
    python paper_code/runs/run_fig34_parallel.py --skip_existing
"""
from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
V11_SCRIPT = REPO / "paper_code" / "model" / "ratinabox_v11.py"
RELOC_SCRIPT = REPO / "paper_code" / "model" / "ratinabox_relocation.py"
HEAD_OUT = REPO / "Repro" / "multi_seed_head"
RELOC_OUT = REPO / "Repro" / "multi_seed_reloc"

# ── Fig 3 base args (same as Fig 2 + head flags) ─────────────────────────────
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
    "--reward_pred_weight","1.5",    # head-specific (vs 0.3 unused in Fig 2)
    "--reward_x",          "0.3",
    "--reward_y",          "0.3",
    "--reward_pred_head",           # THE key flag
    "--no_show",
]

# ── Fig 4 base args (relocation, 2 cycles, head ON) ──────────────────────────
FIG4_BASE: list[str] = [
    "--n_trials",          "1000",
    "--timesteps",         "100",
    "--phase1_epochs",     "100",
    "--max_episodes",      "2000",
    "--max_steps",         "20",
    "--checkpoint_every",  "2000",
    "--n_cycles",          "2",
    "--interleave_epochs", "20",
    "--interleave_lr",     "1e-4",
    "--fixed_goal",
    "--rnn_rl_lr",         "5e-5",
    "--pred_loss_weight",  "0.3",
    "--replay_cycle",      "1",
    "--reward_pred_sigma", "0.15",
    "--reward_pred_weight","1.5",
    "--reward_x",          "0.3",   # reward A
    "--reward_y",          "0.3",
    "--relocate_x",        "0.7",   # reward B
    "--relocate_y",        "0.7",
    "--relocate_at_cycle", "1",     # 0-indexed: cycle 1 = second cycle
    "--reward_pred_head",
    "--no_show",
]


# ─────────────────────────────────────────────────────────────────────────────

def _already_done_fig3(seed: int) -> bool:
    probe = HEAD_OUT / f"head_seed{seed}" / "data" / "df_activations_cycle_1.pkl"
    return probe.exists()


def _already_done_fig4(seed: int) -> bool:
    probe = RELOC_OUT / f"reloc_seed{seed}" / "data" / "df_activations_replay_reloc_postB.pkl"
    return probe.exists()


def _build_cmd(fig: int, seed: int) -> tuple[list[str], Path, Path]:
    """Return (cmd, out_dir, log_path)."""
    if fig == 3:
        run_name = f"head_seed{seed}"
        out_dir = HEAD_OUT
        log = HEAD_OUT / f"{run_name}_runner.log"
        cmd = [sys.executable, str(V11_SCRIPT),
               "--seed", str(seed),
               "--output_dir", str(out_dir),
               "--run_name", run_name,
               *FIG3_BASE]
    else:  # fig == 4
        run_name = f"reloc_seed{seed}"
        out_dir = RELOC_OUT
        log = RELOC_OUT / f"{run_name}_runner.log"
        cmd = [sys.executable, str(RELOC_SCRIPT),
               "--seed", str(seed),
               "--output_dir", str(out_dir),
               "--run_name", run_name,
               *FIG4_BASE]
    return cmd, out_dir, log


def worker(gpu_id: int, job_queue: queue.Queue, dry_run: bool, lock: threading.Lock) -> None:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
    while True:
        try:
            fig, seed = job_queue.get_nowait()
        except queue.Empty:
            break

        cmd, out_dir, log_path = _build_cmd(fig, seed)
        label = f"Fig{fig}/seed{seed} → GPU{gpu_id}"

        with lock:
            print(f"\n{'='*62}")
            print(f"  START  {label}")
            print(f"  log → {log_path}")
            print(f"{'='*62}", flush=True)

        if dry_run:
            with lock:
                print(f"DRY RUN: {' '.join(str(c) for c in cmd)}")
            job_queue.task_done()
            continue

        with open(log_path, "w") as lf:
            lf.write("CMD: " + " ".join(str(c) for c in cmd) + "\n\n")
            lf.flush()
            result = subprocess.run(cmd, stdout=lf, stderr=lf, env=env)

        status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
        with lock:
            print(f"  {status}  {label}", flush=True)

        job_queue.task_done()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)),
                   help="Seeds to run (default 0–9)")
    p.add_argument("--gpus", type=int, nargs="+", default=[0, 1],
                   help="GPU IDs to use (default: 0 1)")
    p.add_argument("--only", choices=["fig3", "fig4"], default=None,
                   help="Run only one figure's seeds (default: both)")
    p.add_argument("--dry_run", action="store_true",
                   help="Print commands without running anything")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip seeds whose sentinel output file already exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    HEAD_OUT.mkdir(parents=True, exist_ok=True)
    RELOC_OUT.mkdir(parents=True, exist_ok=True)

    # Build job list — Fig 4 first so long jobs start immediately on both GPUs
    jobs: list[tuple[int, int]] = []
    figs = [4, 3] if args.only is None else ([4] if args.only == "fig4" else [3])
    for fig in figs:
        for seed in args.seeds:
            if args.skip_existing:
                done = _already_done_fig3(seed) if fig == 3 else _already_done_fig4(seed)
                if done:
                    print(f"[skip] Fig{fig}/seed{seed} — already complete")
                    continue
            jobs.append((fig, seed))

    if not jobs:
        print("All requested seeds already complete. Nothing to do.")
        return

    print(f"\nQueued {len(jobs)} jobs across {len(args.gpus)} GPU(s): {args.gpus}")
    for fig, seed in jobs:
        print(f"  Fig{fig}/seed{seed}")

    jq: queue.Queue[tuple[int, int]] = queue.Queue()
    for job in jobs:
        jq.put(job)

    lock = threading.Lock()
    threads = [
        threading.Thread(target=worker, args=(gpu_id, jq, args.dry_run, lock), daemon=True)
        for gpu_id in args.gpus
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"\n{'='*62}")
    print(f"All jobs finished.  Head outputs: {HEAD_OUT}")
    print(f"                    Reloc outputs: {RELOC_OUT}")


if __name__ == "__main__":
    main()
