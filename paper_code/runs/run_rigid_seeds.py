#!/usr/bin/env python3
"""
Rigid-weight control seeds (rnn_rl_lr = 2e-6) for the "plasticity is necessary"
claim.

Every existing run in Repro/ uses rnn_rl_lr = 5e-5. The only rigid-weight run in
the repo at the time of writing is single-seed. This runner produces a 20-seed
rigid condition MATCHED to the no-head condition on every other axis, so
"plasticity gates remapping" becomes a clean multi-seed contrast rather than an
n=1 anecdote.

The match is enforced STRUCTURALLY: RIGID_BASE is derived from
run_fig23_extra_seeds.FIG2_BASE by swapping a single value, and an assertion
checks that the symmetric difference is exactly {"5e-5", "2e-6"}. Editing the
no-head arg list without editing this one is therefore a loud failure, not a
silent drift.

Usage:
    python paper_code/runs/run_rigid_seeds.py --seeds 42 --dry_run      # inspect
    python paper_code/runs/run_rigid_seeds.py --seeds 42                # smoke test
    python paper_code/runs/run_rigid_seeds.py --seeds $(seq 0 19) --prune
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
sys.path.insert(0, str(REPO / "paper_code" / "lib"))

from run_fig23_extra_seeds import FIG2_BASE  # noqa: E402  (canonical no-head args)

V11_SCRIPT = REPO / "paper_code" / "model" / "ratinabox_v11.py"
RIGID_OUT = REPO / "Repro" / "multi_seed_rigid"

RIGID_LR = "2e-6"
NOHEAD_LR = "5e-5"

# Derive rather than copy: one value differs, and we assert exactly that.
RIGID_BASE: list[str] = list(FIG2_BASE)
_lr_idx = RIGID_BASE.index("--rnn_rl_lr") + 1
assert RIGID_BASE[_lr_idx] == NOHEAD_LR, (
    f"expected no-head lr {NOHEAD_LR!r}, found {RIGID_BASE[_lr_idx]!r} — "
    "run_fig23_extra_seeds.FIG2_BASE changed; re-check the match before running."
)
RIGID_BASE[_lr_idx] = RIGID_LR

_diff = set(FIG2_BASE) ^ set(RIGID_BASE)
assert _diff == {NOHEAD_LR, RIGID_LR}, (
    f"rigid args differ from no-head args by more than the learning rate: {_diff}"
)

# Periods the rigid condition is actually used for: the frozen-replay pair (the
# dissociation panel, the entropy gate) and phase A/B (figures 2/3 contrasts).
PRUNE_KEEP = ["replay_c1_pred", "replay_c1_postrl", "cycle_1", "cycle_1_goal"]


def _already_done(seed: int) -> bool:
    """Completion sentinel.

    Deliberately metrics.json, NOT df_activations_cycle_1.pkl (which the other
    runners use): cycle_1 is written ~5 minutes before the run actually
    finishes, so a crash in that window would be silently treated as complete.
    metrics.json is written last.
    """
    return (RIGID_OUT / f"rigid_seed{seed}" / "metrics.json").exists()


def _build_cmd(seed: int):
    run_name = f"rigid_seed{seed}"
    log = RIGID_OUT / f"{run_name}_runner.log"
    cmd = [sys.executable, str(V11_SCRIPT),
           "--seed", str(seed), "--output_dir", str(RIGID_OUT),
           "--run_name", run_name, *RIGID_BASE]
    return cmd, log, run_name


def _prune(run_dir: Path, lock: threading.Lock) -> None:
    cmd = [sys.executable, str(REPO / "paper_code" / "runs" / "prune_run.py"),
           "--run-dir", str(run_dir), "--keep", *PRUNE_KEEP]
    res = subprocess.run(cmd, capture_output=True, text=True)
    with lock:
        if res.returncode == 0:
            print(f"  pruned {run_dir.name}", flush=True)
        else:
            print(f"  PRUNE FAILED {run_dir.name}: {res.stderr[-400:]}", flush=True)


def worker(gpu_id: int, job_q: "queue.Queue", results: list,
           lock: threading.Lock, dry_run: bool, prune: bool):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
    while True:
        try:
            seed = job_q.get_nowait()
        except queue.Empty:
            return
        cmd, log, run_name = _build_cmd(seed)
        RIGID_OUT.mkdir(parents=True, exist_ok=True)
        label = f"rigid/seed{seed}"

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

        ok = result.returncode == 0
        if ok and prune:
            _prune(RIGID_OUT / run_name, lock)

        with lock:
            status = "OK" if ok else f"FAIL(rc={result.returncode})"
            print(f"\n{'='*60}\n  {status}  {label} -> GPU{gpu_id}\n{'='*60}", flush=True)
            results.append((label, ok))
        job_q.task_done()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    p.add_argument("--skip_existing", action="store_true", default=True)
    p.add_argument("--prune", action="store_true",
                   help=f"after each run, keep only {PRUNE_KEEP} activations (~2.6GB -> ~1.25GB)")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"rigid lr = {RIGID_LR} (no-head reference: {NOHEAD_LR})")
    print(f"arg-list difference vs no-head: {sorted(_diff)}  [asserted]")

    job_q: "queue.Queue" = queue.Queue()
    n_queued = 0
    for seed in args.seeds:
        if args.skip_existing and _already_done(seed):
            print(f"  skip seed {seed} (metrics.json present)")
            continue
        job_q.put(seed)
        n_queued += 1

    print(f"Queued {n_queued} jobs across {len(args.gpus)} GPU(s): {args.gpus}")

    results: list = []
    lock = threading.Lock()
    threads = [threading.Thread(target=worker,
                                args=(gpu, job_q, results, lock, args.dry_run, args.prune))
               for gpu in args.gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if not args.dry_run:
        ok = [r for r, s in results if s]
        bad = [r for r, s in results if not s]
        print(f"\n{'='*60}\nAll jobs finished.  OK: {len(ok)}  Failed: {bad}")
        print(f"  Rigid outputs: {RIGID_OUT}")


if __name__ == "__main__":
    main()
