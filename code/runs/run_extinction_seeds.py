#!/usr/bin/env python3
"""
Extinction seeds: head-ON learning at A (cycle 1, matched seed and
hyperparameters), then a cycle with reward delivery and the head objective
switched off (see code/model/ratinabox_extinction.py), with a frozen-replay
triple.  The simulator preserves the omission cohort's historical one-step PPO
state-recomputation convention, so cycle 1 is not byte-identical to the main
head condition; the manuscript Methods documents this exception.

Matching is enforced structurally, but not with run_rigid_seeds.py's
set-symmetric-difference trick — that breaks here because the changed values
("1", "2000", ...) collide with other flags' values in a flat token set.
Instead the arg list is parsed into a flag->value dict and the dict diff is
asserted to be exactly the three intended changes.

Usage:
    python code/runs/run_extinction_seeds.py --seeds 0 --dry_run
    python code/runs/run_extinction_seeds.py --seeds 0            # smoke test
    python code/runs/run_extinction_seeds.py --seeds $(seq 0 19) --prune
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
sys.path.insert(0, str(REPO / "code" / "lib"))

from run_fig23_extra_seeds import FIG3_BASE  # noqa: E402  (canonical head-ON args)

EXT_SCRIPT = REPO / "code" / "model" / "ratinabox_extinction.py"
EXT_OUT = REPO / "results" / "multi_seed_ext"


def _parse_flags(tokens: list[str]) -> dict[str, str | bool]:
    """Flat CLI token list -> {flag: value-or-True(store_true)} dict."""
    out: dict[str, str | bool] = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        assert tok.startswith("--"), f"expected flag, got {tok!r}"
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            out[tok] = tokens[i + 1]
            i += 2
        else:
            out[tok] = True
            i += 1
    return out


# Derive the extinction args from the head-ON base by exactly three changes.
_head = _parse_flags(FIG3_BASE)
_ext = dict(_head)
_ext["--n_cycles"] = "2"
_ext["--max_episodes"] = "2000"
_ext["--extinguish_at_cycle"] = "1"

_diff = {k: (_head.get(k), _ext.get(k))
         for k in set(_head) | set(_ext) if _head.get(k) != _ext.get(k)}
assert _diff == {
    "--n_cycles": ("1", "2"),
    "--max_episodes": ("1000", "2000"),
    "--extinguish_at_cycle": (None, "1"),
}, f"extinction args differ from head-ON by more than intended: {_diff}"
# Both conditions allocate 1000 episodes to their first cycle.  This checks
# protocol length, not byte identity (see the module docstring).
assert int(_ext["--max_episodes"]) // int(_ext["--n_cycles"]) == 1000

EXT_BASE: list[str] = []
for k, v in _ext.items():
    EXT_BASE.append(k)
    if v is not True:
        EXT_BASE.append(str(v))

# Everything the extinction analysis needs; the rest is pruned.
PRUNE_KEEP = ["replay_ext_pred", "replay_ext_postLearn", "replay_ext_postExt",
              "cycle_1", "cycle_1_goal", "cycle_2", "cycle_2_goal"]


def _already_done(seed: int) -> bool:
    # metrics.json is written last (cycle_1 lands ~5 min before the run ends,
    # so it would silently mark crashed runs as complete).
    return (EXT_OUT / f"ext_seed{seed}" / "metrics.json").exists()


def _build_cmd(seed: int):
    run_name = f"ext_seed{seed}"
    log = EXT_OUT / f"{run_name}_runner.log"
    cmd = [sys.executable, str(EXT_SCRIPT),
           "--seed", str(seed), "--output_dir", str(EXT_OUT),
           "--run_name", run_name, *EXT_BASE]
    return cmd, log, run_name


def _prune(run_dir: Path, lock: threading.Lock) -> None:
    cmd = [sys.executable, str(REPO / "code" / "runs" / "prune_run.py"),
           "--run-dir", str(run_dir), "--keep", *PRUNE_KEEP]
    res = subprocess.run(cmd, capture_output=True, text=True)
    with lock:
        print(f"  {'pruned' if res.returncode == 0 else 'PRUNE FAILED'} {run_dir.name}", flush=True)


def worker(gpu_id: int, job_q: "queue.Queue", results: list,
           lock: threading.Lock, dry_run: bool, prune: bool):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
    while True:
        try:
            seed = job_q.get_nowait()
        except queue.Empty:
            return
        cmd, log, run_name = _build_cmd(seed)
        EXT_OUT.mkdir(parents=True, exist_ok=True)
        label = f"ext/seed{seed}"

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
            _prune(EXT_OUT / run_name, lock)

        with lock:
            print(f"\n{'='*60}\n  {'OK' if ok else f'FAIL(rc={result.returncode})'}  {label} -> GPU{gpu_id}\n{'='*60}", flush=True)
            results.append((label, ok))
        job_q.task_done()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    p.add_argument("--skip_existing", action="store_true", default=True)
    p.add_argument("--prune", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    print(f"arg diff vs head-ON: {_diff}  [asserted]")
    job_q: "queue.Queue" = queue.Queue()
    n = 0
    for s in args.seeds:
        if args.skip_existing and _already_done(s):
            print(f"  skip seed {s} (metrics.json present)")
            continue
        job_q.put(s)
        n += 1
    print(f"Queued {n} jobs across {len(args.gpus)} GPU(s): {args.gpus}")

    results: list = []
    lock = threading.Lock()
    threads = [threading.Thread(target=worker, args=(g, job_q, results, lock, args.dry_run, args.prune))
               for g in args.gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if not args.dry_run:
        bad = [r for r, s in results if not s]
        print(f"\n{'='*60}\nAll jobs finished.  OK: {sum(s for _, s in results)}  Failed: {bad}")
        print(f"  Extinction outputs: {EXT_OUT}")


if __name__ == "__main__":
    main()
