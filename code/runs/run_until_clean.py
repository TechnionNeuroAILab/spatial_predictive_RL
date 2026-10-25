#!/usr/bin/env python3
"""Train until N entropy-gate-passing networks exist for a family.

A training job that exits 0 is not a pass. A pass is seed_is_clean on the
periods that family uses. Seed indices are shared across matched arms and are
never renumbered, so pairing by index stays valid. Failed runs stay on disk.

Families
  paired     no-head + head + rigid at the same seed. Pass = both plastic arms
             clean on the four interaction periods and the rigid arm clean on
             the two frozen-replay periods.
  reloc      relocation seeds. Pass = replay triple + reward-vector periods.
  extinction omission seeds. Pass = extinction replay triple. The matching
             head run for that index is trained if missing; it is not added
             to the paired cohort.

Usage:
    python code/runs/run_until_clean.py --family paired --target 50
    python code/runs/run_until_clean.py --family reloc --target 50 --gpu 0 --jobs 6
    python code/runs/run_until_clean.py --family paired --target 50 --gpu 0,1 --jobs 12
    python code/runs/run_until_clean.py --family paired --target 1 --smoke
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNS))
sys.path.insert(0, str(REPO / "code" / "lib"))

from analysis_helpers import seed_is_clean  # noqa: E402
from run_fig23_extra_seeds import FIG2_BASE, FIG3_BASE  # noqa: E402
from run_fig34_parallel import FIG4_BASE  # noqa: E402
from run_rigid_seeds import RIGID_BASE, RIGID_OUT  # noqa: E402
from run_extinction_seeds import EXT_BASE, EXT_OUT  # noqa: E402

V11 = REPO / "code" / "model" / "ratinabox_v11.py"
RELOC = REPO / "code" / "model" / "ratinabox_relocation.py"
EXT = REPO / "code" / "model" / "ratinabox_extinction.py"

NOHEAD_OUT = REPO / "results" / "multi_seed"
HEAD_OUT = REPO / "results" / "multi_seed_head"
RELOC_OUT = REPO / "results" / "multi_seed_reloc"

# Strictest gate each family uses (see paper Methods / plan).
PAIRED_PLASTIC_PERIODS = [
    "cycle_1", "cycle_1_goal", "replay_c1_pred", "replay_c1_postrl",
]
PAIRED_RIGID_PERIODS = ["replay_c1_pred", "replay_c1_postrl"]
RELOC_PERIODS = [
    "replay_reloc_pred", "replay_reloc_postA", "replay_reloc_postB",
    "cycle_1_goal", "cycle_2_goal", "goal_conditioned",
]
EXT_PERIODS = ["replay_ext_pred", "replay_ext_postLearn", "replay_ext_postExt"]

# Tiny hyperparameters for --smoke (same scale as smoke_test.py --micro-run).
SMOKE_OVERRIDES: dict[str, str | bool] = {
    "--n_trials": "2",
    "--timesteps": "5",
    "--phase1_epochs": "1",
    "--max_episodes": "2",
    "--max_steps": "3",
    "--n_cycles": "1",
    "--interleave_epochs": "1",
    "--replay_steps": "10",
    "--checkpoint_every": "100000",
}


def _override_args(base: list[str], overrides: dict[str, str | bool]) -> list[str]:
    """Replace flag values in a flat CLI token list; append missing flags."""
    out = list(base)
    seen: set[str] = set()
    i = 0
    while i < len(out):
        tok = out[i]
        if tok in overrides:
            seen.add(tok)
            val = overrides[tok]
            if val is True:
                i += 1
                continue
            if i + 1 < len(out) and not out[i + 1].startswith("--"):
                out[i + 1] = str(val)
                i += 2
            else:
                out.insert(i + 1, str(val))
                i += 2
            continue
        i += 1
    for k, v in overrides.items():
        if k in seen:
            continue
        out.append(k)
        if v is not True:
            out.append(str(v))
    # Relocation / extinction need two cycles even in smoke so their periods exist.
    return out


def _smoke_reloc_args(base: list[str]) -> list[str]:
    o = dict(SMOKE_OVERRIDES)
    o["--n_cycles"] = "2"
    o["--max_episodes"] = "4"
    return _override_args(base, o)


def _smoke_ext_args(base: list[str]) -> list[str]:
    o = dict(SMOKE_OVERRIDES)
    o["--n_cycles"] = "2"
    o["--max_episodes"] = "4"
    o["--extinguish_at_cycle"] = "1"
    return _override_args(base, o)


def _run_dir(out_root: Path, prefix: str, seed: int) -> Path:
    return out_root / f"{prefix}seed{seed}"


def _done(out_root: Path, prefix: str, seed: int) -> bool:
    """metrics.json is written last; preferred completion sentinel."""
    return (_run_dir(out_root, prefix, seed) / "metrics.json").exists()


def _clean(out_root: Path, prefix: str, seed: int, periods: list[str]) -> bool:
    data = _run_dir(out_root, prefix, seed) / "data"
    if not data.is_dir():
        return False
    return seed_is_clean(str(data), periods)


def _build_cmd(script: Path, out_root: Path, prefix: str, seed: int,
               base: list[str]) -> list[str]:
    run_name = f"{prefix}seed{seed}"
    return [
        sys.executable, str(script),
        "--seed", str(seed),
        "--output_dir", str(out_root),
        "--run_name", run_name,
        *base,
    ]


def _launch(cmd: list[str], log_path: Path, gpu: int | None, dry_run: bool) -> int:
    if dry_run:
        print(f"DRY RUN: {' '.join(cmd)}")
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ}
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with open(log_path, "w") as lf:
        lf.write("CMD: " + " ".join(cmd) + "\n\n")
        lf.flush()
        return subprocess.run(cmd, stdout=lf, stderr=lf, env=env).returncode


def paired_pass(seed: int) -> bool:
    return (
        _clean(NOHEAD_OUT, "nohead_", seed, PAIRED_PLASTIC_PERIODS)
        and _clean(HEAD_OUT, "head_", seed, PAIRED_PLASTIC_PERIODS)
        and _clean(RIGID_OUT, "rigid_", seed, PAIRED_RIGID_PERIODS)
    )


def reloc_pass(seed: int) -> bool:
    return _clean(RELOC_OUT, "reloc_", seed, RELOC_PERIODS)


def extinction_pass(seed: int) -> bool:
    return _clean(EXT_OUT, "ext_", seed, EXT_PERIODS)


def _count_existing_passes(family: str) -> tuple[list[int], list[int]]:
    """Return (passed_seeds, launched_seeds) already on disk for this family."""
    if family == "paired":
        # Any of the three dirs may hold launched indices; union them.
        launched = set()
        for root, pref in ((NOHEAD_OUT, "nohead_"), (HEAD_OUT, "head_"),
                           (RIGID_OUT, "rigid_")):
            if not root.exists():
                continue
            for p in root.glob(f"{pref}seed*"):
                if p.is_dir():
                    try:
                        launched.add(int(p.name.replace(f"{pref}seed", "")))
                    except ValueError:
                        pass
        launched_l = sorted(launched)
        complete = [
            s for s in launched_l
            if _done(NOHEAD_OUT, "nohead_", s)
            and _done(HEAD_OUT, "head_", s)
            and _done(RIGID_OUT, "rigid_", s)
        ]
        passed = [s for s in complete if paired_pass(s)]
        return passed, launched_l
    if family == "reloc":
        launched = []
        if RELOC_OUT.exists():
            for p in sorted(RELOC_OUT.glob("reloc_seed*")):
                if p.is_dir():
                    try:
                        launched.append(int(p.name.replace("reloc_seed", "")))
                    except ValueError:
                        pass
        complete = [s for s in launched if _done(RELOC_OUT, "reloc_", s)]
        passed = [s for s in complete if reloc_pass(s)]
        return passed, launched
    # extinction
    launched = []
    if EXT_OUT.exists():
        for p in sorted(EXT_OUT.glob("ext_seed*")):
            if p.is_dir():
                try:
                    launched.append(int(p.name.replace("ext_seed", "")))
                except ValueError:
                    pass
        complete = [s for s in launched if _done(EXT_OUT, "ext_", s)]
        passed = [s for s in complete if extinction_pass(s)]
        return passed, launched


def _next_seed(launched: list[int], start: int = 0) -> int:
    used = set(launched)
    s = start
    while s in used:
        s += 1
    return s


def _write_manifest(family: str, entries: list[dict], target: int) -> Path:
    out = REPO / "results" / "aggregate"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"until_clean_{family}.json"
    payload = {
        "family": family,
        "target": target,
        "updated": datetime.now(timezone.utc).isoformat(),
        "n_passed": sum(1 for e in entries if e.get("pass")),
        "n_launched": len(entries),
        "entries": entries,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def _paired_arms(fig2_base, fig3_base, rigid_base) -> list[tuple]:
    return [
        (V11, NOHEAD_OUT, "nohead_", fig2_base),
        (V11, HEAD_OUT, "head_", fig3_base),
        (V11, RIGID_OUT, "rigid_", rigid_base),
    ]


def _reloc_arms(fig4_base) -> list[tuple]:
    return [(RELOC, RELOC_OUT, "reloc_", fig4_base)]


def _ext_arms(fig3_base, ext_base) -> list[tuple]:
    # Matching head partner if missing; not counted as a paired passer.
    return [
        (V11, HEAD_OUT, "head_", fig3_base),
        (EXT, EXT_OUT, "ext_", ext_base),
    ]


def _missing_jobs(arms: list[tuple], seed: int, skip_existing: bool) -> list[dict]:
    jobs = []
    for script, out_root, prefix, base in arms:
        if skip_existing and _done(out_root, prefix, seed):
            continue
        jobs.append({
            "script": script, "out_root": out_root, "prefix": prefix,
            "seed": seed, "base": base,
        })
    return jobs


def _evaluate_seed(family: str, seed: int, dry_run: bool) -> tuple[bool, str]:
    if dry_run:
        return False, "dry_run"
    if family == "paired":
        if not all(_done(r, p, seed) for r, p in
                   ((NOHEAD_OUT, "nohead_"), (HEAD_OUT, "head_"), (RIGID_OUT, "rigid_"))):
            return False, "incomplete"
        if paired_pass(seed):
            return True, "pass"
        return False, "gate_fail"
    if family == "reloc":
        if not _done(RELOC_OUT, "reloc_", seed):
            return False, "incomplete"
        if reloc_pass(seed):
            return True, "pass"
        return False, "gate_fail"
    if not _done(EXT_OUT, "ext_", seed):
        return False, "incomplete"
    if extinction_pass(seed):
        return True, "pass"
    return False, "gate_fail"


def _run_job(job: dict, gpu: int | None, dry_run: bool) -> tuple[dict, int]:
    script, out_root, prefix, seed, base = (
        job["script"], job["out_root"], job["prefix"], job["seed"], job["base"])
    out_root.mkdir(parents=True, exist_ok=True)
    cmd = _build_cmd(script, out_root, prefix, seed, base)
    log = out_root / f"{prefix}seed{seed}_runner.log"
    print(f"  START {prefix}seed{seed}  GPU={gpu}", flush=True)
    rc = _launch(cmd, log, gpu, dry_run)
    print(f"  {'OK' if rc == 0 else f'FAIL rc={rc}'} {prefix}seed{seed}", flush=True)
    return job, rc


def run_family(family: str, target: int, gpus: list[int], jobs: int,
               dry_run: bool, skip_existing: bool, smoke: bool,
               smoke_root: Path | None) -> int:
    """Train until `target` passers, spreading up to `jobs` processes across `gpus`."""
    global NOHEAD_OUT, HEAD_OUT, RIGID_OUT, RELOC_OUT, EXT_OUT

    bases = {
        "fig2": FIG2_BASE, "fig3": FIG3_BASE, "rigid": RIGID_BASE,
        "fig4": FIG4_BASE, "ext": EXT_BASE,
    }

    if smoke:
        assert smoke_root is not None
        NOHEAD_OUT = smoke_root / "multi_seed"
        HEAD_OUT = smoke_root / "multi_seed_head"
        RIGID_OUT = smoke_root / "multi_seed_rigid"
        RELOC_OUT = smoke_root / "multi_seed_reloc"
        EXT_OUT = smoke_root / "multi_seed_ext"
        bases["fig2"] = _override_args(FIG2_BASE, SMOKE_OVERRIDES)
        bases["fig3"] = _override_args(FIG3_BASE, SMOKE_OVERRIDES)
        bases["rigid"] = _override_args(RIGID_BASE, SMOKE_OVERRIDES)
        bases["fig4"] = _smoke_reloc_args(FIG4_BASE)
        bases["ext"] = _smoke_ext_args(EXT_BASE)
        jobs = min(jobs, 3)

    passed, launched = _count_existing_passes(family)
    entries = {
        s: {"seed": s, "pass": s in passed,
            "status": "pass" if s in passed else "pending"}
        for s in launched
    }
    gpu_label = ",".join(str(g) for g in gpus) if gpus else "cpu"
    print(f"family={family}  already passed={len(passed)}/{target}  "
          f"known seeds={len(launched)}  gpus={gpu_label}  jobs={jobs}")

    if family == "paired":
        arms_for = lambda seed: _paired_arms(bases["fig2"], bases["fig3"], bases["rigid"])
    elif family == "reloc":
        arms_for = lambda seed: _reloc_arms(bases["fig4"])
    else:
        def arms_for(seed):
            arms = []
            if not _done(HEAD_OUT, "head_", seed):
                arms.append((V11, HEAD_OUT, "head_", bases["fig3"]))
            arms.append((EXT, EXT_OUT, "ext_", bases["ext"]))
            return arms

    known = set(launched)
    queued_keys: set[tuple] = set()  # (prefix, seed) actually submitted
    crashed_seeds: set[int] = set()
    in_flight: dict = {}
    stop_new = False
    max_new = 1 if smoke else 10_000
    n_new = 0
    gpu_assign = itertools.cycle(gpus if gpus else [None])

    def submit_job(ex: ThreadPoolExecutor, job: dict) -> str:
        """'started', 'skip' (already tried or its seed crashed), or 'full'."""
        key = (job["prefix"], job["seed"])
        if job["seed"] in crashed_seeds or key in queued_keys:
            return "skip"
        if len(in_flight) >= jobs:
            return "full"
        gpu = next(gpu_assign)
        queued_keys.add(key)
        fut = ex.submit(_run_job, job, gpu, dry_run)
        in_flight[fut] = job
        return "started"

    def _queue(ex: ThreadPoolExecutor, job: dict) -> bool:
        """Submit one job. False means the worker slots are full."""
        status = submit_job(ex, job)
        return status != "full"

    def fill(ex: ThreadPoolExecutor) -> None:
        nonlocal n_new, stop_new
        # 1) finish missing arms of known seeds
        for seed in sorted(known):
            if len(in_flight) >= jobs:
                return
            if seed in passed or seed in crashed_seeds:
                continue
            for job in _missing_jobs(arms_for(seed), seed, skip_existing):
                if not _queue(ex, job):
                    return
        # 2) start new seeds until we have enough passers (smoke: at most one)
        while (not stop_new and len(passed) < target and len(in_flight) < jobs
               and n_new < max_new):
            seed = _next_seed(sorted(known))
            known.add(seed)
            n_new += 1
            print(f"--- enqueue seed {seed}  (passed {len(passed)}/{target}) ---",
                  flush=True)
            for job in _missing_jobs(arms_for(seed), seed, skip_existing):
                if not _queue(ex, job):
                    return
            if smoke:
                stop_new = True
                break

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        fill(ex)
        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for fut in done:
                job = in_flight.pop(fut)
                try:
                    _, rc = fut.result()
                except Exception as e:
                    rc = 1
                    print(f"  EXC {job['prefix']}seed{job['seed']}: {e}", flush=True)
                seed = job["seed"]
                if rc != 0:
                    crashed_seeds.add(seed)
                    entries[seed] = {"seed": seed, "pass": False, "status": "crash",
                                     "t": time.time()}
                    print(f"  CRASH seed {seed}; continuing", flush=True)
                    if not smoke:
                        _write_manifest(family, list(entries.values()), target)
                    continue
                if seed in crashed_seeds:
                    continue
                still_missing = _missing_jobs(arms_for(seed), seed, skip_existing=True)
                if still_missing:
                    continue
                is_pass, status = _evaluate_seed(family, seed, dry_run)
                entries[seed] = {"seed": seed, "pass": is_pass, "status": status,
                                 "t": time.time()}
                if is_pass and seed not in passed:
                    passed.append(seed)
                    print(f"  PASS seed {seed}  ({len(passed)}/{target})", flush=True)
                    if len(passed) >= target:
                        stop_new = True
                elif not is_pass:
                    print(f"  {status.upper()} seed {seed}", flush=True)
                if not smoke:
                    _write_manifest(family, list(entries.values()), target)
            if len(passed) >= target:
                stop_new = True
            fill(ex)

    if smoke:
        crashed = [e for e in entries.values() if e.get("status") == "crash"]
        if crashed:
            print(f"SMOKE FAIL: crashes {crashed}")
            return 1
        print(f"SMOKE OK: trained seed(s), gate decision recorded "
              f"(pass={len(passed)}, known={len(known)})")
        return 0

    path = _write_manifest(family, list(entries.values()), target)
    print(f"\nDone. passed={len(passed)}/{target}  known={len(known)}")
    print(f"manifest -> {path}")
    return 0 if len(passed) >= target else 1


def _parse_gpus(text: str) -> list[int]:
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("expected a GPU id or comma-separated list")
    try:
        gpus = [int(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "GPU ids must be integers, comma-separated"
        ) from exc
    if any(g < 0 for g in gpus):
        raise argparse.ArgumentTypeError("GPU ids must be >= 0")
    return gpus


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", choices=["paired", "reloc", "extinction"],
                   required=True)
    p.add_argument("--target", type=int, default=50,
                   help="number of gate-passing networks to collect (default 50)")
    p.add_argument("--gpu", type=_parse_gpus, default=[0],
                   help="GPU id, or a comma-separated list to share the jobs "
                        "(default 0). Example: --gpu 0,1")
    p.add_argument("--jobs", type=int, default=6,
                   help="concurrent training processes, spread across the "
                        "listed GPUs (default 6)")
    p.add_argument("--skip_existing", action="store_true", default=True)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="tiny hyperparams in a temp dir; gate failure is OK")
    args = p.parse_args()

    if args.smoke:
        with tempfile.TemporaryDirectory(prefix="until_clean_smoke_") as td:
            return run_family(args.family, args.target, args.gpu, args.jobs,
                              args.dry_run, skip_existing=False, smoke=True,
                              smoke_root=Path(td))
    return run_family(args.family, args.target, args.gpu, args.jobs,
                      args.dry_run, skip_existing=args.skip_existing,
                      smoke=False, smoke_root=None)


if __name__ == "__main__":
    raise SystemExit(main())
