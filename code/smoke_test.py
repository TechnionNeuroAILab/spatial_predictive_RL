#!/usr/bin/env python3
"""Smoke test for the code/ pipeline.

Verifies that the code *runs* -- it deliberately does not regenerate any figure
(that is slow, and it would overwrite the PDFs the manuscript currently points at).

side-effect free by construction, and that took a bug to learn: scripts without
an `if __name__ == "__main__"` guard do their work at *import* time, and scripts
without argparse ignore `--help` and do their work anyway. An earlier version of
this file hit both -- it rewrote data/sosa2025/MANIFEST.json (destroying
the recorded dataset download timestamp) and regenerated figS_sosa's PDF. Both
classes are now detected and skipped with a `warn`; see has_main_guard() and
uses_argparse(). Keep it that way.

Phases:
  deps     every third-party import resolves, + pdftoppm on $PATH (Poppler is
           shelled out to by every figure assembler)
  imports  every library module and every figure dir's utils/analysis import
           cleanly, using the same sys.path shims the generators use
  cli      `--help` on every entry point (catches import + argparse breakage)
  data     the run dirs and JSON caches each figure reads actually exist
  paper    manuscript figures, methods constants and reported statistics agree
           with the publication pipeline and committed analysis caches
  micro    (--micro-run only) a tiny ratinabox_v11 training run in a temp dir,
           proving the simulator executes and not merely imports

Usage:
    python code/smoke_test.py                # deps + imports + cli + data
    python code/smoke_test.py --micro-run    # also run the simulator
    python code/smoke_test.py --phase cli
Exit code is non-zero if anything failed.
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PC = Path(__file__).resolve().parent
REPO = PC.parent

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
OK, FAIL, WARN = f"{GREEN}ok{RESET}", f"{RED}FAIL{RESET}", f"{YELLOW}warn{RESET}"

results: list[tuple[str, str, str]] = []   # (phase, name, status)


def record(phase: str, name: str, status: str, detail: str = "") -> None:
    results.append((phase, name, status))
    line = f"  [{status}] {name}"
    if detail:
        line += f"\n        {DIM}{detail}{RESET}"
    print(line, flush=True)


# --------------------------------------------------------------------- deps
THIRD_PARTY = [
    "numpy", "pandas", "matplotlib", "scipy", "torch", "ratinabox",
    "seaborn", "sklearn", "shapely", "tqdm", "h5py", "openpyxl", "pypdf", "PIL",
]


def phase_deps() -> None:
    print("\n== deps ==")
    for mod in THIRD_PARTY:
        try:
            m = __import__(mod)
            record("deps", f"{mod} {getattr(m, '__version__', '')}", OK)
        except Exception as e:
            record("deps", mod, FAIL, f"{type(e).__name__}: {e}")
    if shutil.which("pdftoppm"):
        record("deps", "pdftoppm (poppler-utils)", OK)
    else:
        record("deps", "pdftoppm (poppler-utils)", FAIL,
               "every figure assembler shells out to it; install poppler-utils")
    if shutil.which("mutool"):
        record("deps", "mutool (mupdf-tools)", OK)
    else:
        record("deps", "mutool (mupdf-tools)", FAIL,
               "check_figure_type.py requires it; install mupdf-tools")


# ------------------------------------------------------------------ imports
def has_main_guard(path: Path) -> bool:
    """True if the file's work is behind `if __name__ == "__main__"`.

    Files without a guard execute their work at import time. This matters: an
    early version of this smoke test imported every file under external/, which
    executed build_manifest.py's unguarded top-level code and rewrote
    data/sosa2025/MANIFEST.json, replacing the real dataset download
    timestamp with the time the test ran. Never import an unguarded script.
    """
    return '__name__ == "__main__"' in path.read_text() or \
           "__name__ == '__main__'" in path.read_text()


def uses_argparse(path: Path) -> bool:
    """True if the script actually parses argv.

    A script without argparse ignores `--help` and just does its work: that is
    how the same early version regenerated figS_sosa's PDF, because
    generate_figure_sosa.py takes no arguments at all. Only pass --help to
    scripts that will honour it.
    """
    src = path.read_text()
    return "argparse" in src and "add_argument" in src


def _import_isolated(path: Path, extra_paths: list[Path]) -> None:
    """Import one module by file path in a subprocess.

    A subprocess is required, not fastidiousness: ten figure dirs each ship a
    module named `utils` (and seven a module named `analysis`), and the
    generators rely on sys.path ordering + sys.modules juggling to pick the
    right one. Importing them all into a single interpreter would resolve the
    wrong `utils` for some of them and report failures that don't exist in
    real use -- each generator runs in its own process.
    """
    pre = "".join(f"sys.path.insert(0, {str(p)!r})\n" for p in extra_paths)
    code = (
        "import sys, importlib.util\n" + pre +
        f"spec = importlib.util.spec_from_file_location('m', {str(path)!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        # The documented recipe registers the module before executing it.
        # Without this, `dataclasses` resolving a string annotation does
        # sys.modules.get(cls.__module__).__dict__ and dies on None -- a
        # failure in the harness, not in the module under test.
        "sys.modules['m'] = mod\n"
        "spec.loader.exec_module(mod)\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=str(path.parent), timeout=300)
    if r.returncode != 0:
        tail = (r.stderr or "").strip().split("\n")
        raise RuntimeError(tail[-1] if tail else "non-zero exit")


def phase_imports() -> None:
    print("\n== imports ==")
    for p in sorted((PC / "lib").glob("*.py")):
        try:
            _import_isolated(p, [PC / "lib"])
            record("imports", f"lib/{p.name}", OK)
        except Exception as e:
            record("imports", f"lib/{p.name}", FAIL, str(e))

    for d in sorted((PC / "figures").iterdir()):
        if not d.is_dir():
            continue
        for name in ("utils.py", "analysis.py"):
            p = d / name
            if not p.exists():
                continue
            try:
                _import_isolated(p, [d, PC / "lib"])
                record("imports", f"figures/{d.name}/{name}", OK)
            except Exception as e:
                record("imports", f"figures/{d.name}/{name}", FAIL, str(e))

    for p in sorted((PC / "external").rglob("*.py")):
        label = f"external/{p.parent.name}/{p.name}"
        if not has_main_guard(p):
            record("imports", label, WARN,
                   "no __main__ guard: does its work at import time, so this "
                   "test will NOT import it (it would rewrite data files)")
            continue
        try:
            _import_isolated(p, [p.parent])
            record("imports", label, OK)
        except Exception as e:
            record("imports", label, FAIL, str(e))


# ---------------------------------------------------------------------- cli
def _entry_points() -> list[Path]:
    eps: list[Path] = []
    eps += sorted((PC / "figures").glob("*/generate_*.py"))
    eps += sorted((PC / "runs").glob("*.py"))
    eps += sorted((PC / "aggregate").glob("*.py"))
    eps += sorted((PC / "model").glob("*.py"))
    eps += sorted((PC / "tools").glob("*.py"))
    eps += sorted((PC / "external").rglob("analysis*.py"))
    eps.append(PC / "sync_paper_figures.py")
    eps.append(PC / "verify_paper.py")
    return eps


def phase_cli() -> None:
    print("\n== cli (--help) ==")
    for p in _entry_points():
        rel = str(p.relative_to(PC))
        # Executing a script that ignores argv would do its work -- regenerating
        # figures or rewriting data. Only --help scripts that parse argv.
        if not uses_argparse(p):
            # Fall back to an import test where that is safe: with the work
            # behind a __main__ guard, importing exercises the same module-level
            # path resolution and imports without running anything.
            if has_main_guard(p):
                extra = [p.parent, PC / "lib"]
                try:
                    _import_isolated(p, extra)
                    record("cli", rel, OK, "no argparse; import-tested instead")
                except Exception as e:
                    record("cli", rel, FAIL, str(e))
            else:
                record("cli", rel, WARN,
                       "no argparse and no __main__ guard: would run for real, "
                       "so it is not executed here")
            continue
        try:
            r = subprocess.run([sys.executable, str(p), "--help"],
                               capture_output=True, text=True, timeout=300,
                               cwd=str(p.parent))
            if r.returncode == 0:
                record("cli", rel, OK)
            else:
                tail = (r.stderr or r.stdout or "").strip().split("\n")
                record("cli", rel, FAIL, tail[-1] if tail else f"exit {r.returncode}")
        except subprocess.TimeoutExpired:
            record("cli", rel, FAIL, "timeout")
        except Exception as e:
            record("cli", rel, FAIL, str(e))


# --------------------------------------------------------------------- data
# What each figure reads. Paths are relative to the repo root.
DATA_REQS: dict[str, list[str]] = {
    "fig1/figS_pipeline":  ["results/multi_seed",
                            "results/multi_seed/nohead_seed16"],
    "fig2":                ["results/multi_seed", "results/multi_seed/nohead_seed16"],
    "fig3":                ["results/multi_seed_head", "results/multi_seed_head/head_seed28"],
    "fig4/figS_reloc":     ["results/multi_seed_reloc",
                            "results/multi_seed_reloc/reloc_seed40"],
    "fig5 (data confr.)":  ["results/aggregate/extinction_summary.json",
                            "data/krishnan2022/cache/krishnan_data.json",
                            "data/sosa2025/cache/sosa_tests.json",
                            "results/multi_seed_reloc/aggregate/reward_relative_summary.json"],
    "figS_discriminator":  ["results/aggregate/occupancy_matched_estimator.json"],
    "figS_extinction":     ["results/aggregate/extinction_summary.json",
                            "data/krishnan2022/cache/krishnan_data.json"],
    "figS_sr":             ["results/multi_seed", "results/multi_seed_head"],
    "figS_rewardvector":   ["results/multi_seed", "results/multi_seed_head", "results/multi_seed_reloc"],
    "figS_sosa":           ["data/sosa2025/cache/sosa_tests.json"],
}

def phase_data() -> None:
    print("\n== data ==")
    for label, needs in DATA_REQS.items():
        missing = [n for n in needs if not (REPO / n).exists()]
        if missing:
            record("data", label, FAIL, "missing: " + ", ".join(missing))
        else:
            record("data", label, OK)


# -------------------------------------------------------------------- paper
def phase_paper() -> None:
    print("\n== paper/code consistency ==")
    script = PC / "verify_paper.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(REPO),
    )
    if result.returncode == 0:
        summary = result.stdout.strip().splitlines()[-1]
        record("paper", "verify_paper.py", OK, summary)
    else:
        detail = (result.stderr or result.stdout).strip().splitlines()
        record("paper", "verify_paper.py", FAIL, "\n        ".join(detail[-8:]))


# -------------------------------------------------------------------- micro
def phase_micro() -> None:
    print("\n== micro-run (simulator) ==")
    script = PC / "model" / "ratinabox_v11.py"
    with tempfile.TemporaryDirectory(prefix="smoke_v11_") as td:
        cmd = [sys.executable, str(script),
               "--seed", "0", "--output_dir", td,
               "--n_trials", "2", "--timesteps", "5",
               "--phase1_epochs", "1", "--n_cycles", "1",
               "--max_episodes", "2", "--max_steps", "3",
               "--fixed_goal", "--replay_steps", "10",
               "--checkpoint_every", "100000", "--no_show", "--smoke"]
        print(f"  {DIM}{' '.join(cmd[:6])} ...{RESET}")
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        except subprocess.TimeoutExpired:
            record("micro", "ratinabox_v11.py", FAIL, "timeout (>30 min)")
            return
        if r.returncode == 0:
            produced = sorted(p.name for p in Path(td).rglob("*") if p.is_file())[:4]
            record("micro", "ratinabox_v11.py", OK,
                   f"produced {len(produced)}+ files, e.g. {', '.join(produced)}")
        else:
            tail = (r.stderr or r.stdout or "").strip().split("\n")
            record("micro", "ratinabox_v11.py", FAIL, "\n        ".join(tail[-4:]))


PHASES = {"deps": phase_deps, "imports": phase_imports, "cli": phase_cli,
          "data": phase_data, "paper": phase_paper, "micro": phase_micro}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=sorted(PHASES), action="append",
                    help="run only this phase (repeatable)")
    ap.add_argument("--micro-run", action="store_true",
                    help="also run the simulator micro-run phase")
    args = ap.parse_args()

    if args.phase:
        chosen = args.phase
    else:
        chosen = ["deps", "imports", "cli", "data", "paper"]
        if args.micro_run:
            chosen.append("micro")

    print(f"code smoke test — repo {REPO}")
    print(f"python {sys.version.split()[0]}  ({sys.executable})")
    for name in chosen:
        PHASES[name]()

    n_fail = sum(1 for _, _, s in results if s == FAIL)
    n_warn = sum(1 for _, _, s in results if s == WARN)
    n_ok = sum(1 for _, _, s in results if s == OK)
    print(f"\n{'=' * 60}\n{n_ok} ok, {n_warn} warn, {n_fail} FAIL")
    if n_fail:
        print("\nfailures:")
        for phase, name, s in results:
            if s == FAIL:
                print(f"  {phase}: {name}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
