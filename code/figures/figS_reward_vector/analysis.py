"""
Reward-vector supplementary figure — cross-seed analysis helpers.

Loads per-seed results from code/lib/reward_vector_analysis.py, applies an
entropy-collapse gate re-checked on this analysis's own periods (not the
frozen-replay-triple gate used elsewhere in the paper, since this analysis's
primary tuning-curve-construction periods are goal-directed, not replay),
and caches per-condition summaries to <condition_dir>/aggregate/
reward_vector_summary.json (matching the existing project convention, e.g.
results/multi_seed/aggregate/fig2_multi_seed_summary.json).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]

_NEW_CODE = _REPO / "code" / "lib"
if str(_NEW_CODE) not in sys.path:
    sys.path.insert(0, str(_NEW_CODE))

from analysis_helpers import peak_entropy, seed_is_clean  # noqa: E402
from reward_vector_analysis import (  # noqa: E402
    analyze_period, classify_cells, geometric_rotation, circular_diff,
    bootstrap_circular_mean_ci, ring_angular_uniformity_null,
)

GOAL_DIRECTED_PERIODS = ["cycle_1_goal", "goal_conditioned"]  # this analysis's own gate periods
RELOC_GATE_PERIODS = ["cycle_1_goal", "cycle_2_goal", "goal_conditioned"]


# ---------------------------------------------------------------------------
# Seed discovery + gating
# ---------------------------------------------------------------------------

def discover_seeds(cond_dir: Path, prefix: str) -> list[int]:
    """Seed numbers under `cond_dir` matching `<prefix>seed<N>` that have a data/ subdirectory."""
    cond_dir = Path(cond_dir)
    seeds = []
    for p in sorted(cond_dir.glob(f"{prefix}seed*")):
        if not p.is_dir():
            continue
        try:
            s = int(p.name.replace(prefix + "seed", ""))
        except ValueError:
            continue
        if (p / "data").exists():
            seeds.append(s)
    return seeds


def clean_seeds_for_reward_vector(cond_dir: Path, prefix: str, seeds: list[int],
                                   periods: list[str], min_entropy: float = 5.5) -> list[int]:
    """Re-gate on the periods this analysis actually uses (goal-directed
    periods), rather than assuming the existing replay-triple-based gate
    (used elsewhere, e.g. Figure 5) applies unchanged here."""
    clean = []
    for s in seeds:
        data_dir = Path(cond_dir) / f"{prefix}seed{s}" / "data"
        if seed_is_clean(data_dir, periods, min_entropy):
            clean.append(s)
    return clean


# ---------------------------------------------------------------------------
# Panels a/b: per-seed population summary (Gauthier + Sarel)
# ---------------------------------------------------------------------------

def per_seed_population_summary(run_dir: Path, period: str, baseline_period: str = "before",
                                 n_null: int = 100, seed: int = 0) -> dict | None:
    """Runs analyze_period on `period` (e.g. goal_conditioned) and
    `baseline_period` (e.g. before, unbiased random walk), classifies cells
    in both, and returns per-seed summary fractions + means, plus the
    baseline-subtracted delta fractions (the headline, reward-specific
    numbers -- see reward_vector_analysis.classify_cells docstring for why
    raw fractions alone are misleading with a single static reward)."""
    res = analyze_period(run_dir, period, n_null=n_null, seed=seed)
    res_base = analyze_period(run_dir, baseline_period, n_null=n_null, seed=seed)
    if res is None or res_base is None:
        return None

    labels = classify_cells(res["ramp"], res["ramp_p"], res["mrl_ego"], res["mrl_ego_p"],
                             res["mrl_hd"], res["mrl_hd_p"], ring_mrl=res["ring_mrl"])
    labels_base = classify_cells(res_base["ramp"], res_base["ramp_p"], res_base["mrl_ego"],
                                  res_base["mrl_ego_p"], res_base["mrl_hd"], res_base["mrl_hd_p"],
                                  ring_mrl=res_base["ring_mrl"])

    out = {"n_period": res["n"], "n_baseline": res_base["n"]}
    for key in labels:
        frac = float(labels[key].mean())
        frac_base = float(labels_base[key].mean())
        out[f"frac_{key}"] = frac
        out[f"frac_{key}_baseline"] = frac_base
        out[f"delta_frac_{key}"] = frac - frac_base
    out["mean_ramp"] = float(res["ramp"].mean())
    out["mean_ramp_baseline"] = float(res_base["ramp"].mean())
    out["mean_mrl_ego"] = float(res["mrl_ego"].mean())
    out["mean_mrl_ego_baseline"] = float(res_base["mrl_ego"].mean())
    out["mean_mrl_hd"] = float(res["mrl_hd"].mean())
    out["mean_mrl_hd_baseline"] = float(res_base["mrl_hd"].mean())
    # keep raw arrays for pooled histograms (caller decides whether to keep)
    out["_ramp_arr"] = res["ramp"]
    out["_mrl_ego_arr"] = res["mrl_ego"]
    out["_mrl_hd_arr"] = res["mrl_hd"]
    out["_distance_cell_mask"] = labels["distance_cell"]
    out["_goal_direction_cell_mask"] = labels["goal_direction_cell"]
    return out


def load_condition_population_summary(cond_dir: Path, prefix: str, seeds: list[int],
                                       period: str = "goal_conditioned",
                                       baseline_period: str = "before") -> list[dict]:
    """Per-seed population summaries for a condition (no-head or head-ON)."""
    rows = []
    for s in seeds:
        run_dir = Path(cond_dir) / f"{prefix}seed{s}"
        if not run_dir.exists():
            continue
        r = per_seed_population_summary(run_dir, period, baseline_period, seed=s)
        if r is None:
            continue
        r["seed"] = s
        rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# Panels c/d: relocation rotation test
# ---------------------------------------------------------------------------

def per_seed_rotation_test(run_dir: Path, period_a: str, period_b: str,
                            n_null: int = 100, seed: int = 0) -> dict | None:
    """One relocation seed's rotation test for a given period pair (either
    the behavioral cycle_1_goal/cycle_2_goal pair, or the occupancy-
    controlled replay_reloc_postA/postB pair)."""
    res_a = analyze_period(run_dir, period_a, n_null=n_null, seed=seed)
    res_b = analyze_period(run_dir, period_b, n_null=n_null, seed=seed)
    if res_a is None or res_b is None:
        return None

    labels_a = classify_cells(res_a["ramp"], res_a["ramp_p"], res_a["mrl_ego"], res_a["mrl_ego_p"],
                               res_a["mrl_hd"], res_a["mrl_hd_p"], ring_mrl=res_a["ring_mrl"])
    labels_b = classify_cells(res_b["ramp"], res_b["ramp_p"], res_b["mrl_ego"], res_b["mrl_ego_p"],
                               res_b["mrl_hd"], res_b["mrl_hd_p"], ring_mrl=res_b["ring_mrl"])

    stable_goal = labels_a["goal_direction_cell"] & labels_b["goal_direction_cell"]
    stable_hd_only = (labels_a["head_direction_cell"] & labels_b["head_direction_cell"]
                      & ~labels_a["goal_direction_cell"] & ~labels_b["goal_direction_cell"])
    stable_goal_and_hd = stable_goal & labels_a["head_direction_cell"] & labels_b["head_direction_cell"]
    stable_goal_only = stable_goal & ~labels_a["head_direction_cell"] & ~labels_b["head_direction_cell"]

    dphi_goal = circular_diff(res_b["pref_ego"][stable_goal], res_a["pref_ego"][stable_goal])
    dphi_hd_only = circular_diff(res_b["pref_ego"][stable_hd_only], res_a["pref_ego"][stable_hd_only]) \
        if stable_hd_only.sum() > 0 else np.array([])

    delta_geom = geometric_rotation(run_dir, period_a=period_a, period_b=period_b)

    cm_goal, ci_goal = bootstrap_circular_mean_ci(dphi_goal) if stable_goal.sum() > 0 else (float("nan"), (float("nan"), float("nan")))

    return {
        "n_stable_goal": int(stable_goal.sum()),
        "n_stable_hd_only": int(stable_hd_only.sum()),
        "n_stable_goal_and_hd": int(stable_goal_and_hd.sum()),
        "n_stable_goal_only": int(stable_goal_only.sum()),
        "delta_geom": delta_geom,
        "circ_mean_dphi_goal": cm_goal,
        "circ_ci_dphi_goal": ci_goal,
        "dphi_goal_arr": dphi_goal,
        "dphi_hd_only_arr": dphi_hd_only,
    }


def load_reloc_rotation_summary(reloc_dir: Path, seeds: list[int],
                                 period_a: str, period_b: str) -> list[dict]:
    """Per-seed rotation test across relocation seeds, skipping runs that are missing or fail."""
    rows = []
    for s in seeds:
        run_dir = Path(reloc_dir) / f"reloc_seed{s}"
        if not run_dir.exists():
            continue
        r = per_seed_rotation_test(run_dir, period_a, period_b, seed=s)
        if r is None:
            continue
        r["seed"] = s
        rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def _jsonable(obj):
    """Recursively convert numpy types to JSON-serializable ones, dropping private `_`-prefixed keys."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def save_summary_cache(cond_dir: Path, rows: list[dict], name: str = "reward_vector_summary.json"):
    """Write the per-seed rows to `<cond_dir>/aggregate/<name>` as JSON and return the path."""
    out_dir = Path(cond_dir) / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / name
    with open(out_path, "w") as f:
        json.dump(_jsonable(rows), f, indent=2)
    print(f"  Saved: {out_path}")
    return out_path
