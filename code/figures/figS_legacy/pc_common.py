"""Shared helpers for the place-cell analysis scripts.

Used by the generators in this directory: plot_replay_v11.py,
plot_goal_directed_maps.py and analyze_activty_lineardecoder.py.
"""

import json
import os
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Canonical place-cell filter thresholds, so "place cell" means the same
# neuron set in every script. Skaggs bits/spike is scale-invariant and is the
# standard neuroscience criterion; k<=3 excludes diffuse/multi-modal cells.
#
# icl_score (a GMM model-selection confidence score: more negative = a
# tighter, more confident fit) is still used, but as a within-state relative
# quantile rather than a fixed absolute cutoff. An earlier version used a
# fixed absolute cutoff (icl_score <= -5000); that doesn't port across runs
# because icl_score's magnitude is driven by each neuron's total spike-mass,
# which scales with a run's overall activity level, not just spatial
# concentration (it passed ~23-33% of neurons for Negative_Results vs
# ~11-17% for Positive_Results at the same nominal threshold, for reasons
# unrelated to place-field quality). But dropping it entirely lets in a lot
# of genuinely noisy/scattered cells that pass Skaggs+k by chance -- visual
# inspection confirmed the top ~50% by icl_score (computed fresh within each
# state, after the Skaggs+k gate) reliably excludes those while keeping
# clean, spatially-organized fields (point-like place fields as well as
# border/edge cells).
PLACE_CELL_K_MAX = 3
PLACE_CELL_SKAGGS_MIN = 1.0
PLACE_CELL_ICL_QUANTILE = 0.5


def filter_place_cells(df_skaggs: pd.DataFrame, df_centers: pd.DataFrame, skaggs_prec: Optional[float] = None,
                        icl_quantile: float = PLACE_CELL_ICL_QUANTILE) -> pd.DataFrame:
    """
    Canonical "is this a place cell" filter, shared by the generators in this
    directory so they all define the same neuron set.

    df_skaggs: columns neuron_id, skaggs_score.
    df_centers: columns neuron_id, k, icl_score (one row per neuron; if
    exploded to one row per center, duplicates are harmless since neuron_id/
    k/icl_score are the same across a neuron's rows).

    Order of gates: k<=PLACE_CELL_K_MAX -> Skaggs>=PLACE_CELL_SKAGGS_MIN (or
    a within-state Skaggs quantile if skaggs_prec is given) -> icl_score at
    or below the icl_quantile of the *remaining* population's icl_score
    (computed fresh each call, so it's relative to this specific state).
    Returns a dataframe with a single column, neuron_id.
    """
    df_centers = df_centers[df_centers["k"] <= PLACE_CELL_K_MAX]
    df = df_skaggs.merge(df_centers, on="neuron_id", how="inner").drop_duplicates("neuron_id")
    if skaggs_prec:
        skaggs_quantile = df["skaggs_score"].quantile(skaggs_prec)
        df = df[df["skaggs_score"] >= skaggs_quantile]
    else:
        df = df[df["skaggs_score"] >= PLACE_CELL_SKAGGS_MIN]

    if icl_quantile is not None and len(df):
        icl = df["icl_score"].replace([np.inf, -np.inf], np.nan)
        threshold = icl.quantile(icl_quantile)
        if np.isfinite(threshold):
            df = df[df["icl_score"] <= threshold]

    return df[["neuron_id"]].drop_duplicates()


def compute_skaggs_scores_optimized(df_combined: pd.DataFrame, output: Optional[str] = None, m: int = 32) -> np.ndarray:
    """
    Vectorized Skaggs spatial information per spike (bits/spike):

        I = sum_i p_i * (lambda_i / lambda_bar) * log2(lambda_i / lambda_bar)

    where p_i is the empirical occupancy probability of bin i (not 1/N_bins),
    lambda_i is the mean activity in bin i, and lambda_bar is the
    occupancy-weighted mean activity = sum_i p_i * lambda_i.

    df_combined must have columns x, y, activity, neuron_id (behavior merged
    with activations on timestep). If output is given, also writes a
    neuron_id/skaggs_score dataframe there.
    """
    grid_x = np.floor(df_combined["x"] * m).astype(int)
    grid_y = np.floor(df_combined["y"] * m).astype(int)
    grid_cell = grid_x * m + grid_y

    df_with_grid = df_combined.copy()
    df_with_grid["grid_cell"] = grid_cell

    # Empirical occupancy p_i: fraction of timesteps the agent is in bin i.
    # Compute it from any single neuron (occupancy is shared across neurons).
    one_neuron_id = df_with_grid["neuron_id"].iloc[0]
    occ_counts = (
        df_with_grid[df_with_grid["neuron_id"] == one_neuron_id]
        .groupby("grid_cell")
        .size()
    )
    occ_p = (occ_counts / occ_counts.sum()).rename("p_occ").reset_index()

    # Mean activity per (neuron, bin)
    cell_activities = (
        df_with_grid.groupby(["neuron_id", "grid_cell"])["activity"]
        .agg(["mean", "count"])
        .reset_index()
    )
    cell_activities.columns = ["neuron_id", "grid_cell", "mean_activity_cell", "count"]

    # Drop bins with zero mean activity (they contribute 0 to Skaggs by convention).
    cell_activities = cell_activities[cell_activities["mean_activity_cell"] > 0]

    # Attach empirical occupancy
    cell_activities = cell_activities.merge(occ_p, on="grid_cell", how="left")

    # Occupancy-weighted mean activity per neuron (= lambda_bar).
    # NOTE: this must be computed over all bins (not the >0 filter) to match
    # the standard Skaggs definition. We use the full df for that.
    full_mean = (
        df_with_grid.groupby(["neuron_id", "grid_cell"])["activity"].mean().reset_index()
    )
    full_mean = full_mean.merge(occ_p, on="grid_cell", how="left")
    full_mean["weighted"] = full_mean["activity"] * full_mean["p_occ"]
    lambda_bar = (
        full_mean.groupby("neuron_id")["weighted"].sum().rename("lambda_bar").reset_index()
    )

    cell_activities = cell_activities.merge(lambda_bar, on="neuron_id", how="left")

    # bits/spike Skaggs
    cell_activities["log_ratio"] = np.log2(
        cell_activities["mean_activity_cell"] / cell_activities["lambda_bar"]
    )
    cell_activities["skaggs_contribution"] = (
        cell_activities["p_occ"]
        * (cell_activities["mean_activity_cell"] / cell_activities["lambda_bar"])
        * cell_activities["log_ratio"]
    )

    # Sum contributions for each neuron, reindex over the full neuron set so
    # that all-silent neurons get a Skaggs score of 0 (not NaN / dropped).
    all_neurons = pd.Index(sorted(df_with_grid["neuron_id"].unique()), name="neuron_id")
    skaggs_scores = (
        cell_activities.groupby("neuron_id")["skaggs_contribution"]
        .sum()
        .reindex(all_neurons, fill_value=0)
    )

    if output:
        scores_df = pd.DataFrame(
            {"neuron_id": skaggs_scores.index, "skaggs_score": skaggs_scores.values}
        )
        scores_df.to_pickle(output)
    return skaggs_scores.values


def state_paths(run_dir: str, state: str) -> Dict[str, str]:
    """Standard flat-layout paths for a v11 state, e.g. 'before', 'after',
    'replay_c1_pred', 'replay_c1_postrl', 'cycle_1'."""
    data_dir = os.path.join(run_dir, "data")
    return {
        "behavior": os.path.join(data_dir, f"df_behavior_{state}.pkl"),
        "activations": os.path.join(data_dir, f"df_activations_{state}.pkl"),
        "skaggs": os.path.join(data_dir, f"df_skaggs_{state}.pkl"),
        "centers": os.path.join(data_dir, f"df_centers_{state}.pkl"),
    }


def ensure_skaggs(run_dir: str, state: str) -> str:
    """Returns the path to df_skaggs_{state}.pkl, computing it first if missing."""
    paths = state_paths(run_dir, state)
    if not os.path.exists(paths["skaggs"]):
        df_beh = pd.read_pickle(paths["behavior"])
        df_act = pd.read_pickle(paths["activations"])
        df_combined = df_beh.merge(df_act, on="timestep", how="inner")
        print(f"Computing skaggs scores for '{state}'...")
        compute_skaggs_scores_optimized(df_combined, paths["skaggs"])
    return paths["skaggs"]


def load_run_config(run_dir: str) -> Dict[str, Any]:
    """Reads <run_dir>/config.json. reward_pos is coerced to a np.array."""
    path = os.path.join(run_dir, "config.json")
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "reward_pos" in cfg:
        cfg["reward_pos"] = np.asarray(cfg["reward_pos"], dtype=float)
    return cfg


def resolve_reward_pos(run_dir: Optional[str], explicit: Optional[Any]) -> np.ndarray:
    """explicit (if not None) wins; otherwise read reward_pos from run_dir/config.json."""
    if explicit is not None:
        return np.asarray(explicit, dtype=float)
    if run_dir is None:
        raise ValueError("Need either an explicit reward position or --run_dir to read config.json from.")
    return load_run_config(run_dir)["reward_pos"]


def save_all(path_no_ext: str, dpi: int = 150, fig=None) -> None:
    """Saves the current (or given) matplotlib figure as .png, .jpg, and .pdf."""
    os.makedirs(os.path.dirname(path_no_ext) or ".", exist_ok=True)
    target = fig if fig is not None else plt.gcf()
    for ext in ("png", "jpg", "pdf"):
        target.savefig(f"{path_no_ext}.{ext}", dpi=dpi, bbox_inches="tight", facecolor="white")


def append_stat(run_dir: str, script_name: str, record: Dict[str, Any]) -> None:
    """Appends a statistics record to <run_dir>/data/statistics_summary.json."""
    out_path = os.path.join(run_dir, "data", "statistics_summary.json")
    records = []
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            try:
                records = json.load(f)
            except json.JSONDecodeError:
                records = []
    record = dict(record)
    record["script"] = script_name

    def _clean(v):
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    record = {k: _clean(v) for k, v in record.items()}
    records.append(record)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


_NON_PVALUE_KEYS = {"p_obs", "p_chance"}


def _is_pvalue_key(key: str) -> bool:
    """Whether a stats-report key names a p-value."""
    if key in _NON_PVALUE_KEYS:
        return False
    lk = key.lower()
    return lk.startswith("p_") or lk.endswith("_p") or "_p_" in lk or "pvalue" in lk


def _significance_stars(p: float) -> str:
    """Conventional star string for a p-value ('' when not significant)."""
    if p < 0.001:
        return " ***"
    if p < 0.01:
        return " **"
    if p < 0.05:
        return " *"
    return ""


def write_stats_report(run_dir: str) -> Optional[str]:
    """Renders <run_dir>/data/statistics_summary.json as a human-readable
    <run_dir>/data/statistics_report.txt, grouped by script, with
    significance markers (* p<0.05, ** p<0.01, *** p<0.001) on p-value
    fields. Returns the report path, or None if there's nothing to report."""
    in_path = os.path.join(run_dir, "data", "statistics_summary.json")
    if not os.path.exists(in_path):
        return None
    with open(in_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not records:
        return None

    by_script: Dict[str, list] = {}
    for r in records:
        by_script.setdefault(r.get("script", "unknown"), []).append(r)

    lines = [
        "=" * 72,
        f"Statistics report for: {run_dir}",
        f"Generated from: {in_path}",
        "(significance markers: * p<0.05, ** p<0.01, *** p<0.001)",
        "=" * 72,
        "",
    ]
    for script in sorted(by_script):
        recs = by_script[script]
        lines.append(script)
        lines.append("-" * len(script))
        for i, r in enumerate(recs, 1):
            header = f"[{i}] {r.get('test', '(untitled test)')}"
            tags = [f"{k}={r[k]}" for k in ("state_pair", "period") if k in r]
            if tags:
                header += "  (" + ", ".join(tags) + ")"
            lines.append(header)
            for k, v in r.items():
                if k in ("script", "test", "state_pair", "period"):
                    continue
                if isinstance(v, float):
                    val_str = f"{v:.6g}"
                    if _is_pvalue_key(k):
                        val_str += _significance_stars(v)
                else:
                    val_str = str(v)
                lines.append(f"    {k}: {val_str}")
            lines.append("")
        lines.append("")

    out_path = os.path.join(run_dir, "data", "statistics_report.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Regenerate statistics_report.txt from statistics_summary.json.")
    p.add_argument("--run_dir", type=str, required=True)
    args = p.parse_args()
    path = write_stats_report(args.run_dir)
    print(f"Wrote {path}" if path else "Nothing to report (no statistics_summary.json).")
