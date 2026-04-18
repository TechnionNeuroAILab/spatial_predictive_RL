from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import shapiro, ttest_1samp, wilcoxon

from common import ensure_dir, save_json

SEEDS = [42, 123, 7, 2024, 999]
BASE_DIR = "../New_Results/multi_seed"
SCRIPT_PATH = "Code/ratinabox_v9.py"
N_TRIALS = 100
TIMESTEPS = 10
PHASE1_EPOCHS = 10
MAX_EPISODES = 50
MAX_STEPS = 20
CHKPT_EVERY = 10


@dataclass
class SeedRunStatus:
    seed: int
    command: str
    return_code: int | None
    success: bool
    metrics_found: bool
    curves_found: bool
    analyses_successful: bool


def flatten_numeric_metrics(payload: dict[str, Any], prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(flatten_numeric_metrics(value, prefix=name))
        elif isinstance(value, (int, float, np.number)):
            out[name] = float(value)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and aggregate multi-seed robustness experiments.")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--base_dir", type=str, default=BASE_DIR)
    parser.add_argument("--script_path", type=str, default=SCRIPT_PATH)
    parser.add_argument("--python_exec", type=str, default="python")
    parser.add_argument("--n_trials", type=int, default=N_TRIALS)
    parser.add_argument("--timesteps", type=int, default=TIMESTEPS)
    parser.add_argument("--phase1_epochs", type=int, default=PHASE1_EPOCHS)
    parser.add_argument("--max_episodes", type=int, default=MAX_EPISODES)
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS)
    parser.add_argument("--checkpoint_every", type=int, default=CHKPT_EVERY)
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--skip_aggregation", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument(
        "--analysis_scripts",
        type=str,
        nargs="*",
        default=[],
        help="Optional scripts to run per seed after training (e.g., New_Code/ramping_analysis.py).",
    )

    return parser.parse_args()


def run_command_logged(command: list[str], log_path: Path, append: bool = False) -> int:
    ensure_dir(log_path.parent)
    mode = "a" if append else "w"
    with log_path.open(mode, encoding="utf-8") as log_f:
        log_f.write("COMMAND: " + " ".join(command) + "\n")
        log_f.flush()
        result = subprocess.run(command, stdout=log_f, stderr=log_f, check=False)
        log_f.write(f"\nRETURN_CODE: {result.returncode}\n")
    return int(result.returncode)


def build_training_command(args: argparse.Namespace, seed: int) -> list[str]:
    return [
        args.python_exec,
        args.script_path,
        "--seed",
        str(seed),
        "--output_dir",
        args.base_dir,
        "--run_name",
        f"seed_{seed}",
        "--no_show",
        "--n_trials",
        str(args.n_trials),
        "--timesteps",
        str(args.timesteps),
        "--phase1_epochs",
        str(args.phase1_epochs),
        "--max_episodes",
        str(args.max_episodes),
        "--max_steps",
        str(args.max_steps),
        "--checkpoint_every",
        str(args.checkpoint_every),
    ]


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_seed_metrics(seed: int, run_dir: Path) -> dict[str, float]:
    payload = read_json(run_dir / "metrics.json")

    test_rates = payload.get("test_success_rates", [np.nan, np.nan])
    if not isinstance(test_rates, list):
        test_rates = [np.nan, np.nan]

    row = {
        "seed": float(seed),
        "phase1_final_loss": safe_float(payload.get("phase1_final_loss")),
        "phase2_mean_reward_last10": safe_float(
            payload.get("phase2_mean_reward_last10", payload.get("phase2_mean_reward"))
        ),
        "phase2_mean_success_last10": safe_float(
            payload.get("phase2_mean_success_last10", payload.get("phase2_mean_success"))
        ),
        "test_success_center": safe_float(test_rates[0] if len(test_rates) > 0 else np.nan),
        "test_success_generalization": safe_float(test_rates[1] if len(test_rates) > 1 else np.nan),
    }
    return row


def collect_analysis_metrics(seed_dir: Path) -> dict[str, float]:
    collected: dict[str, float] = {}
    skip_names = {
        "metrics.json",
        "config.json",
        "aggregate_metrics.json",
        "aggregate_summary.json",
    }

    for path in seed_dir.rglob("*.json"):
        if path.name in skip_names:
            continue
        try:
            payload = read_json(path)
            if not isinstance(payload, dict):
                continue

            rel = path.relative_to(seed_dir).with_suffix("")
            prefix = "analysis." + str(rel).replace("\\", ".").replace("/", ".")
            collected.update(flatten_numeric_metrics(payload, prefix=prefix))
        except Exception:
            continue

    return collected


def one_sample_stats(values: Iterable[float]) -> dict[str, Any]:
    vals = np.asarray(list(values), dtype=float)
    vals = vals[np.isfinite(vals)]

    out: dict[str, Any] = {
        "n": int(vals.size),
        "mean": np.nan,
        "std": np.nan,
        "ci_95_low": np.nan,
        "ci_95_high": np.nan,
        "ttest_statistic": np.nan,
        "ttest_p_value": np.nan,
        "shapiro_statistic": np.nan,
        "shapiro_p_value": np.nan,
        "wilcoxon_statistic": np.nan,
        "wilcoxon_p_value": np.nan,
        "cohens_d": np.nan,
        "primary_test": "ttest_1samp",
    }

    if vals.size == 0:
        return out

    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan
    sem = float(std / np.sqrt(vals.size)) if vals.size > 1 and np.isfinite(std) else np.nan

    out["mean"] = mean
    out["std"] = std
    if np.isfinite(sem):
        out["ci_95_low"] = float(mean - 1.96 * sem)
        out["ci_95_high"] = float(mean + 1.96 * sem)

    if vals.size >= 2:
        t_res = ttest_1samp(vals, popmean=0.0, nan_policy="omit")
        out["ttest_statistic"] = float(t_res.statistic)
        out["ttest_p_value"] = float(t_res.pvalue)

    if vals.size >= 3:
        sh = shapiro(vals)
        out["shapiro_statistic"] = float(sh.statistic)
        out["shapiro_p_value"] = float(sh.pvalue)
        if sh.pvalue < 0.05:
            try:
                w = wilcoxon(vals, alternative="two-sided", zero_method="wilcox")
                out["wilcoxon_statistic"] = float(w.statistic)
                out["wilcoxon_p_value"] = float(w.pvalue)
                out["primary_test"] = "wilcoxon_signed_rank"
            except ValueError:
                pass

    if np.isfinite(std) and not np.isclose(std, 0.0):
        out["cohens_d"] = float(mean / std)

    return out


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    window = max(1, min(int(window), values.size))
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="valid")


def truncate_to_shortest(arrays: list[np.ndarray]) -> np.ndarray:
    min_len = min(arr.shape[0] for arr in arrays)
    return np.stack([arr[:min_len] for arr in arrays], axis=0)


def find_convergence_episode(success_curve: np.ndarray, window: int = 10) -> int | None:
    success_curve = np.asarray(success_curve, dtype=float)
    if success_curve.size < window:
        return None

    rolling = rolling_mean(success_curve, window=window)
    for i in range(rolling.size):
        if rolling[i] > 0.7 and np.all(rolling[i:] > 0.5):
            return int(i + window - 1)
    return None


def plot_phase1_multiseed(phase1_curves: list[np.ndarray], out_path: Path) -> dict[str, float]:
    stacked = truncate_to_shortest(phase1_curves)
    mean = np.mean(stacked, axis=0)
    std = np.std(stacked, axis=0, ddof=1) if stacked.shape[0] > 1 else np.zeros_like(mean)
    x = np.arange(stacked.shape[1])

    plt.figure(figsize=(10, 5))
    for row in stacked:
        plt.plot(x, row, color="gray", alpha=0.2, linewidth=0.8)

    plt.plot(x, mean, color="#2E5FA3", linewidth=2.0, label="Mean")
    plt.fill_between(x, mean - std, mean + std, color="#2E5FA3", alpha=0.25, label="±1 SD")

    plt.xlabel("Training Epoch")
    plt.ylabel("Prediction MSE")
    plt.ylim(bottom=0.0)
    plt.xlim(left=0)
    plt.title("Phase 1 Prediction Loss Across Seeds")
    plt.legend(loc="best")

    final_txt = f"Final: {mean[-1]:.4f} ± {std[-1]:.4f}"
    plt.text(0.98, 0.95, final_txt, ha="right", va="top", transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    return {
        "final_mean": float(mean[-1]),
        "final_std": float(std[-1]),
        "n_epochs": int(stacked.shape[1]),
    }


def plot_phase2_multiseed(reward_curves: list[np.ndarray],success_curves: list[np.ndarray],convergence_episodes: list[int],out_path: Path,
) -> dict[str, float]:
    reward_smooth = [rolling_mean(c, window=5) for c in reward_curves]
    success_smooth = [rolling_mean(c, window=5) for c in success_curves]

    reward_stack = truncate_to_shortest(reward_smooth)
    success_stack = truncate_to_shortest(success_smooth)

    reward_mean = np.mean(reward_stack, axis=0)
    reward_std = np.std(reward_stack, axis=0, ddof=1) if reward_stack.shape[0] > 1 else np.zeros_like(reward_mean)

    success_mean = np.mean(success_stack, axis=0)
    success_std = np.std(success_stack, axis=0, ddof=1) if success_stack.shape[0] > 1 else np.zeros_like(success_mean)

    x = np.arange(reward_stack.shape[1]) + 4  # rolling window=5 starts at episode 4

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Reward subplot
    ax = axes[0]
    for row in reward_stack:
        ax.plot(x, row, color="gray", alpha=0.15, linewidth=0.8)
    ax.plot(x, reward_mean, color="#2E5FA3", linewidth=2.0, label="Mean")
    ax.fill_between(x, reward_mean - reward_std, reward_mean + reward_std, color="#2E5FA3", alpha=0.25, label="±1 SD")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Phase 2 Reward Learning Curves Across Seeds")
    ax.text(
        0.98,
        0.92,
        f"Final: {reward_mean[-1]:.3f} ± {reward_std[-1]:.3f}",
        ha="right",
        va="top",
        transform=ax.transAxes,
    )
    ax.legend(loc="best")

    # Success subplot
    ax = axes[1]
    for row in success_stack:
        ax.plot(x, row, color="gray", alpha=0.15, linewidth=0.8)
    ax.plot(x, success_mean, color="#1B8A5A", linewidth=2.0, label="Mean")
    ax.fill_between(x, success_mean - success_std, success_mean + success_std, color="#1B8A5A", alpha=0.25, label="±1 SD")
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1.2, label="Chance")
    ax.axhline(0.7, color="darkorange", linestyle="--", linewidth=1.2, label="Convergence threshold")

    mean_conv = float(np.mean(convergence_episodes)) if len(convergence_episodes) > 0 else np.nan
    if np.isfinite(mean_conv):
        ax.axvline(mean_conv, color="gray", linestyle="--", linewidth=1.2)
        ax.text(
            0.02,
            0.12,
            f"Convergence: ep {mean_conv:.1f}",
            ha="left",
            va="bottom",
            transform=ax.transAxes,
        )

    ax.set_ylabel("Success Rate")
    ax.set_xlabel("Episode")
    ax.set_title("Phase 2 Success Curves Across Seeds")
    ax.text(
        0.98,
        0.92,
        f"Final: {success_mean[-1]:.3f} ± {success_std[-1]:.3f}",
        ha="right",
        va="top",
        transform=ax.transAxes,
    )
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

    return {
        "final_reward_mean": float(reward_mean[-1]),
        "final_reward_std": float(reward_std[-1]),
        "final_success_mean": float(success_mean[-1]),
        "final_success_std": float(success_std[-1]),
        "mean_convergence_episode": mean_conv,
    }


def try_load_test_distance_success(seed_dir: Path) -> pd.DataFrame:
    candidates = [
        seed_dir / "test_trials.json",
        seed_dir / "data" / "test_trials.json",
        seed_dir / "test_distance_success.json",
        seed_dir / "data" / "test_distance_success.json",
    ]
    for path in candidates:
        if not path.exists():
            continue

        payload = read_json(path)
        rows: list[dict[str, float]] = []

        if isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    continue
                if "start_distance" in item and "success" in item:
                    rows.append(
                        {
                            "start_distance": safe_float(item.get("start_distance")),
                            "success": safe_float(item.get("success")),
                        }
                    )
        elif isinstance(payload, dict):
            d = payload.get("start_distance", [])
            s = payload.get("success", [])
            if isinstance(d, list) and isinstance(s, list) and len(d) == len(s):
                for di, si in zip(d, s):
                    rows.append({"start_distance": safe_float(di), "success": safe_float(si)})

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            if not df.empty:
                return df

    return pd.DataFrame(columns=["start_distance", "success"])


def plot_success_by_distance(all_trials: pd.DataFrame, out_path: Path) -> dict[str, Any]:
    if all_trials.empty:
        return {"available": False, "note": "No per-trial distance test file found."}

    df = all_trials.copy()
    if df["start_distance"].nunique() < 4:
        return {"available": False, "note": "Not enough unique start distances for quartile binning."}

    df["distance_bin"] = pd.qcut(df["start_distance"], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    grouped = (
        df.groupby("distance_bin")["success"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "success_rate", "std": "success_sd"})
    )
    grouped["success_sem"] = grouped["success_sd"] / np.sqrt(grouped["count"].clip(lower=1))

    plt.figure(figsize=(7, 4.5))
    x = np.arange(grouped.shape[0])
    y = grouped["success_rate"].to_numpy(dtype=float)
    err = grouped["success_sem"].fillna(0.0).to_numpy(dtype=float)
    labels = grouped["distance_bin"].astype(str).tolist()

    plt.bar(x, y, yerr=err, capsize=4, alpha=0.85, color="#2E5FA3")
    plt.xticks(x, labels)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Initial distance-to-goal quartile")
    plt.ylabel("Success rate")
    plt.title("Success rate by initial distance to goal")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    return {
        "available": True,
        "bins": grouped.to_dict(orient="records"),
    }


def run_analysis_script_for_seed(
    python_exec: str,
    script_path: str,
    seed_dir: Path,
    log_path: Path,
) -> bool:
    candidate_data_dirs = [seed_dir]
    if (seed_dir / "data").exists():
        candidate_data_dirs.append(seed_dir / "data")

    for data_path in candidate_data_dirs:
        command_underscore = [python_exec, script_path, "--data_dir", str(data_path)]
        rc = run_command_logged(command_underscore, log_path, append=True)
        if rc == 0:
            return True

        command_hyphen = [python_exec, script_path, "--data-dir", str(data_path)]
        rc2 = run_command_logged(command_hyphen, log_path, append=True)
        if rc2 == 0:
            return True

    return False


def main() -> None:
    args = parse_args()

    runs_root = ensure_dir(args.base_dir)
    aggregate_dir = ensure_dir(runs_root / "aggregate")
    agg_figures_dir = ensure_dir(aggregate_dir / "figures")
    for seed in args.seeds:
        seed_root = ensure_dir(runs_root / f"seed_{seed}")
        ensure_dir(seed_root / "figures")
        ensure_dir(seed_root / "checkpoints")
        ensure_dir(seed_root / "data")

    statuses: list[SeedRunStatus] = []
    metrics_rows: list[dict[str, float]] = []
    phase1_curves: list[np.ndarray] = []
    reward_curves: list[np.ndarray] = []
    success_curves: list[np.ndarray] = []
    convergence_episodes: list[int] = []
    all_test_distance_trials: list[pd.DataFrame] = []
    if not args.skip_training:
        for seed in args.seeds:
            seed_dir = runs_root / f"seed_{seed}"
            log_path = seed_dir / "run.log"

            command = build_training_command(args, seed)
            return_code = run_command_logged(command, log_path, append=False)

            if return_code != 0:
                print(f"WARNING: seed {seed} failed with return code {return_code}")
                if not args.continue_on_error:
                    raise RuntimeError(f"Seed {seed} failed. Re-run with --continue_on_error to continue.")

            analysis_ok = True
            for script in args.analysis_scripts:
                analysis_log = seed_dir / f"analysis_{Path(script).stem}.log"
                this_ok = run_analysis_script_for_seed(
                    python_exec=args.python_exec,
                    script_path=script,
                    seed_dir=seed_dir,
                    log_path=analysis_log,
                )
                analysis_ok = analysis_ok and this_ok
                if not this_ok:
                    print(f"WARNING: analysis script failed for seed {seed}: {script}")

            statuses.append(
                SeedRunStatus(
                    seed=seed,
                    command=" ".join(command),
                    return_code=return_code,
                    success=(return_code == 0),
                    metrics_found=(seed_dir / "metrics.json").exists(),
                    curves_found=(seed_dir / "learning_curves.npz").exists(),
                    analyses_successful=analysis_ok,
                )
            )
    else:
        # Aggregation-only mode still emits status rows.
        for seed in args.seeds:
            seed_dir = runs_root / f"seed_{seed}"
            statuses.append(
                SeedRunStatus(
                    seed=seed,
                    command="",
                    return_code=None,
                    success=True,
                    metrics_found=(seed_dir / "metrics.json").exists(),
                    curves_found=(seed_dir / "learning_curves.npz").exists(),
                    analyses_successful=True,
                )
            )
    # Save status table
    status_df = pd.DataFrame([s.__dict__ for s in statuses])
    status_df.to_csv(aggregate_dir / "seed_status.csv", index=False)

    if args.skip_aggregation:
        print(f"Training complete. Aggregation skipped. Status CSV: {aggregate_dir / 'seed_status.csv'}")
        return

    for seed in args.seeds:
        seed_dir = runs_root / f"seed_{seed}"

        row = extract_seed_metrics(seed, seed_dir)
        row.update(collect_analysis_metrics(seed_dir))

        curve_path = seed_dir / "learning_curves.npz"
        if curve_path.exists():
            curves = np.load(curve_path)

            if "phase1_loss" in curves.files:
                phase1_loss = np.asarray(curves["phase1_loss"], dtype=float).reshape(-1)
                if phase1_loss.size > 0:
                    phase1_curves.append(phase1_loss)

            if "rl_reward" in curves.files:
                rl_reward = np.asarray(curves["rl_reward"], dtype=float).reshape(-1)
                if rl_reward.size > 0:
                    reward_curves.append(rl_reward)

            if "rl_success" in curves.files:
                rl_success = np.asarray(curves["rl_success"], dtype=float).reshape(-1)
                if rl_success.size > 0:
                    success_curves.append(rl_success)
                    conv = find_convergence_episode(rl_success, window=10)
                    if conv is not None:
                        convergence_episodes.append(conv)
                        row["convergence_episode"] = float(conv)
                    else:
                        row["convergence_episode"] = np.nan
                else:
                    row["convergence_episode"] = np.nan
            else:
                row["convergence_episode"] = np.nan
        else:
            row["convergence_episode"] = np.nan

        metrics_rows.append(row)
        per_seed_trials = try_load_test_distance_success(seed_dir)
        if not per_seed_trials.empty:
            per_seed_trials = per_seed_trials.copy()
            per_seed_trials["seed"] = seed
            all_test_distance_trials.append(per_seed_trials)

    metrics_df = pd.DataFrame(metrics_rows).sort_values("seed").reset_index(drop=True)
    metrics_df.to_csv(aggregate_dir / "seed_comparison_table.csv", index=False)

    metric_columns = [
        "phase1_final_loss",
        "phase2_mean_reward_last10",
        "phase2_mean_success_last10",
        "test_success_center",
        "test_success_generalization",
        "convergence_episode",
    ]

    metric_stats: dict[str, Any] = {}
    for metric in metric_columns:
        if metric not in metrics_df.columns:
            continue
        metric_stats[metric] = one_sample_stats(metrics_df[metric].to_numpy(dtype=float))

    analysis_metric_stats: dict[str, Any] = {}
    for col in metrics_df.columns:
        if col == "seed" or col in metric_columns:
            continue
        vals = pd.to_numeric(metrics_df[col], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        analysis_metric_stats[col] = one_sample_stats(vals)

    # Phase-1 figure
    phase1_summary: dict[str, float] = {}
    if len(phase1_curves) >= 1:
        phase1_summary = plot_phase1_multiseed(
            phase1_curves,
            agg_figures_dir / "phase1_loss_multi_seed.png",
        )
        plot_phase1_multiseed(
            phase1_curves,
            agg_figures_dir / "phase1_prediction_loss_multiseed.png",
        )

    # Phase-2 figure
    phase2_summary: dict[str, float] = {}
    if len(reward_curves) >= 1 and len(success_curves) >= 1:
        phase2_summary = plot_phase2_multiseed(
            reward_curves,
            success_curves,
            convergence_episodes,
            agg_figures_dir / "phase2_learning_curves_multi_seed.png",
        )
        plot_phase2_multiseed(
            reward_curves,
            success_curves,
            convergence_episodes,
            agg_figures_dir / "phase2_learning_curves_multiseed.png",
        )

    if len(all_test_distance_trials) > 0:
        distance_trials_df = pd.concat(all_test_distance_trials, ignore_index=True)
    else:
        distance_trials_df = pd.DataFrame(columns=["start_distance", "success", "seed"])
    distance_success_summary = plot_success_by_distance(
        distance_trials_df,
        agg_figures_dir / "success_by_start_distance.png",
    )
    final_success_vals = pd.to_numeric(metrics_df["phase2_mean_success_last10"], errors="coerce").dropna().to_numpy(dtype=float)
    t_success = ttest_1samp(final_success_vals, popmean=0.0, nan_policy="omit") if final_success_vals.size >= 2 else None

    paper_report = {
        "final_mean_reward": metric_stats.get("phase2_mean_reward_last10", {}).get("mean", np.nan),
        "final_mean_success": metric_stats.get("phase2_mean_success_last10", {}).get("mean", np.nan),
        "convergence_episode": metric_stats.get("convergence_episode", {}).get("mean", np.nan),
        "test_success_center": metric_stats.get("test_success_center", {}).get("mean", np.nan),
        "test_success_generalization": metric_stats.get("test_success_generalization", {}).get("mean", np.nan),
        "phase2_ttest_vs_zero": {
            "t_statistic": float(t_success.statistic) if t_success is not None else np.nan,
            "p_value": float(t_success.pvalue) if t_success is not None else np.nan,
            "n": int(final_success_vals.size),
        },
    }

    summary = {
        "config": {
            "seeds": args.seeds,
            "base_dir": args.base_dir,
            "script_path": args.script_path,
            "n_trials": args.n_trials,
            "timesteps": args.timesteps,
            "phase1_epochs": args.phase1_epochs,
            "max_episodes": args.max_episodes,
            "max_steps": args.max_steps,
            "checkpoint_every": args.checkpoint_every,
            "analysis_scripts": args.analysis_scripts,
        },
        "n_requested_seeds": int(len(args.seeds)),
        "n_rows_in_seed_table": int(metrics_df.shape[0]),
        "seed_status_csv": str((aggregate_dir / "seed_status.csv").as_posix()),
        "seed_comparison_table_csv": str((aggregate_dir / "seed_comparison_table.csv").as_posix()),
        "metrics": metric_stats,
        "analysis_metrics": analysis_metric_stats,
        "phase1_curve_summary": phase1_summary,
        "phase2_curve_summary": phase2_summary,
        "success_by_start_distance": distance_success_summary,
        "paper_report": paper_report,
    }
    save_json(aggregate_dir / "aggregate_metrics.json", summary)

    print("Done. Multi-seed aggregation written to:")
    print(aggregate_dir)


if __name__ == "__main__":
    main()
