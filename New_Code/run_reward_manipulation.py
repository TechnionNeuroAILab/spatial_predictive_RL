from __future__ import annotations
import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from reward_manipulation import (
    compute_ori_single_goal,
    ensure_dir,
    load_baseline_centers,
    load_goal_position,
    run_all_section_d,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Section D reward-manipulation branches and aggregation.")
    parser.add_argument("--data-dir", "--data_dir", dest="data_dir", type=str, default="New_Code/data")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=str, default="../New_Results/reward_manipulation")
    parser.add_argument("--phase2-checkpoint", type=str, default="Results/ratinabox_v9/checkpoints/phase2_final.pt")
    parser.add_argument("--phase1-checkpoint", type=str, default="Results/ratinabox_v9/checkpoints/phase1_final.pt")
    parser.add_argument("--goal-file", type=str, default="goal_position.npy")
    parser.add_argument("--reward-x", type=float, default=0.5)
    parser.add_argument("--reward-y", type=float, default=0.5)
    parser.add_argument("--goal-area-fraction", type=float, default=0.20)
    parser.add_argument("--old-goal-x", type=float, default=0.75)
    parser.add_argument("--old-goal-y", type=float, default=0.75)
    parser.add_argument("--new-goal-x", type=float, default=0.25)
    parser.add_argument("--new-goal-y", type=float, default=0.25)
    parser.add_argument("--goal1-x", type=float, default=0.25)
    parser.add_argument("--goal1-y", type=float, default=0.75)
    parser.add_argument("--goal2-x", type=float, default=0.75)
    parser.add_argument("--goal2-y", type=float, default=0.25)
    parser.add_argument("--reward-removal-dir", type=str, default="reward_removal")
    parser.add_argument("--obs-corruption-dir", type=str, default="obs_corruption")
    parser.add_argument("--reward-relocation-dir", type=str, default="reward_relocation")
    parser.add_argument("--multi-reward-dir", type=str, default="multi_reward")
    parser.add_argument("--obs-noise-level", type=float, default=0.5)
    parser.add_argument("--run-d1-cmd", type=str, default="", help="Optional shell command to generate D1/D4a branch outputs.")
    parser.add_argument("--run-d4b-cmd", type=str, default="", help="Optional shell command to generate D4b branch outputs.")
    parser.add_argument("--run-d3-cmd", type=str, default="", help="Optional shell command to generate D3 branch outputs.")
    parser.add_argument("--run-d2-cmd", type=str, default="", help="Optional shell command to generate D2 branch outputs.")
    parser.add_argument("--cwd", type=str, default=".")
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_optional_command(label: str, command: str, cwd: Path, log_dir: Path) -> dict[str, Any]:
    if command.strip() == "":
        return {"label": label, "ran": False, "return_code": None, "log": "", "command": ""}

    ensure_dir(log_dir)
    log_path = log_dir / f"{label}.log"

    with log_path.open("w", encoding="utf-8") as f:
        f.write("COMMAND\n")
        f.write(command + "\n\n")
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            shell=True,
            stdout=f,
            stderr=f,
            check=False,
        )
        f.write(f"\nRETURN_CODE={proc.returncode}\n")

    return {
        "label": label,
        "ran": True,
        "return_code": int(proc.returncode),
        "log": str(log_path),
        "command": command,
    }


def resolve_branch_path(base_out: Path, branch: str) -> Path:
    p = Path(branch)
    if p.is_absolute():
        return p
    return base_out / p


def print_phase2_sanity(data_dir: Path, phase2_checkpoint: Path, goal: np.ndarray, goal_radius: float) -> dict[str, Any]:
    c1, c2, src1, src2 = load_baseline_centers(data_dir)
    ori1 = compute_ori_single_goal(c1, goal, goal_radius)["ori"]
    ori2 = compute_ori_single_goal(c2, goal, goal_radius)["ori"]

    run_dir = phase2_checkpoint.parent.parent if phase2_checkpoint.parent.name == "checkpoints" else phase2_checkpoint.parent
    metrics_path = run_dir / "metrics.json"

    mean_success = np.nan
    if metrics_path.exists():
        try:
            with metrics_path.open("r", encoding="utf-8") as f:
                m = json.load(f)
            if "phase2_mean_success_last10" in m and m["phase2_mean_success_last10"] is not None:
                mean_success = float(m["phase2_mean_success_last10"])
            elif "phase2_mean_success" in m and m["phase2_mean_success"] is not None:
                mean_success = float(m["phase2_mean_success"])
        except Exception:
            mean_success = np.nan

    sanity = {
        "phase2_checkpoint_exists": phase2_checkpoint.exists(),
        "phase2_checkpoint": str(phase2_checkpoint),
        "phase1_centers_source": src1,
        "phase2_centers_source": src2,
        "phase1_ori": float(ori1) if np.isfinite(ori1) else np.nan,
        "phase2_ori": float(ori2) if np.isfinite(ori2) else np.nan,
        "phase2_mean_success": float(mean_success) if np.isfinite(mean_success) else np.nan,
        "metrics_path": str(metrics_path),
    }
    return sanity


def main() -> None:
    args = parse_args()

    cwd = Path(args.cwd).resolve()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (cwd / data_dir).resolve()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = (cwd / out_dir).resolve()
    ensure_dir(out_dir)

    log_dir = ensure_dir(out_dir / "logs")

    phase2_ckpt = Path(args.phase2_checkpoint)
    if not phase2_ckpt.is_absolute():
        phase2_ckpt = (cwd / phase2_ckpt).resolve()

    goal_default = np.array([args.reward_x, args.reward_y], dtype=float)
    goal, goal_source = load_goal_position(data_dir, args.goal_file, goal_default)
    goal_radius = float(np.sqrt(args.goal_area_fraction / np.pi))

    sanity = print_phase2_sanity(data_dir, phase2_ckpt, goal, goal_radius)
    command_runs = []
    command_runs.append(run_optional_command("D1_reward_removal", args.run_d1_cmd, cwd, log_dir))
    command_runs.append(run_optional_command("D4b_obs_corruption", args.run_d4b_cmd, cwd, log_dir))
    command_runs.append(run_optional_command("D3_reward_relocation", args.run_d3_cmd, cwd, log_dir))
    command_runs.append(run_optional_command("D2_multi_reward", args.run_d2_cmd, cwd, log_dir))
    analysis_args = argparse.Namespace(
        data_dir=str(data_dir),
        output_dir=str(out_dir),
        goal_file=args.goal_file,
        reward_x=float(args.reward_x),
        reward_y=float(args.reward_y),
        goal_area_fraction=float(args.goal_area_fraction),
        old_goal_x=float(args.old_goal_x),
        old_goal_y=float(args.old_goal_y),
        new_goal_x=float(args.new_goal_x),
        new_goal_y=float(args.new_goal_y),
        goal1_x=float(args.goal1_x),
        goal1_y=float(args.goal1_y),
        goal2_x=float(args.goal2_x),
        goal2_y=float(args.goal2_y),
        reward_removal_dir=args.reward_removal_dir,
        multi_reward_dir=args.multi_reward_dir,
        reward_relocation_dir=args.reward_relocation_dir,
        obs_corruption_dir=args.obs_corruption_dir,
        obs_noise_level=float(args.obs_noise_level),
        n_permutations=int(args.n_permutations),
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed),
    )

    manifest = run_all_section_d(analysis_args)

    run_summary = {
        "sanity": sanity,
        "goal_position": goal.tolist(),
        "goal_position_source": goal_source,
        "goal_radius": goal_radius,
        "commands": command_runs,
        "analysis_manifest": manifest,
    }

    save_path = out_dir / "run_reward_manipulation_manifest.json"
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2, sort_keys=True)

    print("Saved run manifest:")
    print(str(save_path))


if __name__ == "__main__":
    main()
