#!/usr/bin/env python3
"""
Panel-h style grids (cycle_1 Prediction vs RL) across learning rates, no reward head.

Example:
  python "code_analysis/figure 1/plot_panel_h_across_lrs.py" \\
      --scan-root New_Results_v3/dose_sweep_lr_nohead \\
      --output figures/panels/panel_h_nohead_across_lrs \\
      --n-neurons 10 --no_show
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from rate_maps import (  # noqa: E402
    build_rate_maps,
    compute_skaggs,
    load_cycle_rate_maps,
    smooth_and_norm,
)
from utils import (  # noqa: E402
    HIDDEN_DIM,
    M,
    RATEMAP_CMAP,
    apply_nature_style,
    ensure_dir,
    save_figure,
)

apply_nature_style()


def discover_runs(scan_root: Path, explicit: list[Path] | None) -> list[tuple[float, Path]]:
    """Return (lr, run_dir) sorted by lr for no-head runs that have cycle_1 probes."""
    candidates: list[Path] = []
    if explicit:
        candidates = explicit
    else:
        candidates = [p.parent for p in sorted(scan_root.glob("*/config.json"))]
        if not candidates and (scan_root / "config.json").exists():
            candidates = [scan_root]

    out: list[tuple[float, Path]] = []
    for run in candidates:
        cfg_path = run / "config.json"
        data = run / "data"
        if not cfg_path.exists():
            continue
        cfg = json.load(open(cfg_path))
        if cfg.get("reward_pred_head"):
            continue
        need = [
            data / "df_activations_cycle_1.pkl",
            data / "df_activations_cycle_1_goal.pkl",
        ]
        if not all(p.exists() for p in need):
            print(f"  skip {run}: missing cycle_1 probes")
            continue
        lr = float(cfg.get("rnn_rl_learning_rate"))
        out.append((lr, run))
    out.sort(key=lambda t: t[0])
    return out


def reward_bin_from_cfg(cfg: dict, fallback=(0.3, 0.3)) -> tuple[int, int]:
    pos = cfg.get("reward_pos", list(fallback))
    return int(float(pos[0]) * M), int(float(pos[1]) * M)


def top_neuron_ids(goal_cycles: dict, n: int) -> np.ndarray:
    key = next(k for k in goal_cycles if "cycle_1" in k)
    sk = goal_cycles[key][2]
    return np.argsort(sk)[::-1][:n]


def plot_lr_block(axes_row0, axes_row1, rw_cycles, goal_cycles, neuron_ids, reward_bin):
    rw_key = next(k for k in rw_cycles if "cycle_1" in k and "goal" not in k)
    goal_key = next(k for k in goal_cycles if "cycle_1" in k)
    rx, ry = reward_bin
    for col, nid in enumerate(neuron_ids):
        for ax, cycles, key in (
            (axes_row0[col], rw_cycles, rw_key),
            (axes_row1[col], goal_cycles, goal_key),
        ):
            rm = cycles[key][0]
            ax.imshow(smooth_and_norm(rm[int(nid)]), origin="lower", cmap=RATEMAP_CMAP, vmin=0, vmax=1)
            ax.plot(rx, ry, "w*", markersize=3, markeredgewidth=0.3, markeredgecolor="0.2")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)


def format_lr(lr: float) -> str:
    if lr >= 1e-3:
        return f"{lr:g}"
    return f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan-root", type=str, default="New_Results_v3/dose_sweep_lr_nohead")
    ap.add_argument("--run-dirs", nargs="*", default=None,
                    help="explicit run dirs (overrides scan-root discovery)")
    ap.add_argument("--output", type=str,
                    default=str(SCRIPT_DIR / "figures" / "panels" / "panel_h_nohead_across_lrs"))
    ap.add_argument("--n-neurons", type=int, default=10)
    ap.add_argument("--no_show", action="store_true")
    args = ap.parse_args()

    scan_root = Path(args.scan_root).resolve()
    explicit = [Path(p).resolve() for p in args.run_dirs] if args.run_dirs else None
    runs = discover_runs(scan_root, explicit)
    if not runs:
        raise SystemExit("No no-head runs with cycle_1 / cycle_1_goal probes found.")

    print(f"Found {len(runs)} no-head LR runs:")
    for lr, run in runs:
        print(f"  lr={lr:g}  {run}")

    n_lr = len(runs)
    n = args.n_neurons
    # 2 rows per LR
    fig, axes = plt.subplots(
        2 * n_lr, n,
        figsize=(n * 1.05, 2.15 * n_lr),
        squeeze=False,
    )
    for i, (lr, run) in enumerate(runs):
        cfg = json.load(open(run / "config.json"))
        reward_bin = reward_bin_from_cfg(cfg)
        data_dir = str(run / "data")
        rw_cycles = load_cycle_rate_maps(data_dir, M, HIDDEN_DIM, probe_type="rw")
        goal_cycles = load_cycle_rate_maps(data_dir, M, HIDDEN_DIM, probe_type="goal")
        neuron_ids = top_neuron_ids(goal_cycles, n)
        r0, r1 = 2 * i, 2 * i + 1
        plot_lr_block(axes[r0], axes[r1], rw_cycles, goal_cycles, neuron_ids, reward_bin)
        for col, nid in enumerate(neuron_ids):
            axes[r0][col].set_title(str(int(nid)), fontsize=5, pad=1)
        axes[r0][0].set_ylabel(f"lr={format_lr(lr)}\nPrediction", fontsize=6)
        axes[r1][0].set_ylabel("RL", fontsize=6)

    fig.suptitle(
        "No reward head — cycle 1 Prediction (RW) vs RL (goal) across learning rates",
        fontsize=8, y=1.01,
    )
    fig.tight_layout(pad=0.35, w_pad=0.12, h_pad=0.25)
    stem = Path(args.output)
    ensure_dir(stem.parent)
    save_figure(fig, stem)
    print(f"Saved {stem}.png / .pdf")


if __name__ == "__main__":
    main()
