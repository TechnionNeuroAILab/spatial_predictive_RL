"""Analysis helpers for Figure 3 (high plasticity + LR sweep)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils import (
    DEFAULT_REWARD,
    DEFAULT_SWEEP_ROOT,
    FIG1_DIR,
    FIG2_DIR,
    LR_ORDER,
    apply_nature_style,
    save_figure,
)

# Keep a handle to this package's utils (already imported above).
import utils as _fig3_utils


def _exec_as(name: str, path: Path, utils_mod):
    """Load a sibling module while forcing ``utils`` to resolve to ``utils_mod``."""
    prev = sys.modules.get("utils")
    sys.modules["utils"] = utils_mod
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        assert spec.loader is not None
        sys.path.insert(0, str(path.parent))
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.modules["utils"] = prev if prev is not None else _fig3_utils


# Figure-1 / figure-2 utils are standalone packages.
_fig1_utils_spec = importlib.util.spec_from_file_location("fig1_utils", FIG1_DIR / "utils.py")
_fig1_utils = importlib.util.module_from_spec(_fig1_utils_spec)
assert _fig1_utils_spec.loader is not None
_fig1_utils_spec.loader.exec_module(_fig1_utils)
sys.modules["fig1_utils"] = _fig1_utils

_fig2_utils_spec = importlib.util.spec_from_file_location("fig2_utils", FIG2_DIR / "utils.py")
_fig2_utils = importlib.util.module_from_spec(_fig2_utils_spec)
assert _fig2_utils_spec.loader is not None
_fig2_utils_spec.loader.exec_module(_fig2_utils)
sys.modules["fig2_utils"] = _fig2_utils

_fig1_rm = _exec_as("fig1_rate_maps", FIG1_DIR / "rate_maps.py", _fig1_utils)
_fig2 = _exec_as("fig2_analysis", FIG2_DIR / "analysis.py", _fig2_utils)

load_phases = _fig2.load_phases
shared_place_cell_ids = _fig2.shared_place_cell_ids
paired_distance_stats = _fig2.paired_distance_stats
fig_centers_abc = _fig2.fig_centers_abc
fig_replay_remap_grid = _fig2.fig_replay_remap_grid
select_toward_reward_neurons = _fig1_rm.select_toward_reward_neurons
select_random_neurons = _fig1_rm.select_random_neurons

# Restore figure-3 utils for the rest of this module / callers.
sys.modules["utils"] = _fig3_utils

apply_nature_style()

PHASE_COLORS = {
    "A": "#4C72B0",
    "B": "#C44E52",
    "C_post": "#8172B2",
}


def lr_label(lr: float) -> str:
    if lr < 1e-4:
        return f"{lr:.0e}".replace("e-0", "e−").replace("e-", "e−")
    return f"{lr:g}"


def collect_sweep_metrics(
    sweep_root: Path,
    reward_pos=DEFAULT_REWARD,
    skaggs_threshold: float = 0.05,
) -> list[dict]:
    """Per-LR remapping / map-health metrics from dose_sweep_lr_nohead."""
    rows = []
    for dirname, lr in LR_ORDER:
        run_dir = Path(sweep_root) / dirname
        data_dir = run_dir / "data"
        if not data_dir.exists():
            print(f"  SKIP {dirname} (no data)")
            continue
        phases = load_phases(data_dir)
        ids = shared_place_cell_ids(phases, skaggs_threshold=skaggs_threshold)
        stats = paired_distance_stats(phases, reward_pos, ids)
        # Map health on Phase A / B place cells (same gate).
        sk_a = phases["A"]["sk"][ids]
        sk_b = phases["B"]["sk"][ids]
        rows.append({
            "dirname": dirname,
            "lr": float(lr),
            "n_active": int(len(ids)),
            "com_mean_delta_AB": stats["com_mean_delta_AB"],
            "com_frac_toward_AB": stats["com_frac_toward_AB"],
            "peak_mean_delta_AB": stats["peak_mean_delta_AB"],
            "peak_frac_toward_AB": stats["peak_frac_toward_AB"],
            "com_mean_delta_AC": stats["com_mean_delta_AC"],
            "peak_mean_delta_AC": stats["peak_mean_delta_AC"],
            "mean_d_com_A": float(np.mean(stats["com"]["A"])) if len(ids) else float("nan"),
            "mean_d_com_B": float(np.mean(stats["com"]["B"])) if len(ids) else float("nan"),
            "mean_d_com_C": float(np.mean(stats["com"]["C"])) if len(ids) else float("nan"),
            "mean_d_peak_A": float(np.mean(stats["peak"]["A"])) if len(ids) else float("nan"),
            "mean_d_peak_B": float(np.mean(stats["peak"]["B"])) if len(ids) else float("nan"),
            "mean_d_peak_C": float(np.mean(stats["peak"]["C"])) if len(ids) else float("nan"),
            "skaggs_mean_A": float(np.mean(sk_a)) if len(ids) else float("nan"),
            "skaggs_mean_B": float(np.mean(sk_b)) if len(ids) else float("nan"),
            "delta_com_AB": np.asarray(stats["delta_com_AB"], dtype=float),
            "delta_peak_AB": np.asarray(stats["delta_peak_AB"], dtype=float),
            "d_com_A": np.asarray(stats["com"]["A"], dtype=float),
            "d_com_B": np.asarray(stats["com"]["B"], dtype=float),
            "d_peak_A": np.asarray(stats["peak"]["A"], dtype=float),
            "d_peak_B": np.asarray(stats["peak"]["B"], dtype=float),
            "phases": phases,
            "ids": ids,
            "stats": stats,
        })
        print(
            f"  {dirname}: n={len(ids)} COM_AB={stats['com_mean_delta_AB']:+.3f} "
            f"peak_AB={stats['peak_mean_delta_AB']:+.3f}"
        )
    return rows


def pick_strong_remap_neurons(phases: dict, reward_pos, n: int = 8, seed: int = 42) -> list[int]:
    """Prefer toward-reward remappers (stronger visual); fall back to random active."""
    ids, _meta = select_toward_reward_neurons(
        phases["A"]["rm"], phases["B"]["rm"],
        phases["A"]["sk"], phases["B"]["sk"],
        reward_pos, n=n,
    )
    if len(ids) >= max(4, n // 2):
        return ids[:n]
    return select_random_neurons(phases["A"]["rm"], phases["B"]["rm"], n=n, seed=seed)


def fig_plasticity_boxplots(rows: list[dict], stem):
    """Panel c — boxplots of per-cell remapping Δd vs RL learning rate."""
    if not rows:
        raise RuntimeError("No sweep rows for plasticity boxplots.")

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.3))
    positions = np.arange(1, len(rows) + 1)
    labels = [lr_label(r["lr"]) for r in rows]

    for ax, key, title, color in [
        (axes[0], "delta_com_AB", r"COM $\Delta$d (A→B)", "#E69F00"),
        (axes[1], "delta_peak_AB", r"Peak $\Delta$d (A→B)", "#009E73"),
    ]:
        data = [r[key] for r in rows]
        bp = ax.boxplot(
            data, positions=positions, widths=0.55, patch_artist=True,
            showfliers=False, medianprops=dict(color="0.15", lw=1.0),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
            patch.set_edgecolor("0.25")
        means = [float(np.mean(d)) if len(d) else np.nan for d in data]
        ax.plot(positions, means, "o-", color="0.2", ms=3.5, lw=0.9, label="mean")
        ax.axhline(0, color="0.4", lw=0.6, ls="--")
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_xlabel(r"RL learning rate (plasticity)")
        ax.set_ylabel(title)
        ax.set_title(title + "\n(< 0 = toward reward)", fontsize=6, pad=2)
        ax.legend(frameon=False, loc="best", fontsize=5)

    fig.suptitle(
        "Plasticity sweep — remapping strengthens with RL learning rate (no head, seed 42)",
        fontsize=6.5, y=1.02,
    )
    fig.tight_layout(pad=0.4, w_pad=0.5)
    return save_figure(fig, stem)


def fig_supp_mean_distance(rows: list[dict], stem):
    """Supp a — mean COM / peak distance to reward vs LR (A and B)."""
    lrs = [r["lr"] for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.1))

    for ax, metric, title in [
        (axes[0], "com", "COM"),
        (axes[1], "peak", "Peak"),
    ]:
        a = [r[f"mean_d_{metric}_A"] for r in rows]
        b = [r[f"mean_d_{metric}_B"] for r in rows]
        ax.plot(x, a, "o-", color=PHASE_COLORS["A"], label="A prediction", ms=4)
        ax.plot(x, b, "s-", color=PHASE_COLORS["B"], label="B RL policy", ms=4)
        ax.set_xticks(x)
        ax.set_xticklabels([lr_label(lr) for lr in lrs], rotation=25, ha="right")
        ax.set_xlabel("RL learning rate")
        ax.set_ylabel(f"Mean {title} distance to reward")
        ax.set_title(f"Mean {title} distance vs plasticity", fontsize=6, pad=2)
        ax.legend(frameon=False, fontsize=5)

    fig.suptitle("Supplementary — mean distance to reward across plasticity", fontsize=6.5, y=1.02)
    fig.tight_layout(pad=0.35, w_pad=0.45)
    return save_figure(fig, stem)


def fig_supp_n_active(rows: list[dict], stem):
    """Supp b — number of active place cells vs LR."""
    fig, ax = plt.subplots(figsize=(3.2, 2.1))
    x = np.arange(len(rows))
    ns = [r["n_active"] for r in rows]
    ax.bar(x, ns, color="#56B4E9", edgecolor="0.3", width=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels([lr_label(r["lr"]) for r in rows], rotation=25, ha="right")
    ax.set_xlabel("RL learning rate")
    ax.set_ylabel("# place cells (A∩B∩C)")
    ax.set_title("Active place cells vs plasticity", fontsize=6, pad=2)
    fig.tight_layout(pad=0.35)
    return save_figure(fig, stem)


def fig_supp_skaggs(rows: list[dict], stem):
    """Supp c — mean Skaggs information vs LR."""
    fig, ax = plt.subplots(figsize=(3.2, 2.1))
    x = np.arange(len(rows))
    ax.plot(x, [r["skaggs_mean_A"] for r in rows], "o-", color=PHASE_COLORS["A"],
            label="A prediction", ms=4)
    ax.plot(x, [r["skaggs_mean_B"] for r in rows], "s-", color=PHASE_COLORS["B"],
            label="B RL policy", ms=4)
    ax.set_xticks(x)
    ax.set_xticklabels([lr_label(r["lr"]) for r in rows], rotation=25, ha="right")
    ax.set_xlabel("RL learning rate")
    ax.set_ylabel("Mean Skaggs information")
    ax.set_title("Spatial information vs plasticity", fontsize=6, pad=2)
    ax.legend(frameon=False, fontsize=5)
    fig.tight_layout(pad=0.35)
    return save_figure(fig, stem)


def fig_supp_distance_violins(rows: list[dict], stem):
    """Supp d — violins of COM distance-to-reward (A vs B) per LR + mean markers."""
    fig, ax = plt.subplots(figsize=(6.2, 2.4))
    # Two violins per LR: A and B, offset
    positions_a, positions_b = [], []
    data_a, data_b = [], []
    means_a, means_b = [], []
    xticks, xticklabels = [], []
    for i, r in enumerate(rows):
        pa, pb = 3 * i + 1, 3 * i + 2
        positions_a.append(pa)
        positions_b.append(pb)
        data_a.append(r["d_com_A"])
        data_b.append(r["d_com_B"])
        means_a.append(r["mean_d_com_A"])
        means_b.append(r["mean_d_com_B"])
        xticks.append(pa + 0.5)
        xticklabels.append(lr_label(r["lr"]))

    parts_a = ax.violinplot(data_a, positions=positions_a, widths=0.85,
                            showmeans=False, showextrema=False, showmedians=False)
    parts_b = ax.violinplot(data_b, positions=positions_b, widths=0.85,
                            showmeans=False, showextrema=False, showmedians=False)
    for body in parts_a["bodies"]:
        body.set_facecolor(PHASE_COLORS["A"])
        body.set_alpha(0.55)
        body.set_edgecolor("0.25")
    for body in parts_b["bodies"]:
        body.set_facecolor(PHASE_COLORS["B"])
        body.set_alpha(0.55)
        body.set_edgecolor("0.25")

    ax.plot(positions_a, means_a, "o", color=PHASE_COLORS["A"], ms=4, zorder=5, label="A mean")
    ax.plot(positions_b, means_b, "s", color=PHASE_COLORS["B"], ms=4, zorder=5, label="B mean")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=25, ha="right")
    ax.set_xlabel("RL learning rate")
    ax.set_ylabel("COM distance to reward")
    ax.set_title("Field-centre distance distributions (A vs B) across plasticity", fontsize=6, pad=2)
    ax.legend(frameon=False, fontsize=5, loc="upper right")
    fig.tight_layout(pad=0.35)
    return save_figure(fig, stem)


def rows_to_jsonable(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        out.append({
            k: (float(v) if isinstance(v, (float, np.floating)) else
                int(v) if isinstance(v, (int, np.integer)) else v)
            for k, v in r.items()
            if k not in ("delta_com_AB", "delta_peak_AB", "d_com_A", "d_com_B",
                         "d_peak_A", "d_peak_B", "phases", "ids", "stats")
        })
    return out
