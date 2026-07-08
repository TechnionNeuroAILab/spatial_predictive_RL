#!/usr/bin/env python3
"""
Filtered place-cell shift analysis for interleaved RatInABox runs.

This is a corrected version of the earlier top-100-only analysis. It tries to do
what the previous project analysis did conceptually:

1. Build rate maps per neuron.
2. Compute Skaggs spatial information.
3. Fit 1..K Gaussian mixtures to each map and select K by ICL.
4. Filter place cells by Skaggs + allowed number of centers, usually K in {1, 2}.
5. Compare cycle A vs cycle B and test whether fields moved toward reward.

The script supports three data sources, in this order when --source auto:

A) Exact long-format probe data:
   df_behavior_*.pkl with columns timestep,x,y and df_activations_*.pkl with
   columns timestep,neuron_id,activity.

B) Exact/near-exact saved rate maps:
   NPZ files with a rate_maps array and optional neuron_ids/skaggs/occupancy.

C) Fallback for existing v11 runs:
   the saved top-100 Skaggs PNGs + skaggs_post_*.npz. This is approximate and
   only covers the top-100 plotted cells; it cannot be considered a true all-cell
   place-cell filter.

Example, existing v11 folder fallback:
    python analyze_filtered_place_cell_shifts.py \
        --run_dir /path/to/run \
        --cycle_a 1 --cycle_b 5 --phase pred \
        --reward 0.5 0.5 \
        --allowed_k 1 2 \
        --no_show

Example, exact data if you saved per-cycle probes:
    python analyze_filtered_place_cell_shifts.py \
        --behavior_a data/cycle_01/df_behavior_post_pred.pkl \
        --activity_a data/cycle_01/df_activations_post_pred.pkl \
        --behavior_b data/cycle_05/df_behavior_post_pred.pkl \
        --activity_b data/cycle_05/df_activations_post_pred.pkl \
        --reward 0.5 0.5 --allowed_k 1 2
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from scipy import ndimage, stats
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist
from sklearn.mixture import GaussianMixture

from pc_common import resolve_reward_pos, save_all


EPS = 1e-12


@dataclass
class ConditionData:
    label: str
    source: str
    rate_maps: np.ndarray          # shape: (N, nbins, nbins), y,x order
    neuron_ids: np.ndarray         # shape: (N,)
    occupancy: Optional[np.ndarray]
    skaggs_from_file: Optional[np.ndarray]
    diagnostics: Dict[str, object]


@dataclass
class NeuronFields:
    neuron_id: int
    skaggs: float
    k: int
    icl: float
    centers: np.ndarray            # shape: (k, 2), x,y
    peak_rate: float
    mean_rate: float
    is_place_cell: bool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filtered place-cell shift analysis.")

    p.add_argument("--run_dir", type=str, default=None,
                   help="Run directory containing figures/, data/, checkpoints/. Optional if explicit files are given.")
    p.add_argument("--cycle_a", type=int, default=1)
    p.add_argument("--cycle_b", type=int, default=5)
    p.add_argument("--phase", choices=["pred", "rl"], default="pred")
    p.add_argument("--representation", choices=["rnn", "controller"], default="rnn")
    p.add_argument("--reward", nargs=2, type=float, default=None,
                   help="Reward (x, y). Defaults to run_dir/config.json's reward_pos.")

    p.add_argument("--source", choices=["auto", "long", "maps", "png"], default="auto",
                   help="Data source. auto tries long, then maps, then png.")
    p.add_argument("--behavior_a", type=str, default=None)
    p.add_argument("--activity_a", type=str, default=None)
    p.add_argument("--behavior_b", type=str, default=None)
    p.add_argument("--activity_b", type=str, default=None)
    p.add_argument("--state_a", type=str, default=None,
                   help="v11 state token for condition A, e.g. before/after/cycle_1/cycle_5. "
                        "Resolves directly to run_dir/data/df_behavior_{state}.pkl etc.")
    p.add_argument("--state_b", type=str, default=None,
                   help="v11 state token for condition B, e.g. before/after/cycle_1/cycle_5.")
    p.add_argument("--maps_a", type=str, default=None,
                   help="NPZ containing rate_maps for condition A.")
    p.add_argument("--maps_b", type=str, default=None,
                   help="NPZ containing rate_maps for condition B.")

    p.add_argument("--nbins", type=int, default=32)
    p.add_argument("--smooth_sigma", type=float, default=1.0)
    p.add_argument("--min_rate_frac", type=float, default=0.30,
                   help="Use bins >= min_rate_frac * peak for GMM fitting.")
    p.add_argument("--k_max", type=int, default=4)
    p.add_argument("--field_method", choices=["components", "gmm_icl"], default="components",
                   help="components is fast and robust; gmm_icl approximates the older ICL/GMM method but is slower.")
    p.add_argument("--min_component_bins", type=int, default=2,
                   help="Minimum connected hot bins for a field when --field_method components.")
    p.add_argument("--max_gmm_bins", type=int, default=200,
                   help="Maximum active bins used for GMM/ICL per neuron. Speeds analysis and stabilizes PNG fallback.")
    p.add_argument("--allowed_k", nargs="+", type=int, default=[1, 2],
                   help="Allowed number of place-field centers for shift analysis.")
    p.add_argument("--skaggs_threshold", type=float, default=0.20,
                   help="Absolute Skaggs threshold. Use 0 to disable.")
    p.add_argument("--skaggs_quantile", type=float, default=None,
                   help="Optional within-condition quantile threshold, e.g. 0.80.")
    p.add_argument("--icl_max", type=float, default=None,
                   help="Optional ICL cutoff. Older scripts used values like -29000, but this is source-dependent.")
    p.add_argument("--filter_mode", choices=["baseline", "both"], default="both",
                   help="baseline: cell must pass in A and have allowed centers in B. both: pass in both A and B.")

    p.add_argument("--matching", choices=["min_to_reward", "hungarian", "both"], default="both",
                   help="How to summarize multi-center shifts.")
    p.add_argument("--near_radius", type=float, default=0.15,
                   help="Radius used for near-reward fractions.")
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--top_n", type=int, default=100,
                   help="Expected plotted cells in PNG fallback.")
    p.add_argument("--png_hot_percentile", type=float, default=85.0,
                   help="Percentile used to form hot bins in PNG fallback maps.")
    p.add_argument("--png_min_component_area", type=int, default=3000)

    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--no_show", action="store_true", help="Accepted for compatibility. Figures are saved only.")
    return p.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_float(x: object) -> Optional[float]:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return None


def cycle_paths(run_dir: Path, cycle: int, phase: str, representation: str) -> Dict[str, Path]:
    fig_dir = run_dir / "figures" / f"cycle_{cycle:02d}"
    data_dir = run_dir / "data" / f"cycle_{cycle:02d}"

    if representation == "rnn":
        fig_name = f"post_{phase}_top100_skaggs.png"
        npz_name = f"skaggs_post_{phase}.npz"
    else:
        fig_name = f"post_{phase}_controller_top100_skaggs.png"
        npz_name = f"skaggs_post_{phase}_controller.npz"

    return {
        "fig": fig_dir / fig_name,
        "skaggs_npz": data_dir / npz_name,
        "data_dir": data_dir,
    }


def candidate_long_paths(run_dir: Path, cycle: int, phase: str) -> Tuple[Optional[Path], Optional[Path]]:
    data_dir = run_dir / "data" / f"cycle_{cycle:02d}"
    labels = [
        f"post_{phase}", phase, f"cycle_{cycle:02d}_{phase}", f"cycle{cycle}_{phase}",
        f"after_{phase}", f"{phase}_probe",
    ]
    behavior_names = []
    activity_names = []
    for label in labels:
        behavior_names.extend([
            f"df_behavior_{label}.pkl", f"behavior_{label}.pkl",
        ])
        activity_names.extend([
            f"df_activations_{label}.pkl", f"df_activity_{label}.pkl", f"activations_{label}.pkl",
        ])

    beh = next((data_dir / n for n in behavior_names if (data_dir / n).exists()), None)
    act = next((data_dir / n for n in activity_names if (data_dir / n).exists()), None)
    return beh, act


def candidate_map_path(run_dir: Path, cycle: int, phase: str, representation: str) -> Optional[Path]:
    data_dir = run_dir / "data" / f"cycle_{cycle:02d}"
    names = [
        f"rate_maps_post_{phase}_{representation}.npz",
        f"rate_maps_post_{phase}.npz" if representation == "rnn" else f"rate_maps_post_{phase}_controller.npz",
        f"place_maps_post_{phase}_{representation}.npz",
        f"place_maps_post_{phase}.npz" if representation == "rnn" else f"place_maps_post_{phase}_controller.npz",
        f"maps_post_{phase}_{representation}.npz",
        f"maps_post_{phase}.npz" if representation == "rnn" else f"maps_post_{phase}_controller.npz",
    ]
    return next((data_dir / n for n in names if (data_dir / n).exists()), None)


def flat_state_paths(run_dir: Path, state: str) -> Tuple[Path, Path]:
    """Direct flat-layout paths matching v11's actual output:
    run_dir/data/df_behavior_{state}.pkl, df_activations_{state}.pkl
    (state e.g. "before", "after", "cycle_1", "cycle_5")."""
    data_dir = run_dir / "data"
    return data_dir / f"df_behavior_{state}.pkl", data_dir / f"df_activations_{state}.pkl"


def find_long_files(args: argparse.Namespace, cycle: int) -> Tuple[Optional[Path], Optional[Path]]:
    if cycle == args.cycle_a and args.behavior_a and args.activity_a:
        return Path(args.behavior_a), Path(args.activity_a)
    if cycle == args.cycle_b and args.behavior_b and args.activity_b:
        return Path(args.behavior_b), Path(args.activity_b)
    if args.run_dir and cycle == args.cycle_a and args.state_a:
        return flat_state_paths(Path(args.run_dir), args.state_a)
    if args.run_dir and cycle == args.cycle_b and args.state_b:
        return flat_state_paths(Path(args.run_dir), args.state_b)
    if args.run_dir:
        return candidate_long_paths(Path(args.run_dir), cycle, args.phase)
    return None, None


def find_map_file(args: argparse.Namespace, cycle: int) -> Optional[Path]:
    if cycle == args.cycle_a and args.maps_a:
        return Path(args.maps_a)
    if cycle == args.cycle_b and args.maps_b:
        return Path(args.maps_b)
    if args.run_dir:
        return candidate_map_path(Path(args.run_dir), cycle, args.phase, args.representation)
    return None


def build_rate_maps_from_long(behavior_path: Path, activity_path: Path, nbins: int, smooth_sigma: float,
                              label: str) -> ConditionData:
    df_beh = pd.read_pickle(behavior_path)
    df_act = pd.read_pickle(activity_path)

    required_beh = {"timestep", "x", "y"}
    required_act = {"timestep", "neuron_id", "activity"}
    if not required_beh.issubset(df_beh.columns):
        raise ValueError(f"{behavior_path} missing columns {required_beh - set(df_beh.columns)}")
    if not required_act.issubset(df_act.columns):
        raise ValueError(f"{activity_path} missing columns {required_act - set(df_act.columns)}")

    df_beh_small = df_beh[["timestep", "x", "y"]].copy()
    df_beh_small["timestep"] = df_beh_small["timestep"].astype(np.int64)
    bx = np.clip((df_beh_small["x"].to_numpy(float) * nbins).astype(int), 0, nbins - 1)
    by = np.clip((df_beh_small["y"].to_numpy(float) * nbins).astype(int), 0, nbins - 1)
    bin_idx_by_timestep = pd.DataFrame({
        "timestep": df_beh_small["timestep"].to_numpy(np.int64),
        "bin_idx": (by * nbins + bx).astype(np.int32),
    })

    occ = np.bincount(bin_idx_by_timestep["bin_idx"].to_numpy(), minlength=nbins * nbins).astype(np.float64)
    occupancy = occ / max(float(occ.sum()), 1.0)

    df_act_small = df_act[["timestep", "neuron_id", "activity"]].copy()
    df_act_small["timestep"] = df_act_small["timestep"].astype(np.int64)
    df_act_small["neuron_id"] = df_act_small["neuron_id"].astype(np.int64)
    df_act_small["activity"] = df_act_small["activity"].astype(np.float32)
    merged = df_act_small.merge(bin_idx_by_timestep, on="timestep", how="inner")

    neuron_ids = np.sort(merged["neuron_id"].unique().astype(int))
    nid_to_row = {int(nid): i for i, nid in enumerate(neuron_ids)}
    merged["row"] = merged["neuron_id"].map(nid_to_row).astype(np.int32)

    rate_sum = np.zeros((len(neuron_ids), nbins * nbins), dtype=np.float64)
    np.add.at(
        rate_sum,
        (merged["row"].to_numpy(np.int32), merged["bin_idx"].to_numpy(np.int32)),
        merged["activity"].to_numpy(np.float64),
    )
    denom = np.maximum(occ, 1.0)[None, :]
    rate_flat = rate_sum / denom
    rate_flat[:, occ == 0] = 0.0
    rate_maps = rate_flat.reshape(len(neuron_ids), nbins, nbins)

    if smooth_sigma and smooth_sigma > 0:
        for i in range(rate_maps.shape[0]):
            rate_maps[i] = ndimage.gaussian_filter(rate_maps[i], sigma=float(smooth_sigma))

    return ConditionData(
        label=label,
        source="long",
        rate_maps=rate_maps.astype(np.float32),
        neuron_ids=neuron_ids.astype(int),
        occupancy=occupancy.reshape(nbins, nbins),
        skaggs_from_file=None,
        diagnostics={
            "behavior_path": str(behavior_path),
            "activity_path": str(activity_path),
            "n_behavior_rows": int(len(df_beh)),
            "n_activity_rows": int(len(df_act)),
            "n_neurons": int(len(neuron_ids)),
        },
    )


def load_maps_npz(path: Path, nbins: int, smooth_sigma: float, label: str) -> ConditionData:
    z = np.load(path, allow_pickle=True)
    map_key = None
    for k in ["rate_maps", "activity_maps", "maps", "place_maps"]:
        if k in z.files:
            map_key = k
            break
    if map_key is None:
        raise ValueError(f"{path} does not contain one of rate_maps/activity_maps/maps/place_maps")

    maps = np.asarray(z[map_key], dtype=np.float32)
    if maps.ndim != 3:
        raise ValueError(f"{path}:{map_key} must have shape (N,H,W), got {maps.shape}")
    if maps.shape[1] != nbins or maps.shape[2] != nbins:
        resized = np.zeros((maps.shape[0], nbins, nbins), dtype=np.float32)
        for i in range(maps.shape[0]):
            img = Image.fromarray(maps[i].astype(np.float32))
            resized[i] = np.asarray(img.resize((nbins, nbins), Image.Resampling.BILINEAR), dtype=np.float32)
        maps = resized

    if smooth_sigma and smooth_sigma > 0:
        for i in range(maps.shape[0]):
            maps[i] = ndimage.gaussian_filter(maps[i], sigma=float(smooth_sigma))

    if "neuron_ids" in z.files:
        neuron_ids = np.asarray(z["neuron_ids"], dtype=int)
    elif "top_idx" in z.files:
        neuron_ids = np.asarray(z["top_idx"], dtype=int)[:maps.shape[0]]
    else:
        neuron_ids = np.arange(maps.shape[0], dtype=int)

    skaggs = np.asarray(z["skaggs"], dtype=float) if "skaggs" in z.files else None
    if skaggs is not None and skaggs.size != maps.shape[0]:
        # It may be full length with neuron IDs indexing into it.
        try:
            skaggs = skaggs[neuron_ids]
        except Exception:
            skaggs = None

    occupancy = np.asarray(z["occupancy"], dtype=float) if "occupancy" in z.files else None
    if occupancy is not None and occupancy.shape != (nbins, nbins):
        occupancy = None

    return ConditionData(
        label=label,
        source="maps",
        rate_maps=maps,
        neuron_ids=neuron_ids[:maps.shape[0]].astype(int),
        occupancy=occupancy,
        skaggs_from_file=skaggs,
        diagnostics={"maps_path": str(path), "map_key": map_key, "n_neurons": int(maps.shape[0])},
    )


def rgb_to_jet_activity(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.float32) / 255.0
    hsv = mcolors.rgb_to_hsv(arr)
    hue = hsv[..., 0]
    sat = hsv[..., 1]
    val = hsv[..., 2]
    activity = 1.0 - np.clip(hue, 0.0, 2.0 / 3.0) / (2.0 / 3.0)
    activity = activity * np.clip(sat * 1.25, 0.0, 1.0) * np.clip(val * 1.10, 0.0, 1.0)
    return np.nan_to_num(activity.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def colored_tile_components(image_path: Path, min_area: int) -> Tuple[List[Tuple[int, int, int, int, int]], Dict[str, object]]:
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img)
    rgb = arr.astype(np.float32) / 255.0
    maxc = rgb.max(axis=2)
    minc = rgb.min(axis=2)
    sat = (maxc - minc) / np.maximum(maxc, 1e-6)
    mask = (sat > 0.35) & (maxc > 0.20)
    mask = ndimage.binary_opening(mask, structure=np.ones((2, 2)))
    labels, nlab = ndimage.label(mask)

    H, W = mask.shape
    boxes = []
    for lab in range(1, nlab + 1):
        ys, xs = np.where(labels == lab)
        if ys.size == 0:
            continue
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        area = int(ys.size)
        h = y1 - y0
        w = x1 - x0
        if area < min_area or h < 40 or w < 40:
            continue
        if h > 0.30 * H or w > 0.30 * W:
            continue
        boxes.append((y0, y1, x0, x1, area))
    boxes = sorted(boxes, key=lambda b: (b[0], b[2]))
    return boxes, {"image_size": [int(W), int(H)], "raw_components": int(nlab), "kept_components": int(len(boxes))}


def load_png_fallback(fig_path: Path, skaggs_npz_path: Path, nbins: int, top_n: int,
                      hot_percentile: float, min_component_area: int, label: str) -> ConditionData:
    if not fig_path.exists():
        raise FileNotFoundError(fig_path)
    if not skaggs_npz_path.exists():
        raise FileNotFoundError(skaggs_npz_path)

    z = np.load(skaggs_npz_path, allow_pickle=True)
    top_idx = np.asarray(z["top_idx"], dtype=int)[:top_n] if "top_idx" in z.files else np.arange(top_n, dtype=int)
    skaggs_full = np.asarray(z["skaggs"], dtype=float) if "skaggs" in z.files else None
    skaggs = None
    if skaggs_full is not None:
        if skaggs_full.size >= int(np.max(top_idx)) + 1:
            skaggs = skaggs_full[top_idx]
        else:
            skaggs = skaggs_full[:len(top_idx)]

    boxes, diag = colored_tile_components(fig_path, min_component_area)
    if len(boxes) < min(top_n, len(top_idx)):
        raise RuntimeError(f"Detected only {len(boxes)} heatmap tiles in {fig_path}; expected about {top_n}.")

    img = Image.open(fig_path).convert("RGB")
    arr = np.asarray(img)
    n = min(top_n, len(top_idx), len(boxes))
    maps = np.zeros((n, nbins, nbins), dtype=np.float32)
    for i, (y0, y1, x0, x1, area) in enumerate(boxes[:n]):
        crop = arr[y0:y1, x0:x1, :]
        act = rgb_to_jet_activity(crop)
        # Remove low-value label or axis residues by using only colored values.
        thr = np.percentile(act, max(1.0, min(99.0, hot_percentile)))
        act = np.where(act >= 0.02, act, 0.0)
        if np.nanmax(act) > 0:
            act = act / np.nanmax(act)
        pil = Image.fromarray((act * 255.0).astype(np.uint8))
        maps[i] = np.asarray(pil.resize((nbins, nbins), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        if np.nanmax(maps[i]) > 0:
            maps[i] = maps[i] / np.nanmax(maps[i])

    return ConditionData(
        label=label,
        source="png_fallback",
        rate_maps=maps,
        neuron_ids=top_idx[:n].astype(int),
        occupancy=None,
        skaggs_from_file=skaggs[:n] if skaggs is not None else None,
        diagnostics={
            "fig_path": str(fig_path),
            "skaggs_npz_path": str(skaggs_npz_path),
            "warning": "PNG fallback is approximate and only includes plotted top cells, not all neurons.",
            **diag,
        },
    )


def load_condition(args: argparse.Namespace, cycle: int, label: str) -> ConditionData:
    errors = []

    if args.source in ["auto", "long"]:
        beh, act = find_long_files(args, cycle)
        if beh is not None and act is not None and beh.exists() and act.exists():
            return build_rate_maps_from_long(beh, act, args.nbins, args.smooth_sigma, label)
        errors.append(f"long data not found for cycle {cycle}")
        if args.source == "long":
            raise FileNotFoundError(errors[-1])

    if args.source in ["auto", "maps"]:
        mp = find_map_file(args, cycle)
        if mp is not None and mp.exists():
            return load_maps_npz(mp, args.nbins, args.smooth_sigma, label)
        errors.append(f"map NPZ not found for cycle {cycle}")
        if args.source == "maps":
            raise FileNotFoundError(errors[-1])

    if args.source in ["auto", "png"]:
        if not args.run_dir:
            raise ValueError("--run_dir is required for PNG fallback")
        paths = cycle_paths(Path(args.run_dir), cycle, args.phase, args.representation)
        if paths["fig"].exists() and paths["skaggs_npz"].exists():
            return load_png_fallback(
                paths["fig"], paths["skaggs_npz"], args.nbins, args.top_n,
                args.png_hot_percentile, args.png_min_component_area, label,
            )
        errors.append(f"PNG fallback files not found for cycle {cycle}")
        if args.source == "png":
            raise FileNotFoundError(errors[-1])

    raise FileNotFoundError("Could not load condition data. Tried: " + "; ".join(errors))


def compute_skaggs_from_maps(rate_maps: np.ndarray, occupancy: Optional[np.ndarray]) -> np.ndarray:
    N, H, W = rate_maps.shape
    rm = np.nan_to_num(rate_maps.reshape(N, H * W).astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if occupancy is None:
        occ = np.ones(H * W, dtype=np.float64) / float(H * W)
    else:
        occ = np.asarray(occupancy, dtype=np.float64).reshape(H * W)
        if occ.sum() <= 0:
            occ = np.ones(H * W, dtype=np.float64) / float(H * W)
        else:
            occ = occ / occ.sum()
    mean_rate = rm @ occ
    ratio = rm / (mean_rate[:, None] + EPS)
    log_ratio = np.where(ratio > 0, np.log2(ratio + EPS), 0.0)
    sk = np.sum(occ[None, :] * rm * log_ratio / (mean_rate[:, None] + EPS), axis=1)
    return np.nan_to_num(sk, nan=0.0, posinf=0.0, neginf=0.0)


def map_to_weighted_coords(rate_map: np.ndarray, min_rate_frac: float, max_bins: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    H, W = rate_map.shape
    rm = np.nan_to_num(rate_map.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(rm)) if rm.size else 0.0
    if peak <= 0:
        return np.empty((0, 2), dtype=float), np.empty((0,), dtype=float)
    mask = rm >= (min_rate_frac * peak)
    if int(mask.sum()) < 3:
        mask = rm >= np.percentile(rm[rm > 0], 90) if np.any(rm > 0) else mask
    ys, xs = np.where(mask)
    coords = np.column_stack(((xs + 0.5) / W, (ys + 0.5) / H))
    weights = rm[ys, xs]
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    weights = np.maximum(weights, EPS)
    if max_bins is not None and max_bins > 0 and len(weights) > max_bins:
        keep = np.argsort(weights)[-int(max_bins):]
        coords = coords[keep]
        weights = weights[keep]
    return coords, weights


def weighted_log_likelihood(gmm: GaussianMixture, X: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * gmm.score_samples(X)))


def icl_score(gmm: GaussianMixture, X: np.ndarray, w: np.ndarray, ll: float) -> float:
    k = int(gmm.n_components)
    # Full covariance in 2D: means 2*k, cov 3*k, weights k-1.
    n_params = k * (2 + 3) + (k - 1)
    n_eff = max(float(np.sum(w)), 1.0)
    bic = -2.0 * ll + n_params * np.log(n_eff)
    resp = np.clip(gmm.predict_proba(X), EPS, 1.0)
    entropy = -float(np.sum(w[:, None] * resp * np.log(resp)))
    return float(bic + 2.0 * entropy)


def fit_fields_by_components(rate_map: np.ndarray, k_max: int, min_rate_frac: float, min_component_bins: int) -> Tuple[int, float, np.ndarray]:
    """Fast field extraction by connected components of high-activity bins.

    Returns k, pseudo_icl, centers. The pseudo_icl is negative total hot-bin mass
    so lower values are better, but it should not be compared to true ICL.
    """
    H, W = rate_map.shape
    rm = np.nan_to_num(rate_map.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(rm)) if rm.size else 0.0
    if peak <= 0:
        return 0, float("inf"), np.empty((0, 2), dtype=float)
    mask = rm >= (float(min_rate_frac) * peak)
    if int(mask.sum()) < int(min_component_bins):
        vals = rm[rm > 0]
        if vals.size:
            mask = rm >= np.percentile(vals, 90)
    labels, nlab = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    comps = []
    for lab in range(1, nlab + 1):
        ys, xs = np.where(labels == lab)
        if ys.size < int(min_component_bins):
            continue
        weights = rm[ys, xs]
        mass = float(np.sum(weights))
        if mass <= 0:
            continue
        cx = float(np.sum(((xs + 0.5) / W) * weights) / mass)
        cy = float(np.sum(((ys + 0.5) / H) * weights) / mass)
        comps.append((mass, cx, cy, int(ys.size)))
    if not comps:
        ys, xs = np.unravel_index(int(np.argmax(rm)), rm.shape)
        return 1, -peak, np.asarray([[(xs + 0.5) / W, (ys + 0.5) / H]], dtype=float)
    comps = sorted(comps, key=lambda t: t[0], reverse=True)[:int(k_max)]
    centers = np.asarray([[c[1], c[2]] for c in comps], dtype=float)
    pseudo_icl = -float(np.sum([c[0] for c in comps]))
    return int(len(comps)), pseudo_icl, np.clip(centers, 0.0, 1.0)


def fit_fields_for_map(rate_map: np.ndarray, k_max: int, min_rate_frac: float, random_state: int, max_gmm_bins: int) -> Tuple[int, float, np.ndarray]:
    X, w = map_to_weighted_coords(rate_map, min_rate_frac, max_bins=max_gmm_bins)
    if X.shape[0] == 0:
        return 0, float("inf"), np.empty((0, 2), dtype=float)
    if X.shape[0] < 3:
        c = np.average(X, axis=0, weights=w) if w.sum() > 0 else X.mean(axis=0)
        return 1, float("inf"), np.asarray([c], dtype=float)

    max_k = min(int(k_max), X.shape[0])
    best = None
    for k in range(1, max_k + 1):
        try:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                reg_covar=1e-3,
                random_state=random_state,
                n_init=2,
            )
            gmm.fit(X)
            ll = weighted_log_likelihood(gmm, X, w)
            icl = icl_score(gmm, X, w, ll)
            if best is None or icl < best[0]:
                best = (icl, k, gmm.means_.copy())
        except Exception:
            continue

    if best is None:
        c = np.average(X, axis=0, weights=w) if w.sum() > 0 else X.mean(axis=0)
        return 1, float("inf"), np.asarray([c], dtype=float)
    icl, k, centers = best
    centers = np.clip(np.asarray(centers, dtype=float), 0.0, 1.0)
    # Sort centers by distance to reward is done later; here stable sort by x then y.
    order = np.lexsort((centers[:, 1], centers[:, 0]))
    return int(k), float(icl), centers[order]


def analyze_condition(cond: ConditionData, args: argparse.Namespace) -> pd.DataFrame:
    if cond.skaggs_from_file is not None and len(cond.skaggs_from_file) == cond.rate_maps.shape[0]:
        skaggs = np.asarray(cond.skaggs_from_file, dtype=float)
    else:
        skaggs = compute_skaggs_from_maps(cond.rate_maps, cond.occupancy)

    if args.skaggs_quantile is not None:
        qthr = float(np.nanquantile(skaggs, args.skaggs_quantile))
        sk_thr = max(float(args.skaggs_threshold), qthr)
    else:
        qthr = None
        sk_thr = float(args.skaggs_threshold)

    rows = []
    allowed = set(int(x) for x in args.allowed_k)
    for i, nid in enumerate(cond.neuron_ids):
        rm = cond.rate_maps[i]
        if args.field_method == "components":
            k, icl, centers = fit_fields_by_components(rm, args.k_max, args.min_rate_frac, args.min_component_bins)
        else:
            k, icl, centers = fit_fields_for_map(rm, args.k_max, args.min_rate_frac, args.seed + int(nid), args.max_gmm_bins)
        peak = float(np.nanmax(rm)) if rm.size else 0.0
        mean = float(np.nanmean(rm)) if rm.size else 0.0
        passes = (float(skaggs[i]) >= sk_thr) and (k in allowed)
        if args.icl_max is not None:
            passes = passes and (icl <= float(args.icl_max))
        rows.append({
            "condition": cond.label,
            "source": cond.source,
            "neuron_id": int(nid),
            "skaggs": float(skaggs[i]),
            "k": int(k),
            "icl": float(icl),
            "centers": json.dumps(centers.tolist()),
            "peak_rate": peak,
            "mean_rate": mean,
            "skaggs_threshold_used": sk_thr,
            "skaggs_quantile_threshold": qthr,
            "is_place_cell": bool(passes),
        })
    return pd.DataFrame(rows)


def parse_centers(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(float)
    if isinstance(value, str):
        arr = np.asarray(json.loads(value), dtype=float)
    else:
        arr = np.asarray(value, dtype=float)
    if arr.ndim == 1 and arr.size == 2:
        arr = arr.reshape(1, 2)
    if arr.size == 0:
        return np.empty((0, 2), dtype=float)
    return arr.astype(float)


def dist_to_reward(centers: np.ndarray, reward: np.ndarray) -> np.ndarray:
    if centers.size == 0:
        return np.empty((0,), dtype=float)
    return np.linalg.norm(centers - reward[None, :], axis=1)


def nearest_reward_center(centers: np.ndarray, reward: np.ndarray) -> Tuple[np.ndarray, float, int]:
    d = dist_to_reward(centers, reward)
    if d.size == 0:
        return np.array([np.nan, np.nan]), np.nan, -1
    idx = int(np.argmin(d))
    return centers[idx], float(d[idx]), idx


def projection_toward_reward(center_a: np.ndarray, center_b: np.ndarray, reward: np.ndarray) -> float:
    if np.any(~np.isfinite(center_a)) or np.any(~np.isfinite(center_b)):
        return np.nan
    target_vec = reward - center_a
    norm = float(np.linalg.norm(target_vec))
    if norm < EPS:
        return np.nan
    return float(np.dot(center_b - center_a, target_vec / norm))


def paired_shift_tables(df_a: pd.DataFrame, df_b: pd.DataFrame, reward: np.ndarray,
                        filter_mode: str, allowed_k: Sequence[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    allowed = set(int(x) for x in allowed_k)
    a = df_a.copy()
    b = df_b.copy()
    if filter_mode == "both":
        a = a[a["is_place_cell"]]
        b = b[b["is_place_cell"]]
    elif filter_mode == "baseline":
        a = a[a["is_place_cell"]]
        b = b[b["k"].isin(allowed)]
    else:
        raise ValueError(filter_mode)

    merged = a.merge(b, on="neuron_id", suffixes=("_a", "_b"), how="inner")
    neuron_rows = []
    field_rows = []
    for _, row in merged.iterrows():
        nid = int(row["neuron_id"])
        ca = parse_centers(row["centers_a"])
        cb = parse_centers(row["centers_b"])
        if ca.size == 0 or cb.size == 0:
            continue

        d_a = dist_to_reward(ca, reward)
        d_b = dist_to_reward(cb, reward)
        cna, min_da, idx_a = nearest_reward_center(ca, reward)
        cnb, min_db, idx_b = nearest_reward_center(cb, reward)
        min_shift = float(min_da - min_db)
        min_proj = projection_toward_reward(cna, cnb, reward)

        centroid_a = ca.mean(axis=0)
        centroid_b = cb.mean(axis=0)
        centroid_da = float(np.linalg.norm(centroid_a - reward))
        centroid_db = float(np.linalg.norm(centroid_b - reward))
        centroid_shift = centroid_da - centroid_db
        centroid_proj = projection_toward_reward(centroid_a, centroid_b, reward)

        cost = np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=2)
        rr, cc = linear_sum_assignment(cost)
        h_shifts = []
        h_projs = []
        for ia, ib in zip(rr, cc):
            da = float(np.linalg.norm(ca[ia] - reward))
            db = float(np.linalg.norm(cb[ib] - reward))
            sh = da - db
            pr = projection_toward_reward(ca[ia], cb[ib], reward)
            h_shifts.append(sh)
            h_projs.append(pr)
            field_rows.append({
                "neuron_id": nid,
                "field_a_index": int(ia),
                "field_b_index": int(ib),
                "match_distance": float(cost[ia, ib]),
                "center_a_x": float(ca[ia, 0]),
                "center_a_y": float(ca[ia, 1]),
                "center_b_x": float(cb[ib, 0]),
                "center_b_y": float(cb[ib, 1]),
                "dist_reward_a": da,
                "dist_reward_b": db,
                "shift_toward_reward": float(sh),
                "projection_toward_reward": float(pr) if math.isfinite(pr) else np.nan,
                "k_a": int(row["k_a"]),
                "k_b": int(row["k_b"]),
                "transition": f"{int(row['k_a'])}->{int(row['k_b'])}",
            })

        neuron_rows.append({
            "neuron_id": nid,
            "k_a": int(row["k_a"]),
            "k_b": int(row["k_b"]),
            "transition": f"{int(row['k_a'])}->{int(row['k_b'])}",
            "skaggs_a": float(row["skaggs_a"]),
            "skaggs_b": float(row["skaggs_b"]),
            "delta_skaggs": float(row["skaggs_b"] - row["skaggs_a"]),
            "n_centers_a": int(ca.shape[0]),
            "n_centers_b": int(cb.shape[0]),
            "min_dist_reward_a": float(min_da),
            "min_dist_reward_b": float(min_db),
            "min_shift_toward_reward": min_shift,
            "min_projection_toward_reward": float(min_proj) if math.isfinite(min_proj) else np.nan,
            "centroid_dist_reward_a": centroid_da,
            "centroid_dist_reward_b": centroid_db,
            "centroid_shift_toward_reward": float(centroid_shift),
            "centroid_projection_toward_reward": float(centroid_proj) if math.isfinite(centroid_proj) else np.nan,
            "hungarian_mean_shift_toward_reward": float(np.nanmean(h_shifts)) if h_shifts else np.nan,
            "hungarian_mean_projection_toward_reward": float(np.nanmean(h_projs)) if h_projs else np.nan,
            "centers_a": json.dumps(ca.tolist()),
            "centers_b": json.dumps(cb.tolist()),
        })

    return pd.DataFrame(neuron_rows), pd.DataFrame(field_rows)


def one_sample_positive_tests(values: Sequence[float], n_boot: int, seed: int) -> Dict[str, object]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    out: Dict[str, object] = {"n": int(x.size)}
    if x.size == 0:
        return out
    out.update({
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "fraction_positive": float(np.mean(x > 0)),
    })
    if x.size >= 2:
        t = stats.ttest_1samp(x, popmean=0.0, alternative="greater")
        out["ttest_t"] = float(t.statistic)
        out["ttest_p_greater"] = float(t.pvalue)
        try:
            w = stats.wilcoxon(x, alternative="greater", zero_method="wilcox")
            out["wilcoxon_stat"] = float(w.statistic)
            out["wilcoxon_p_greater"] = float(w.pvalue)
        except Exception:
            out["wilcoxon_stat"] = None
            out["wilcoxon_p_greater"] = None
        pos = int(np.sum(x > 0))
        nonzero = int(np.sum(x != 0))
        if nonzero > 0:
            bt = stats.binomtest(pos, nonzero, p=0.5, alternative="greater")
            out["sign_test_positive"] = pos
            out["sign_test_nonzero"] = nonzero
            out["sign_test_p_greater"] = float(bt.pvalue)
    if n_boot and x.size > 1:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, x.size, size=(int(n_boot), x.size))
        means = x[idx].mean(axis=1)
        out["bootstrap_mean_ci95"] = [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]
    return out


def collect_all_centers(df: pd.DataFrame, only_pc: bool = True) -> np.ndarray:
    if only_pc and "is_place_cell" in df.columns:
        df = df[df["is_place_cell"]]
    centers = []
    for c in df["centers"]:
        arr = parse_centers(c)
        if arr.size:
            centers.append(arr)
    if not centers:
        return np.empty((0, 2), dtype=float)
    return np.vstack(centers)


def spatial_spread_metrics(centers: np.ndarray, reward: np.ndarray, nbins: int, near_radius: float) -> Dict[str, object]:
    out: Dict[str, object] = {"n_centers": int(len(centers))}
    if len(centers) == 0:
        return out
    d_reward = np.linalg.norm(centers - reward[None, :], axis=1)
    out.update({
        "mean_dist_to_reward": float(np.mean(d_reward)),
        "median_dist_to_reward": float(np.median(d_reward)),
        "fraction_within_near_radius": float(np.mean(d_reward <= near_radius)),
    })
    if len(centers) >= 2:
        out["mean_pairwise_center_distance"] = float(np.mean(pdist(centers)))
        # nearest neighbor distance
        distmat = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
        np.fill_diagonal(distmat, np.inf)
        out["mean_nearest_neighbor_distance"] = float(np.mean(np.min(distmat, axis=1)))
    bx = np.clip((centers[:, 0] * nbins).astype(int), 0, nbins - 1)
    by = np.clip((centers[:, 1] * nbins).astype(int), 0, nbins - 1)
    counts = np.bincount(by * nbins + bx, minlength=nbins * nbins).astype(float)
    p = counts[counts > 0] / max(counts.sum(), 1.0)
    entropy = -float(np.sum(p * np.log2(p))) if p.size else 0.0
    out["occupied_bins"] = int(np.sum(counts > 0))
    out["spatial_entropy_bits"] = entropy
    out["spatial_entropy_fraction_of_max"] = entropy / np.log2(nbins * nbins) if nbins > 1 else np.nan
    return out


def plot_hist(values: np.ndarray, title: str, xlabel: str, out_path: Path) -> None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    plt.figure(figsize=(8, 5))
    if values.size:
        plt.hist(values, bins=24, alpha=0.8)
        plt.axvline(0, linestyle="--", linewidth=1)
        plt.axvline(float(np.mean(values)), linestyle="-", linewidth=2, label=f"mean={np.mean(values):.4f}")
        plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    save_all(out_path.with_suffix(""), dpi=150)
    plt.close()


def plot_scatter_distance(df: pd.DataFrame, col_a: str, col_b: str, title: str, out_path: Path) -> None:
    plt.figure(figsize=(6, 6))
    if len(df):
        x = df[col_a].to_numpy(float)
        y = df[col_b].to_numpy(float)
        plt.scatter(x, y, alpha=0.75)
        mn = float(np.nanmin([np.nanmin(x), np.nanmin(y)]))
        mx = float(np.nanmax([np.nanmax(x), np.nanmax(y)]))
        pad = 0.02 + 0.05 * (mx - mn)
        plt.plot([mn - pad, mx + pad], [mn - pad, mx + pad], linestyle="--", linewidth=1)
        plt.xlim(mn - pad, mx + pad)
        plt.ylim(mn - pad, mx + pad)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(title)
    plt.xlabel("distance to reward, cycle A")
    plt.ylabel("distance to reward, cycle B")
    plt.tight_layout()
    save_all(out_path.with_suffix(""), dpi=150)
    plt.close()


def plot_centers_overlay(df_a: pd.DataFrame, df_b: pd.DataFrame, reward: np.ndarray, out_path: Path) -> None:
    ca = collect_all_centers(df_a, only_pc=True)
    cb = collect_all_centers(df_b, only_pc=True)
    plt.figure(figsize=(7, 7))
    if len(ca):
        plt.scatter(ca[:, 0], ca[:, 1], alpha=0.55, label="cycle A place centers")
    if len(cb):
        plt.scatter(cb[:, 0], cb[:, 1], alpha=0.55, label="cycle B place centers")
    plt.scatter([reward[0]], [reward[1]], marker="*", s=220, label="reward")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("Filtered place-field centers")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="best")
    plt.tight_layout()
    save_all(out_path.with_suffix(""), dpi=150)
    plt.close()


def plot_shift_by_transition(df: pd.DataFrame, out_path: Path, metric: str) -> None:
    plt.figure(figsize=(8, 5))
    if len(df):
        transitions = sorted(df["transition"].dropna().unique())
        vals = [df.loc[df["transition"] == tr, metric].dropna().to_numpy(float) for tr in transitions]
        if vals:
            plt.boxplot(vals, labels=transitions, showmeans=True)
            plt.axhline(0, linestyle="--", linewidth=1)
    plt.title(f"{metric} by k transition")
    plt.xlabel("k transition")
    plt.ylabel(metric)
    plt.tight_layout()
    save_all(out_path.with_suffix(""), dpi=150)
    plt.close()


def summarize_by_transition(df: pd.DataFrame, metric: str) -> List[Dict[str, object]]:
    rows = []
    if len(df) == 0:
        return rows
    for tr, sub in df.groupby("transition"):
        vals = sub[metric].dropna().to_numpy(float)
        if vals.size == 0:
            continue
        rows.append({
            "transition": str(tr),
            "n": int(vals.size),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "fraction_positive": float(np.mean(vals > 0)),
        })
    return rows


def main() -> None:
    args = parse_args()
    reward = resolve_reward_pos(args.run_dir, args.reward)

    label_a = args.state_a if args.state_a else f"cycle{args.cycle_a:02d}_{args.phase}_{args.representation}"
    label_b = args.state_b if args.state_b else f"cycle{args.cycle_b:02d}_{args.phase}_{args.representation}"

    if args.out_dir:
        out_dir = ensure_dir(Path(args.out_dir))
    elif args.run_dir:
        out_dir = ensure_dir(
            Path(args.run_dir) / "analysis" / f"filtered_place_shift_{label_a}_vs_{label_b}"
        )
    else:
        out_dir = ensure_dir(Path("filtered_place_shift_analysis"))

    cond_a = load_condition(args, args.cycle_a, label_a)
    cond_b = load_condition(args, args.cycle_b, label_b)

    df_a = analyze_condition(cond_a, args)
    df_b = analyze_condition(cond_b, args)
    df_a.to_csv(out_dir / "place_cell_metrics_cycle_a.csv", index=False)
    df_b.to_csv(out_dir / "place_cell_metrics_cycle_b.csv", index=False)

    neuron_df, field_df = paired_shift_tables(df_a, df_b, reward, args.filter_mode, args.allowed_k)
    neuron_df.to_csv(out_dir / "paired_neuron_shifts.csv", index=False)
    field_df.to_csv(out_dir / "paired_field_shifts_hungarian.csv", index=False)

    centers_a_pc = collect_all_centers(df_a, only_pc=True)
    centers_b_pc = collect_all_centers(df_b, only_pc=True)

    summary: Dict[str, object] = {
        "settings": {
            "label_a": label_a,
            "label_b": label_b,
            "cycle_a": args.cycle_a,
            "cycle_b": args.cycle_b,
            "phase": args.phase,
            "representation": args.representation,
            "reward": reward.tolist(),
            "allowed_k": [int(x) for x in args.allowed_k],
            "skaggs_threshold": float(args.skaggs_threshold),
            "skaggs_quantile": args.skaggs_quantile,
            "icl_max": args.icl_max,
            "max_gmm_bins": int(args.max_gmm_bins),
            "field_method": args.field_method,
            "min_component_bins": int(args.min_component_bins),
            "filter_mode": args.filter_mode,
            "source_requested": args.source,
        },
        "condition_a": cond_a.diagnostics,
        "condition_b": cond_b.diagnostics,
        "source_a": cond_a.source,
        "source_b": cond_b.source,
        "n_neurons_a": int(len(df_a)),
        "n_neurons_b": int(len(df_b)),
        "n_place_cells_a": int(df_a["is_place_cell"].sum()),
        "n_place_cells_b": int(df_b["is_place_cell"].sum()),
        "k_counts_a_all": {str(k): int(v) for k, v in df_a["k"].value_counts().sort_index().items()},
        "k_counts_b_all": {str(k): int(v) for k, v in df_b["k"].value_counts().sort_index().items()},
        "k_counts_a_place": {str(k): int(v) for k, v in df_a.loc[df_a["is_place_cell"], "k"].value_counts().sort_index().items()},
        "k_counts_b_place": {str(k): int(v) for k, v in df_b.loc[df_b["is_place_cell"], "k"].value_counts().sort_index().items()},
        "n_paired_neurons": int(len(neuron_df)),
        "n_paired_fields_hungarian": int(len(field_df)),
        "spread_a_place_centers": spatial_spread_metrics(centers_a_pc, reward, args.nbins, args.near_radius),
        "spread_b_place_centers": spatial_spread_metrics(centers_b_pc, reward, args.nbins, args.near_radius),
    }

    if len(neuron_df):
        summary["tests_min_to_reward"] = one_sample_positive_tests(
            neuron_df["min_shift_toward_reward"].to_numpy(float), args.bootstrap, args.seed
        )
        summary["tests_min_projection"] = one_sample_positive_tests(
            neuron_df["min_projection_toward_reward"].to_numpy(float), args.bootstrap, args.seed
        )
        summary["tests_centroid_shift"] = one_sample_positive_tests(
            neuron_df["centroid_shift_toward_reward"].to_numpy(float), args.bootstrap, args.seed
        )
        summary["tests_hungarian_mean_shift"] = one_sample_positive_tests(
            neuron_df["hungarian_mean_shift_toward_reward"].to_numpy(float), args.bootstrap, args.seed
        )
        summary["by_transition_min_shift"] = summarize_by_transition(neuron_df, "min_shift_toward_reward")
        summary["by_transition_hungarian_mean_shift"] = summarize_by_transition(neuron_df, "hungarian_mean_shift_toward_reward")
    if len(field_df):
        summary["tests_field_hungarian_shift"] = one_sample_positive_tests(
            field_df["shift_toward_reward"].to_numpy(float), args.bootstrap, args.seed
        )
        summary["tests_field_hungarian_projection"] = one_sample_positive_tests(
            field_df["projection_toward_reward"].to_numpy(float), args.bootstrap, args.seed
        )

    if cond_a.source == "png_fallback" or cond_b.source == "png_fallback":
        summary["important_warning"] = (
            "This analysis used PNG fallback for at least one condition. This is approximate and only covers "
            "the plotted top cells. It is not equivalent to a true all-neuron place-cell filter from raw activity."
        )

    with open(out_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    # Plots
    plot_centers_overlay(df_a, df_b, reward, out_dir / "filtered_centers_overlay.png")
    if len(neuron_df):
        plot_hist(
            neuron_df["min_shift_toward_reward"].to_numpy(float),
            "Neuron-level shift: nearest field to reward",
            "dist_to_reward(A nearest) - dist_to_reward(B nearest)",
            out_dir / "hist_min_to_reward_shift.png",
        )
        plot_hist(
            neuron_df["hungarian_mean_shift_toward_reward"].to_numpy(float),
            "Neuron-level shift: mean Hungarian-matched fields",
            "mean matched field shift toward reward",
            out_dir / "hist_hungarian_mean_shift.png",
        )
        plot_hist(
            neuron_df["min_projection_toward_reward"].to_numpy(float),
            "Projection of nearest-field displacement toward reward",
            "projection length toward reward",
            out_dir / "hist_min_projection_toward_reward.png",
        )
        plot_scatter_distance(
            neuron_df,
            "min_dist_reward_a",
            "min_dist_reward_b",
            "Nearest field distance to reward: A vs B",
            out_dir / "scatter_min_distance_to_reward.png",
        )
        plot_shift_by_transition(neuron_df, out_dir / "box_min_shift_by_k_transition.png", "min_shift_toward_reward")
        plot_shift_by_transition(neuron_df, out_dir / "box_hungarian_shift_by_k_transition.png", "hungarian_mean_shift_toward_reward")
    if len(field_df):
        plot_hist(
            field_df["shift_toward_reward"].to_numpy(float),
            "Field-level shift: Hungarian-matched centers",
            "dist_to_reward(A field) - dist_to_reward(B field)",
            out_dir / "hist_field_hungarian_shift.png",
        )
        plot_scatter_distance(
            field_df,
            "dist_reward_a",
            "dist_reward_b",
            "Hungarian-matched field distance to reward: A vs B",
            out_dir / "scatter_field_hungarian_distance.png",
        )

    # Human-readable report
    lines = []
    lines.append("Filtered place-cell reward-shift analysis")
    lines.append("=" * 48)
    lines.append(f"Output directory: {out_dir}")
    lines.append(f"Comparison: {label_a} vs {label_b}")
    lines.append(f"Reward: {tuple(reward.tolist())}")
    lines.append(f"Sources: A={cond_a.source}, B={cond_b.source}")
    lines.append(f"Allowed K: {args.allowed_k}; filter_mode={args.filter_mode}; Skaggs >= {args.skaggs_threshold}; field_method={args.field_method}")
    if args.skaggs_quantile is not None:
        lines.append(f"Also using Skaggs quantile threshold: {args.skaggs_quantile}")
    if args.icl_max is not None:
        lines.append(f"Also using ICL <= {args.icl_max}")
    if "important_warning" in summary:
        lines.append("")
        lines.append("WARNING: " + str(summary["important_warning"]))
    lines.append("")
    lines.append(f"N neurons A/B: {summary['n_neurons_a']} / {summary['n_neurons_b']}")
    lines.append(f"N place cells A/B: {summary['n_place_cells_a']} / {summary['n_place_cells_b']}")
    lines.append(f"K counts A place: {summary['k_counts_a_place']}")
    lines.append(f"K counts B place: {summary['k_counts_b_place']}")
    lines.append(f"N paired neurons: {summary['n_paired_neurons']}")
    lines.append(f"N paired fields (Hungarian): {summary['n_paired_fields_hungarian']}")
    lines.append("")

    def add_test(name: str, key: str) -> None:
        t = summary.get(key, {})
        if not t:
            return
        lines.append(name)
        lines.append("-" * len(name))
        lines.append(json.dumps(t, indent=2, sort_keys=True))
        lines.append("")

    add_test("Neuron-level nearest-field shift toward reward", "tests_min_to_reward")
    add_test("Neuron-level nearest-field projection toward reward", "tests_min_projection")
    add_test("Neuron-level Hungarian mean shift toward reward", "tests_hungarian_mean_shift")
    add_test("Field-level Hungarian shift toward reward", "tests_field_hungarian_shift")

    lines.append("Interpretation guide:")
    lines.append("Positive shift_toward_reward = field is closer to reward in condition B than A.")
    lines.append("Positive projection_toward_reward = displacement vector points in the reward direction.")
    lines.append("For K=2 fields, nearest-to-reward asks whether any field became more reward-proximal; Hungarian is more conservative and tracks spatially matched fields.")

    with open(out_dir / "reward_shift_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("Analysis complete.")
    print(f"Wrote: {out_dir}")
    print(f"Report: {out_dir / 'reward_shift_report.txt'}")
    if "important_warning" in summary:
        print("WARNING: " + str(summary["important_warning"]))


if __name__ == "__main__":
    main()
