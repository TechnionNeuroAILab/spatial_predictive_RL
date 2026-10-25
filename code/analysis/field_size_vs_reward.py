#!/usr/bin/env python3
"""Are place fields closer to the reward smaller or larger?

Standalone companion to the reward-shift analyses. It does not write into the
paper. Field size is measured on the same rate maps as the shift statistics
(occupancy-normalized, goal-directed reset transients dropped, Gaussian
smoothing σ = 1.5 bins).

Size. Equivalent radius of the 8-connected component that contains the peak
and stays at or above half the smoothed map's maximum:

    r = sqrt(A / π),   A = (bins in the component) / 32²

in arena units (the arena is the unit square). Two checks use the same peak:
the component above 20% of the maximum, and the activity-weighted RMS radius
of the whole smoothed map. A sharper smooth (σ = 0.75) repeats the half-max
radius, because σ = 1.5 puts a floor under small fields.

Population. The 55 networks retained for the paired cohort. Place cells are
units with Skaggs information ≥ 0.3 in the period being described. Reward is
at (0.3, 0.3). The fixed-trajectory pair is replay_c1_pred → replay_c1_postrl
(same random walk, weights change). cycle_1_goal is the goal-directed probe.

Question the numbers answer. Within each network, split place cells into the
closer and farther halves of peak-to-reward distance. The network's statistic
is mean radius(closer) − mean radius(farther). Negative means the fields
closer to reward are smaller. The test is a two-sided Wilcoxon signed-rank
test across networks. The same contrast before learning, and a regression of
radius on distance to reward that also includes distance to the nearest wall,
ask whether a gradient is reward learning or the arena boundary.

Usage:
    python code/analysis/field_size_vs_reward.py [--workers 32]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, label
from scipy.stats import spearmanr, wilcoxon

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code" / "lib"))

from analysis_helpers import (  # noqa: E402
    SKAGGS_THRESHOLD,
    campaign_passers,
    load_period_centers_cached,
    load_period_frames,
)
from ratemaps import build_rate_maps, compute_skaggs  # noqa: E402

RES = REPO / "results"
CACHE = RES / "aggregate" / "field_size_cache_v1"
OUT_JSON = RES / "aggregate" / "field_size_vs_reward.json"
OUT_PDF = RES / "aggregate" / "field_size_vs_reward.pdf"

REWARD = np.array([0.3, 0.3])
OPPOSITE = np.array([0.7, 0.7])
M = 32
SIGMA = 1.5
SIGMA_SHARP = 0.75
HIDDEN = 512
MIN_CELLS = 20
BIN_MIN = 5
EDGES = np.array([0.0, 0.20, 0.35, 0.50, 0.65, 1.05])
BIN_LABELS = ["0–0.20", "0.20–0.35", "0.35–0.50", "0.50–0.65", "0.65–1.05"]
STRUCT = np.ones((3, 3), dtype=int)
MEASURES = ("r50", "r20", "rrms", "r50_sharp")

ARMS = {
    "nohead": ("multi_seed", "nohead_seed"),
    "head": ("multi_seed_head", "head_seed"),
    "rigid": ("multi_seed_rigid", "rigid_seed"),
}
PERIODS = ("replay_c1_pred", "replay_c1_postrl", "cycle_1_goal")
FROZEN = ("replay_c1_pred", "replay_c1_postrl")


def _component(sm, ix, iy, thr, visited):
    """Equivalent radius and fraction of the component that was visited."""
    lab, _ = label(sm >= thr, structure=STRUCT)
    comp = lab == lab[ix, iy]
    nb = int(comp.sum())
    if nb <= 0:
        return np.nan, np.nan
    radius = float(np.sqrt((nb / float(M * M)) / np.pi))
    support = float(np.count_nonzero(comp & visited) / nb)
    return radius, support


def _sizes_one_sigma(rm, visited, sigma):
    """Peak (x, y), half-max radius, 20% radius, RMS radius, half-max support."""
    n = rm.shape[0]
    sm = gaussian_filter(rm, sigma=(0.0, sigma, sigma))
    flat = sm.reshape(n, -1)
    ix, iy = np.unravel_index(flat.argmax(1), (M, M))
    peak = np.column_stack((ix / M, iy / M))
    mx = flat.max(1)
    r50 = np.full(n, np.nan)
    r20 = np.full(n, np.nan)
    support = np.full(n, np.nan)
    for i in range(n):
        if mx[i] <= 0:
            continue
        r50[i], support[i] = _component(sm[i], ix[i], iy[i], 0.5 * mx[i], visited)
        r20[i], _ = _component(sm[i], ix[i], iy[i], 0.2 * mx[i], visited)
    xs = np.arange(M) / M
    total = sm.sum(axis=(1, 2))
    ok = total > 0
    cx = np.full(n, np.nan)
    cy = np.full(n, np.nan)
    cx[ok] = (sm[ok] * xs[None, :, None]).sum(axis=(1, 2)) / total[ok]
    cy[ok] = (sm[ok] * xs[None, None, :]).sum(axis=(1, 2)) / total[ok]
    dx = xs[None, :, None] - cx[:, None, None]
    dy = xs[None, None, :] - cy[:, None, None]
    var = np.full(n, np.nan)
    var[ok] = (sm[ok] * (dx[ok] ** 2 + dy[ok] ** 2)).sum(axis=(1, 2)) / total[ok]
    rrms = np.sqrt(var)
    return peak, r50, r20, rrms, support


def measure_period(data_dir: str, period: str) -> dict:
    beh, act = load_period_frames(data_dir, period, HIDDEN)
    rm, occ = build_rate_maps(beh, act, M, HIDDEN)
    sk = compute_skaggs(rm, occ)
    visited = occ > 0
    peak, r50, r20, rrms, support = _sizes_one_sigma(rm, visited, SIGMA)
    _, r50_s, _, _, _ = _sizes_one_sigma(rm, visited, SIGMA_SHARP)
    return {
        "sk": sk.astype(np.float64),
        "peak": peak.astype(np.float64),
        "r50": r50.astype(np.float64),
        "r20": r20.astype(np.float64),
        "rrms": rrms.astype(np.float64),
        "r50_sharp": r50_s.astype(np.float64),
        "support": support.astype(np.float64),
    }


def cache_path(arm: str, seed: int, period: str) -> Path:
    return CACHE / arm / f"seed{seed}" / f"{period}.npz"


def data_dir(arm: str, seed: int) -> Path:
    family, prefix = ARMS[arm]
    return RES / family / f"{prefix}{seed}" / "data"


def build_job(job: tuple[str, int, str]) -> str:
    arm, seed, period = job
    d = data_dir(arm, seed)
    beh = d / f"df_activations_{period}.parquet"
    path = cache_path(arm, seed, period)
    if not beh.exists():
        return f"missing {arm} {seed} {period}"
    mtime = beh.stat().st_mtime
    if path.exists():
        with np.load(path) as z:
            if float(z["mtime"]) == mtime and int(z["version"]) == 1:
                return "cached"
    try:
        measured = measure_period(str(d), period)
    except Exception as exc:  # noqa: BLE001 — one bad run should not kill the cohort
        return f"fail {arm} {seed} {period}: {type(exc).__name__}: {exc}"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, mtime=mtime, version=np.int32(1), **measured)
    os.replace(tmp, path)
    return "ok"


def load_cached(arm: str, seed: int, period: str) -> dict | None:
    path = cache_path(arm, seed, period)
    if not path.exists():
        return None
    z = np.load(path)
    return {k: z[k] for k in ("sk", "peak", "r50", "r20", "rrms", "r50_sharp", "support")}


def _self_check() -> None:
    """An impulse smoothed at σ = 1.5 must come out near the kernel width."""
    rm = np.zeros((1, M, M))
    rm[0, M // 2, M // 2] = 1.0
    visited = np.ones((M, M), dtype=bool)
    _, r50, r20, rrms, _ = _sizes_one_sigma(rm, visited, SIGMA)
    # Continuous half-max radius of a Gaussian is σ√(2 ln 2) bins.
    expected = SIGMA * np.sqrt(2.0 * np.log(2.0)) / M
    if not (0.5 * expected < r50[0] < 2.0 * expected):
        raise RuntimeError(f"impulse half-max radius {r50[0]:.4f}, expected ~{expected:.4f}")
    if not (r20[0] > r50[0] and rrms[0] > 0):
        raise RuntimeError("size measures are not ordered as expected on an impulse")


def _validate_against_centers(seed: int = 0) -> None:
    """Peaks and Skaggs must match the maps used for the published shifts."""
    period = "replay_c1_pred"
    d = str(data_dir("nohead", seed))
    sk, _, peak = load_period_centers_cached(d, period)
    z = load_cached("nohead", seed, period)
    if z is None:
        raise RuntimeError("validation cache missing")
    if not np.allclose(z["sk"], sk):
        raise RuntimeError(f"Skaggs mismatch, max {np.max(np.abs(z['sk'] - sk)):.3e}")
    if not np.allclose(z["peak"], peak):
        raise RuntimeError(f"peak mismatch, max {np.max(np.abs(z['peak'] - peak)):.3e}")


def dist_reward(peak: np.ndarray, point: np.ndarray) -> np.ndarray:
    return np.linalg.norm(peak - point[None, :], axis=1)


def dist_wall(peak: np.ndarray) -> np.ndarray:
    x, y = peak[:, 0], peak[:, 1]
    return np.minimum(np.minimum(x, y), np.minimum(1.0 - x, 1.0 - y))


def _finite(*cols):
    mask = np.ones(len(cols[0]), dtype=bool)
    for c in cols:
        mask &= np.isfinite(c)
    return mask


def half_split(size: np.ndarray, dist: np.ndarray) -> dict | None:
    """Closer half minus farther half. Negative: closer fields are smaller."""
    mask = _finite(size, dist)
    size, dist = size[mask], dist[mask]
    if len(size) < MIN_CELLS:
        return None
    order = np.argsort(dist, kind="mergesort")
    half = len(order) // 2
    near_i, far_i = order[:half], order[-half:]
    near, far = size[near_i], size[far_i]
    diff = float(near.mean() - far.mean())
    return {
        "diff": diff,
        "near": float(near.mean()),
        "far": float(far.mean()),
        "pct": float(100.0 * diff / far.mean()) if far.mean() != 0 else np.nan,
        "median_dist": float(np.median(dist)),
        "n": int(len(size)),
        "mean_size": float(size.mean()),
    }


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = _finite(x, y)
    if mask.sum() < MIN_CELLS:
        return np.nan
    if np.unique(x[mask]).size < 2 or np.unique(y[mask]).size < 2:
        return np.nan
    r, _ = spearmanr(x[mask], y[mask])
    return float(r)


def _residual(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    X = np.column_stack((np.ones(len(x)), x))
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ coef


def partial_spearman(size, d_rew, d_wall) -> float:
    """Rank correlation of size with distance to reward after removing wall distance."""
    mask = _finite(size, d_rew, d_wall)
    if mask.sum() < MIN_CELLS:
        return np.nan
    s, r, w = size[mask], d_rew[mask], d_wall[mask]
    if np.unique(w).size < 2:
        return _spearman(s, r)
    return _spearman(_residual(s, w), _residual(r, w))


def wall_slope(size, d_rew, d_wall) -> float:
    """Linear coefficient of distance-to-reward in size ~ reward + wall.

    Positive: fields farther from reward are larger, so closer fields are smaller.
    """
    mask = _finite(size, d_rew, d_wall)
    if mask.sum() < MIN_CELLS:
        return np.nan
    y = size[mask]
    X = np.column_stack((np.ones(mask.sum()), d_rew[mask], d_wall[mask]))
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(coef[1])


def bin_means(size: np.ndarray, dist: np.ndarray) -> list[float]:
    mask = _finite(size, dist)
    size, dist = size[mask], dist[mask]
    idx = np.digitize(dist, EDGES) - 1
    out = []
    for b in range(len(EDGES) - 1):
        sel = idx == b
        out.append(float(size[sel].mean()) if sel.sum() >= BIN_MIN else np.nan)
    return out


def cross_section(rec: dict, measure: str, support_min: float | None = None) -> dict | None:
    place = rec["sk"] >= SKAGGS_THRESHOLD
    if support_min is not None:
        place = place & (rec["support"] >= support_min)
    if place.sum() < MIN_CELLS:
        return None
    size = rec[measure][place]
    peak = rec["peak"][place]
    d_r = dist_reward(peak, REWARD)
    d_w = dist_wall(peak)
    d_o = dist_reward(peak, OPPOSITE)
    split = half_split(size, d_r)
    if split is None:
        return None
    opp = half_split(size, d_o)
    return {
        **split,
        "spearman": _spearman(size, d_r),
        "partial_spearman": partial_spearman(size, d_r, d_w),
        "wall_slope": wall_slope(size, d_r, d_w),
        "opp_diff": None if opp is None else opp["diff"],
        "bins": bin_means(size, d_r),
        "support": float(np.nanmean(rec["support"][place])),
        "frac_supported": float(np.mean(rec["support"][place] >= 0.95)),
    }


def paired(pre: dict, post: dict, measure: str) -> dict | None:
    """Same cells before and after, on one trajectory."""
    gate = (pre["sk"] >= SKAGGS_THRESHOLD) & (post["sk"] >= SKAGGS_THRESHOLD)
    if gate.sum() < MIN_CELLS:
        return None
    s0, s1 = pre[measure][gate], post[measure][gate]
    d0 = dist_reward(pre["peak"][gate], REWARD)
    d1 = dist_reward(post["peak"][gate], REWARD)
    mask = _finite(s0, s1, d0, d1)
    s0, s1, d0, d1 = s0[mask], s1[mask], d0[mask], d1[mask]
    if len(s0) < MIN_CELLS:
        return None
    dd, ds = d1 - d0, s1 - s0
    toward, away = dd < 0, dd > 0
    # Cells that finish in the closer vs farther half: were they already small?
    order = np.argsort(d1, kind="mergesort")
    half = len(order) // 2
    near, far = order[:half], order[-half:]
    return {
        "n": int(len(s0)),
        "mean_dsize": float(ds.mean()),
        "dsize_toward": float(ds[toward].mean()) if toward.sum() >= 5 else np.nan,
        "dsize_away": float(ds[away].mean()) if away.sum() >= 5 else np.nan,
        "frac_toward": float(toward.mean()),
        "spearman_dd_ds": _spearman(dd, ds),
        "spearman_presize_dd": _spearman(s0, dd),
        "pre_size_of_post_near": float(s0[near].mean()),
        "pre_size_of_post_far": float(s0[far].mean()),
        "post_size_near": float(s1[near].mean()),
        "post_size_far": float(s1[far].mean()),
        "inherited_diff": float(s0[near].mean() - s0[far].mean()),
        "post_diff": float(s1[near].mean() - s1[far].mean()),
    }


def summarize(values: dict[int, float]) -> dict | None:
    seeds = sorted(values)
    v = np.array([values[s] for s in seeds], dtype=float)
    ok = np.isfinite(v)
    seeds = [s for s, keep in zip(seeds, ok) if keep]
    v = v[ok]
    if len(v) < 5:
        return None
    sd = float(v.std(ddof=1))
    p = None
    if not np.allclose(v, 0):
        p = float(wilcoxon(v, alternative="two-sided", zero_method="wilcox").pvalue)
    return {
        "n": int(len(v)),
        "mean": float(v.mean()),
        "sd": sd,
        "sem": float(sd / np.sqrt(len(v))),
        "median": float(np.median(v)),
        "n_neg": int((v < 0).sum()),
        "n_pos": int((v > 0).sum()),
        "p": p,
        "seeds": seeds,
        "values": [float(x) for x in v],
    }


def _column(rows: dict[int, dict], key: str) -> dict[int, float]:
    return {s: r[key] for s, r in rows.items() if r is not None and r.get(key) is not None}


def _paired_delta(a: dict[int, float], b: dict[int, float]) -> dict[int, float]:
    return {s: b[s] - a[s] for s in a.keys() & b.keys()}


def _mean_bins(rows: dict[int, dict]) -> dict:
    mat = [r["bins"] for r in rows.values() if r is not None]
    n_bins = len(EDGES) - 1
    if not mat:
        return {"mean": [None] * n_bins, "sem": [None] * n_bins, "n": [0] * n_bins}
    arr = np.asarray(mat, dtype=float)
    mean, sem, n = [], [], []
    for b in range(arr.shape[1]):
        col = arr[:, b]
        col = col[np.isfinite(col)]
        n.append(int(len(col)))
        mean.append(float(col.mean()) if len(col) else None)
        sem.append(float(col.std(ddof=1) / np.sqrt(len(col))) if len(col) >= 2 else None)
    return {"mean": mean, "sem": sem, "n": n}


def _pack_cross(rows: dict[int, dict]) -> dict:
    keys = (
        "diff", "near", "far", "pct", "median_dist", "n", "mean_size",
        "spearman", "partial_spearman", "wall_slope", "opp_diff",
        "support", "frac_supported",
    )
    out = {k: summarize(_column(rows, k)) for k in keys}
    out["bins"] = _mean_bins(rows)
    return out


def _drop_values(summary: dict | None) -> dict | None:
    """JSON keeps the summary; per-seed lists stay only where plots need them."""
    if summary is None:
        return None
    keep = {k: summary[k] for k in ("n", "mean", "sd", "sem", "median", "n_neg", "n_pos", "p")}
    return keep


def aggregate(seeds: list[int]) -> dict:
    cross: dict[str, dict[str, dict]] = {}
    supported: dict[str, dict[str, dict]] = {}
    pair: dict[str, dict] = {}
    # per-seed diff series kept for the figure and for arm contrasts
    diff_series: dict[str, dict[int, float]] = {}
    for arm in ARMS:
        cross[arm] = {}
        for period in PERIODS:
            loaded = {}
            for seed in seeds:
                rec = load_cached(arm, seed, period)
                if rec is None:
                    continue
                loaded[seed] = rec
            by_measure = {}
            for measure in MEASURES:
                rows = {s: cross_section(rec, measure) for s, rec in loaded.items()}
                by_measure[measure] = _pack_cross(rows)
                if measure == "r50":
                    diff_series[f"{arm}|{period}"] = _column(rows, "diff")
                    rows_hi = {
                        s: cross_section(rec, measure, support_min=0.95)
                        for s, rec in loaded.items()
                    }
                    supported.setdefault(arm, {})[period] = _pack_cross(rows_hi)
                    diff_series[f"{arm}|{period}|supported"] = _column(rows_hi, "diff")
            cross[arm][period] = by_measure
        pair[arm] = {}
        for measure in MEASURES:
            rows = {}
            for seed in seeds:
                pre = load_cached(arm, seed, FROZEN[0])
                post = load_cached(arm, seed, FROZEN[1])
                if pre is None or post is None:
                    continue
                rows[seed] = paired(pre, post, measure)
            keys = (
                "n", "mean_dsize", "dsize_toward", "dsize_away", "frac_toward",
                "spearman_dd_ds", "spearman_presize_dd",
                "pre_size_of_post_near", "pre_size_of_post_far",
                "post_size_near", "post_size_far", "inherited_diff", "post_diff",
            )
            pair[arm][measure] = {k: summarize(_column(rows, k)) for k in keys}
            if measure == "r50":
                diff_series[f"{arm}|paired_post"] = _column(rows, "post_diff")
                diff_series[f"{arm}|paired_inherited"] = _column(rows, "inherited_diff")
                diff_series[f"{arm}|dsize_toward"] = _column(rows, "dsize_toward")

    contrasts = {}
    for name, left, right in (
        ("nohead_post_minus_pre", "nohead|replay_c1_pred", "nohead|replay_c1_postrl"),
        ("head_post_minus_pre", "head|replay_c1_pred", "head|replay_c1_postrl"),
        ("head_minus_nohead_post", "nohead|replay_c1_postrl", "head|replay_c1_postrl"),
        ("nohead_minus_rigid_post", "rigid|replay_c1_postrl", "nohead|replay_c1_postrl"),
        ("head_minus_rigid_post", "rigid|replay_c1_postrl", "head|replay_c1_postrl"),
        ("nohead_goal_minus_replay", "nohead|replay_c1_postrl", "nohead|cycle_1_goal"),
        ("head_goal_minus_replay", "head|replay_c1_postrl", "head|cycle_1_goal"),
    ):
        contrasts[name] = summarize(_paired_delta(diff_series[left], diff_series[right]))

    return {
        "definition": {
            "cohort": "paired",
            "n_seeds_requested": len(seeds),
            "skaggs": SKAGGS_THRESHOLD,
            "reward": [0.3, 0.3],
            "sigma_bins": SIGMA,
            "sigma_sharp_bins": SIGMA_SHARP,
            "grid": M,
            "size": "equivalent radius of the half-max connected component, arena units",
            "split": "closer half minus farther half of peak distance; negative => closer fields smaller",
            "test": "two-sided Wilcoxon signed-rank across networks",
            "bin_edges": EDGES.tolist(),
            "bin_labels": BIN_LABELS,
            "frozen": list(FROZEN),
            "goal": "cycle_1_goal",
        },
        "cross": cross,
        "supported_r50": supported,
        "paired": pair,
        "contrasts_r50_diff": contrasts,
        "diff_series_r50": {k: {"seeds": list(v.keys()), "values": list(v.values())} for k, v in diff_series.items()},
    }


def _fmt_p(p) -> str:
    if p is None:
        return "NA"
    return f"{p:.2e}" if p < 1e-3 else f"{p:.3f}"


def _line(label: str, summary: dict | None, unit: str = "") -> str:
    if summary is None:
        return f"{label}: insufficient"
    sign = "closer smaller" if summary["mean"] < 0 else "closer larger"
    extra = f"  [{sign}]" if unit == "diff" else ""
    return (
        f"{label}: mean={summary['mean']:+.5f}{unit if unit != 'diff' else ''}  "
        f"sd={summary['sd']:.5f}  sem={summary['sem']:.5f}  "
        f"neg {summary['n_neg']}/{summary['n']}  p={_fmt_p(summary['p'])}{extra}"
    )


def print_report(result: dict) -> None:
    print("\n=== Place-field size vs distance to reward ===")
    print("diff = radius(closer half) − radius(farther half), arena units. Negative: closer fields are smaller.")
    print("Two-sided Wilcoxon across networks. Place cells: Skaggs ≥ 0.3.\n")
    for arm in ("nohead", "head", "rigid"):
        print(f"--- {arm} ---")
        for period, tag in (
            ("replay_c1_pred", "frozen before learning"),
            ("replay_c1_postrl", "frozen after learning"),
            ("cycle_1_goal", "goal-directed after learning"),
        ):
            block = result["cross"][arm][period]["r50"]
            print(_line(f"  {tag:32s} r50 diff", block["diff"], "diff"))
            print(_line(f"  {'':32s} radius near", block["near"]))
            print(_line(f"  {'':32s} radius far ", block["far"]))
            print(_line(f"  {'':32s} percent    ", block["pct"], "%"))
            print(_line(f"  {'':32s} Spearman r ", block["spearman"]))
            print(_line(f"  {'':32s} partial r  ", block["partial_spearman"]))
            print(_line(f"  {'':32s} wall slope ", block["wall_slope"]))
            print(_line(f"  {'':32s} opposite Δ ", block["opp_diff"], "diff"))
            sup = block["support"]
            frac = block["frac_supported"]
            if sup and frac:
                print(
                    f"  {'':32s} support    mean fraction of half-max visited={sup['mean']:.3f}  "
                    f"fraction of cells ≥0.95={frac['mean']:.3f}"
                )
            for measure in ("r20", "rrms", "r50_sharp"):
                print(_line(f"  {'':32s} {measure:11s}", result["cross"][arm][period][measure]["diff"], "diff"))
        print("  within cell, frozen trajectory, same place cells")
        pr = result["paired"][arm]["r50"]
        print(_line("    Δradius | moved toward", pr["dsize_toward"]))
        print(_line("    Δradius | moved away  ", pr["dsize_away"]))
        print(_line("    Spearman(Δd, Δradius) ", pr["spearman_dd_ds"]))
        print(_line("    Spearman(pre size, Δd)", pr["spearman_presize_dd"]))
        print(_line("    pre size of post-near − post-far", pr["inherited_diff"], "diff"))
        print(_line("    post size of post-near − post-far", pr["post_diff"], "diff"))
    print("\n--- change in the closer−farther gap ---")
    print("  Value is the second condition minus the first, as named. Positive: closer fields became relatively larger.")
    for name, summary in result["contrasts_r50_diff"].items():
        print(_line(f"  {name}", summary))
    print("\n--- half-max region ≥95% visited (drops fields that run into the unsampled border) ---")
    for arm in ("nohead", "head", "rigid"):
        for period, tag in (
            ("replay_c1_pred", "before"),
            ("replay_c1_postrl", "after"),
            ("cycle_1_goal", "goal"),
        ):
            block = result["supported_r50"][arm][period]["diff"]
            ncells = result["supported_r50"][arm][period]["n"]
            n_txt = f"  cells/net={ncells['mean']:.0f}" if ncells else ""
            print(_line(f"  {arm:6} {tag:6}", block, "diff") + n_txt)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items() if k != "values" or True}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def plot(result: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = [
        ("nohead", "replay_c1_pred", "No proximity, before", "#8aa4c4", "-"),
        ("nohead", "replay_c1_postrl", "No proximity, after", "#1f4e79", "-"),
        ("head", "replay_c1_postrl", "Proximity teaching, after", "#d4762c", "-"),
        ("rigid", "replay_c1_postrl", "Near-rigid weights, after", "#6e6e6e", "--"),
    ]
    centers = [0.10, 0.275, 0.425, 0.575, 0.85]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))

    ax = axes[0]
    for arm, period, label, color, ls in series:
        bins = result["cross"][arm][period]["r50"]["bins"]
        y = np.array([np.nan if v is None else v for v in bins["mean"]], float)
        e = np.array([0 if v is None else v for v in bins["sem"]], float)
        ax.errorbar(centers, y, yerr=e, color=color, ls=ls, marker="o", ms=3.5,
                    lw=1.2, capsize=2, label=label)
    ax.set_xlabel("Distance of field peak to reward")
    ax.set_ylabel("Half-max radius (arena units)")
    ax.set_title("Fixed trajectory", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=7)
    ax.set_xlim(0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    groups = [
        ("nohead|replay_c1_pred", "No prox.\nbefore", "#8aa4c4"),
        ("nohead|replay_c1_postrl", "No prox.\nafter", "#1f4e79"),
        ("head|replay_c1_pred", "Proximity\nbefore", "#f0c08a"),
        ("head|replay_c1_postrl", "Proximity\nafter", "#d4762c"),
        ("rigid|replay_c1_postrl", "Rigid\nafter", "#6e6e6e"),
    ]
    rng = np.random.default_rng(0)
    for i, (key, label, color) in enumerate(groups):
        vals = np.array(result["diff_series_r50"][key]["values"], float)
        jitter = rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=8, color=color, alpha=0.45, zorder=2)
        ax.plot([i - 0.18, i + 0.18], [vals.mean(), vals.mean()], color="black", lw=1.4, zorder=3)
    ax.axhline(0, color="#b0b0b0", lw=0.8, zorder=1)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[1] for g in groups], fontsize=7.5)
    ax.set_ylabel("Closer − farther radius")
    ax.set_title("One point per network", loc="left", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PDF.with_suffix(".png"), dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seeds", type=str, default="", help="comma-separated seeds; default is the paired cohort")
    args = ap.parse_args()
    _self_check()
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s]
    else:
        seeds = sorted(campaign_passers("paired"))
    jobs = [(arm, seed, period) for arm in ARMS for seed in seeds for period in PERIODS]
    print(f"{len(seeds)} networks × {len(ARMS)} arms × {len(PERIODS)} periods = {len(jobs)} maps", flush=True)
    if args.workers <= 1:
        notes = [build_job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(args.workers) as ex:
            notes = list(ex.map(build_job, jobs, chunksize=1))
    fails = [n for n in notes if n.startswith("fail") or n.startswith("missing")]
    print(f"built {notes.count('ok')}  cached {notes.count('cached')}  problems {len(fails)}", flush=True)
    for n in fails[:20]:
        print(n)
    if 0 in seeds or seeds == [0]:
        _validate_against_centers(0 if 0 in seeds else seeds[0])
        print("peaks and Skaggs match the published rate-map pipeline")
    elif seeds:
        _validate_against_centers(seeds[0])
        print(f"peaks and Skaggs match on seed {seeds[0]}")
    result = aggregate(seeds)
    print_report(result)
    OUT_JSON.write_text(json.dumps(_jsonable(result), indent=1))
    print(f"\nwrote {OUT_JSON}")
    plot(result)
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
