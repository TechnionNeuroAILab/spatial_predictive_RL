"""Confrontation of Krishnan et al. 2022 (Nat Commun 13:6662) published Source
Data against our model's reward-prediction head.

Computes and caches to cache/krishnan_data.json:
  1. reliability      -- per-animal place-field reliability drop (rewarded ->
                         RE-low) and recovery (-> Re-Rewarded), paired Wilcoxon
                         signed-rank + paired t-tests (pairing variable: animal).
  2. da_ramp          -- per-axon VTA-DA reward-normalized fluorescence at
                         Start/Middle/End of track (mean +/- SEM), with the
                         model reward-head target exp(-d^2 / 2 sigma^2),
                         sigma = 0.15, evaluated at equivalent normalized
                         distances-to-reward for overlay.
  3. field_density    -- per-condition fraction of place fields per track
                         region vs the uniform expectation (region width /
                         track length); near-reward (end-of-track) enrichment.
  4. spatial_correlation_by_region -- what the Figure2 sheet actually provides
                         (it does NOT contain field density; see utils docstrings).

House rules honoured: every empirical number comes from MOESM4_source_data.xlsx
via utils.py; values quoted from the paper text appear only inside
'annotation_*' fields, clearly marked as annotations.

Track-orientation choice for the DA overlay (documented): the reward sits at
the END of the 200 cm track -- the sheet's rewarded-condition DA signal ramps
Start < Middle < End, and the paper describes a reward-proximity ramp rising
toward the reward zone at track end. Hence
    Start (0-50 cm)   = FAR  from reward  (normalized distance-to-reward ~0.875)
    Middle (50-150 cm) = MID                (~0.5)
    End (150-200 cm)   = NEAR reward       (~0.125)
using region centers on a track normalized to length 1. Bin-averaged model
values (mean of the kernel over each region span) are also provided, since the
data are per-region means rather than point samples.
"""

from __future__ import annotations

import datetime
import json
import os

import numpy as np
from scipy import stats

from utils import (
    REGION_SPANS_CM,
    TRACK_LENGTH_CM,
    load_da_ramp,
    load_da_ramp_sheet_ci,
    load_field_counts_by_region,
    load_field_density_profiles,
    load_pf_fraction_by_region,
    load_reliability,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(_HERE, "cache", "krishnan_data.json")

SIGMA = 0.15          # model reward-head width (normalized track units), paper/main_v3.tex
REWARD_POS = 1.0      # reward at track end (normalized); see module docstring
REGIONS = ("start", "middle", "end")


def _py(obj):
    """Recursively convert numpy scalars/arrays for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_py(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def head_target(d):
    """Model reward-prediction head target: exp(-d^2 / 2 sigma^2), sigma=0.15."""
    d = np.asarray(d, dtype=float)
    return np.exp(-(d ** 2) / (2.0 * SIGMA ** 2))


# --------------------------------------------------------------------------- #
# 1. Reliability drop / recovery (Fig 3B, n = 3 animals)                       #
# --------------------------------------------------------------------------- #
def reliability_analysis() -> dict:
    df = load_reliability()
    n = len(df)

    # Paired tests; pairing variable: ANIMAL (each row = one animal measured
    # in all three conditions).
    def paired(a, b):
        w = stats.wilcoxon(df[a], df[b], alternative="two-sided", method="exact")
        t = stats.ttest_rel(df[a], df[b])
        return {
            "wilcoxon_W": float(w.statistic),
            "wilcoxon_p": float(w.pvalue),
            "ttest_t": float(t.statistic),
            "ttest_p": float(t.pvalue),
            "n_pairs": n,
        }

    means = {c: float(df[c].mean()) for c in ("rewarded", "re_low", "re_rewarded")}
    drop = df["re_low"] - df["rewarded"]           # per-animal drop
    recov = df["re_rewarded"] - df["re_low"]       # per-animal recovery
    frac_drop_group = (means["re_low"] - means["rewarded"]) / means["rewarded"]
    # Fraction of the group-level drop recovered on re-reward:
    frac_recovered = (means["re_rewarded"] - means["re_low"]) / (
        means["rewarded"] - means["re_low"]
    )

    return {
        "source": "sheet 'Figure3', block Fig 3B 'Reliability' (per-animal)",
        "n_animals": n,
        "per_animal": {
            "animal": df["animal"].tolist(),
            "rewarded": df["rewarded"].round(6).tolist(),
            "re_low": df["re_low"].round(6).tolist(),
            "re_rewarded": df["re_rewarded"].round(6).tolist(),
            "drop_rewarded_to_re_low": drop.round(6).tolist(),
            "frac_drop": (drop / df["rewarded"]).round(4).tolist(),
            "recovery_re_low_to_re_rewarded": recov.round(6).tolist(),
        },
        "group_means": means,
        "frac_drop_group_mean": float(frac_drop_group),
        "frac_of_drop_recovered_group_mean": float(frac_recovered),
        "tests": {
            "rewarded_vs_re_low": paired("rewarded", "re_low"),
            "re_low_vs_re_rewarded": paired("re_low", "re_rewarded"),
            "rewarded_vs_re_rewarded": paired("rewarded", "re_rewarded"),
        },
        "caveat_power": (
            "Only n=3 imaged animals in the source data; the exact two-sided "
            "Wilcoxon p cannot go below 0.25 with 3 pairs. Direction is "
            "consistent in 3/3 animals for both drop and recovery."
        ),
        "annotation_expected_from_paper": (
            "ANNOTATION (not from xlsx): recon/paper text suggested group means "
            "~0.47 -> 0.32 -> 0.41 (~-32% drop); the xlsx per-animal values "
            "instead give the group means and fractional drop recorded above "
            "(~-45%). The cell-level Fig 3C means give a similar-sized drop."
        ),
    }


# --------------------------------------------------------------------------- #
# 2. DA ramp + model head-target overlay (Fig 5D, n = 7 axons)                 #
# --------------------------------------------------------------------------- #
def da_ramp_analysis() -> dict:
    df = load_da_ramp("rewarded")
    n_axons = len(df)

    mean = {r: float(df[r].mean()) for r in REGIONS}
    sem = {r: float(stats.sem(df[r])) for r in REGIONS}
    sheet_ci = load_da_ramp_sheet_ci("rewarded")

    # Paired t-test end vs start across axons; pairing variable: AXON.
    t_ramp = stats.ttest_rel(df["end"], df["start"])
    w_ramp = stats.wilcoxon(df["end"], df["start"], method="exact")

    # ---- model overlay -----------------------------------------------------
    # Region centers/edges in normalized track units; reward at REWARD_POS=1.0.
    centers, spans_norm, d_centers, target_center, target_binavg = {}, {}, {}, {}, {}
    for r in REGIONS:
        lo, hi = (e / TRACK_LENGTH_CM for e in REGION_SPANS_CM[r])
        spans_norm[r] = (lo, hi)
        centers[r] = 0.5 * (lo + hi)
        d_centers[r] = abs(REWARD_POS - centers[r])
        target_center[r] = float(head_target(d_centers[r]))
        xs = np.linspace(lo, hi, 501)
        target_binavg[r] = float(head_target(np.abs(REWARD_POS - xs)).mean())

    grid_x = np.linspace(0.0, 1.0, 101)
    grid_target = head_target(np.abs(REWARD_POS - grid_x))

    max_norm = lambda d: {r: d[r] / d["end"] for r in REGIONS}  # noqa: E731

    # Context: extinction conditions from the same sheet.
    extinction = {}
    for cond in ("re_high", "re_low"):
        dfe = load_da_ramp(cond)
        extinction[cond] = {
            "mean": {r: float(dfe[r].mean()) for r in REGIONS},
            "sem": {r: float(stats.sem(dfe[r])) for r in REGIONS},
        }

    return {
        "source": "sheet 'Figure 5', block Fig 5D 'Reward Normalized Mean F' (per-axon)",
        "n_axons": n_axons,
        "per_axon": {
            "axon": df["axon"].tolist(),
            **{r: df[r].tolist() for r in REGIONS},
        },
        "mean": mean,
        "sem": sem,
        "sheet_95ci": sheet_ci,
        "mean_max_normalized": max_norm(mean),
        "ramp_test_end_vs_start": {
            "pairing": "axon",
            "ttest_t": float(t_ramp.statistic),
            "ttest_p": float(t_ramp.pvalue),
            "wilcoxon_W": float(w_ramp.statistic),
            "wilcoxon_p": float(w_ramp.pvalue),
            "n_pairs": n_axons,
        },
        "extinction_context": extinction,
        "extinction_note": (
            "In the source data the 're_low' block has identical Middle and End "
            "values for every axon (as published). DA ramp flattens: rewarded "
            "end/start ~2.4x vs re_low end/start ~1.3x."
        ),
        "model_overlay": {
            "definition": "head target exp(-d^2/(2*sigma^2)), sigma=0.15, d = |x - x_reward| in normalized track units",
            "sigma": SIGMA,
            "reward_position_norm": REWARD_POS,
            "orientation_choice": (
                "Reward at track END: DA ramps Start<Middle<End in the rewarded "
                "condition, so Start=far (d~0.875), Middle=mid (d~0.5), "
                "End=near reward (d~0.125), using region centers of "
                "0-50 / 50-150 / 150-200 cm on a 200 cm track."
            ),
            "region_centers_norm": centers,
            "distance_to_reward_at_center": d_centers,
            "target_at_center": target_center,
            "target_bin_averaged": target_binavg,
            "target_bin_averaged_max_normalized": max_norm(target_binavg),
            "fine_grid": {
                "x_norm": grid_x.round(4).tolist(),
                "target": grid_target.round(6).tolist(),
            },
            "comparison_note": (
                "Both data and model increase monotonically toward the reward, "
                "but the sigma=0.15 head target is far sharper: bin-averaged it "
                "is <1e-4 at start and ~0.036 at middle, whereas the measured DA "
                "signal is already ~41% / ~62% of its end-of-track value there. "
                "The published DA ramp is shallow/quasi-linear over the whole "
                "track rather than a narrow near-reward bump."
            ),
        },
    }


# --------------------------------------------------------------------------- #
# 3. Field-density near-reward enrichment (Fig 3C counts + Fig 3E per animal)  #
# --------------------------------------------------------------------------- #
def field_density_analysis() -> dict:
    counts = load_field_counts_by_region()
    frac_df = load_pf_fraction_by_region()

    # Uniform expectation = region width / track length (start .25, middle .5, end .25)
    uniform = {
        r: (REGION_SPANS_CM[r][1] - REGION_SPANS_CM[r][0]) / TRACK_LENGTH_CM
        for r in REGIONS
    }

    by_cond = {}
    for cond, sub in counts.groupby("condition"):
        total = int(sub["n_fields"].sum())
        fracs = {row.region: row.n_fields / total for row in sub.itertuples()}
        # chi-square GoF of observed region counts vs uniform-by-width expectation
        obs = [int(sub.loc[sub.region == r, "n_fields"].iloc[0]) for r in REGIONS]
        exp = [uniform[r] * total for r in REGIONS]
        chi2 = stats.chisquare(obs, exp)
        by_cond[cond] = {
            "n_fields_total": total,
            "n_fields": {r: int(sub.loc[sub.region == r, "n_fields"].iloc[0]) for r in REGIONS},
            "fraction": {r: float(fracs[r]) for r in REGIONS},
            "enrichment_vs_uniform": {r: float(fracs[r] / uniform[r]) for r in REGIONS},
            "chi2_vs_uniform": {"chi2": float(chi2.statistic), "p": float(chi2.pvalue)},
        }

    # Per-animal end-of-track (reward-zone) fraction; paired tests, pairing: ANIMAL.
    piv = frac_df.pivot(index="animal", columns="condition", values="pct_end") / 100.0
    piv = piv[["rewarded", "re_low", "re_rewarded"]]

    def paired(a, b):
        w = stats.wilcoxon(piv[a], piv[b], method="exact")
        t = stats.ttest_rel(piv[a], piv[b])
        return {
            "wilcoxon_W": float(w.statistic),
            "wilcoxon_p": float(w.pvalue),
            "ttest_t": float(t.statistic),
            "ttest_p": float(t.pvalue),
            "n_pairs": int(len(piv)),
        }

    return {
        "source": (
            "counts: sheet 'Figure3' Fig 3C sample sizes (per-condition place-field "
            "counts by region); per-animal: Fig 3E '% placefields' (n=3 animals)"
        ),
        "reward_zone": "end of track (150-200 cm)",
        "uniform_expectation": uniform,
        "by_condition_counts": by_cond,
        "per_animal_end_fraction": {
            "animal": piv.index.tolist(),
            "rewarded": piv["rewarded"].round(4).tolist(),
            "re_low": piv["re_low"].round(4).tolist(),
            "re_rewarded": piv["re_rewarded"].round(4).tolist(),
            "group_mean": {c: float(piv[c].mean()) for c in piv.columns},
            "tests": {
                "rewarded_vs_re_low": paired("rewarded", "re_low"),
                "re_low_vs_re_rewarded": paired("re_low", "re_rewarded"),
            },
        },
        "figure2_note": (
            "The Figure2 sheet does NOT contain field density: its per-region "
            "blocks (Fig 2D) are spatial-correlation values for the SAME tracked "
            "cells in every condition (n fixed at 133/262/207). Field density "
            "had to be taken from Figure3 (Fig 3C counts, Fig 3E per-animal %)."
        ),
    }


# --------------------------------------------------------------------------- #
# 4. What Figure2 actually provides                                            #
# --------------------------------------------------------------------------- #
def spatial_correlation_analysis() -> dict:
    df = load_field_density_profiles()  # see its docstring: spatial corr, not density
    out = {}
    for cond, sub in df.groupby("condition"):
        out[cond] = {
            row.region: {"n_cells": int(row.n_cells), "mean": row.mean_spatial_corr, "ci95": row.ci95}
            for row in sub.itertuples()
        }
    return {
        "source": "sheet 'Figure2', block Fig 2D (spatial correlation by track region)",
        "measure": "spatial correlation of tracked cells vs rewarded-session template",
        "pairing_note": "same cells across conditions (paired design), n fixed per region",
        "by_condition": out,
    }


def main() -> dict:
    results = {
        "meta": {
            "generated": datetime.date.today().isoformat(),
            "script": "analysis.py",
            "source_file": "MOESM4_source_data.xlsx",
            "source_sha256": "52a8168e6c603bb31d116c9730cb8481837d2c6efc5e114ba57db63995e5946b",
            "paper_doi": "10.1038/s41467-022-34465-5",
            "tier": (
                "Tier-1 confrontation: PUBLISHED SOURCE DATA only; raw data were "
                "not reanalysed (available on request, sheffield@uchicago.edu)."
            ),
        },
        "reliability": reliability_analysis(),
        "da_ramp": da_ramp_analysis(),
        "field_density": field_density_analysis(),
        "spatial_correlation_by_region": spatial_correlation_analysis(),
    }

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(_py(results), f, indent=2)

    # ------------------------- printed summary ----------------------------- #
    rel, da, fd = results["reliability"], results["da_ramp"], results["field_density"]
    print(f"cached -> {CACHE_PATH}\n")
    print(f"RELIABILITY (Fig 3B, n={rel['n_animals']} animals)")
    m = rel["group_means"]
    print(
        f"  group means: rewarded {m['rewarded']:.3f} -> RE-low {m['re_low']:.3f} "
        f"-> re-rewarded {m['re_rewarded']:.3f}"
    )
    print(
        f"  fractional drop (group mean): {rel['frac_drop_group_mean']:+.1%}; "
        f"fraction of drop recovered: {rel['frac_of_drop_recovered_group_mean']:.1%}"
    )
    for k, t in rel["tests"].items():
        print(
            f"  {k}: Wilcoxon W={t['wilcoxon_W']:.0f}, p={t['wilcoxon_p']:.3f}; "
            f"paired t={t['ttest_t']:+.2f}, p={t['ttest_p']:.3f} (n={t['n_pairs']})"
        )
    print(f"  ({rel['caveat_power']})\n")

    print(f"DA RAMP (Fig 5D, n={da['n_axons']} axons, rewarded condition)")
    for r in REGIONS:
        print(
            f"  {r:6s}: mean {da['mean'][r]:.3f} +/- {da['sem'][r]:.3f} SEM "
            f"(sheet 95% CI {da['sheet_95ci'][r]:.3f}); model target "
            f"(bin-avg) {da['model_overlay']['target_bin_averaged'][r]:.4f}"
        )
    t = da["ramp_test_end_vs_start"]
    print(
        f"  end vs start: paired t={t['ttest_t']:+.2f}, p={t['ttest_p']:.4f}; "
        f"Wilcoxon p={t['wilcoxon_p']:.4f} (pairing: axon)"
    )
    print(f"  {da['model_overlay']['comparison_note']}\n")

    print("FIELD DENSITY (Fig 3C counts / Fig 3E per-animal), reward zone = end")
    for cond in ("rewarded", "re_low", "re_rewarded"):
        c = fd["by_condition_counts"][cond]
        print(
            f"  {cond:11s}: end fraction {c['fraction']['end']:.3f} "
            f"(uniform 0.25, enrichment x{c['enrichment_vs_uniform']['end']:.2f}, "
            f"chi2 p={c['chi2_vs_uniform']['p']:.2e}, n={c['n_fields_total']})"
        )
    pa = fd["per_animal_end_fraction"]
    print(
        f"  per-animal end-fraction means: rewarded {pa['group_mean']['rewarded']:.3f}, "
        f"re_low {pa['group_mean']['re_low']:.3f}, re_rewarded {pa['group_mean']['re_rewarded']:.3f}"
    )
    for k, t in pa["tests"].items():
        print(f"  {k}: Wilcoxon p={t['wilcoxon_p']:.3f} (n={t['n_pairs']}, pairing: animal)")
    print(f"\n  NOTE: {fd['figure2_note']}")
    return results


if __name__ == "__main__":
    main()
