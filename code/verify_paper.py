#!/usr/bin/env python3
"""Fail when the manuscript drifts from the released code or result caches.

This is a fast publication gate, not a replacement for recomputing the
analyses.  It checks the manuscript's figure allowlist, canonical exemplar
runs, cohort definitions, principal Methods constants, and every numerical
claim in the main Results that has a compact machine-readable source.

Run from anywhere:
    python code/verify_paper.py
    python code/verify_paper.py --json
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
REPO = CODE.parent
PAPER = REPO / "paper_repo" / "main.tex"


class Audit:
    """Collect named checks so one run reports every inconsistency."""

    def __init__(self) -> None:
        self.rows: list[dict[str, str | bool]] = []

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        self.rows.append({"name": name, "ok": bool(condition), "detail": detail})

    def equal(self, name: str, actual, expected) -> None:
        self.check(name, actual == expected, f"actual={actual!r}; expected={expected!r}")

    def close(self, name: str, actual: float, expected: float,
              atol: float = 1e-12) -> None:
        self.check(
            name,
            math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=atol),
            f"actual={actual!r}; expected={expected!r}",
        )

    @property
    def failures(self) -> list[dict[str, str | bool]]:
        return [row for row in self.rows if not row["ok"]]


def load_json(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def assigned_literal(path: Path, name: str):
    """Read a literal module constant without importing the module."""
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        target = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        elif isinstance(node, ast.AnnAssign):
            target = node.target
        if isinstance(target, ast.Name) and target.id == name:
            return ast.literal_eval(node.value)
    raise KeyError(f"{name} not found in {path}")


def flag_dict(tokens: list[str]) -> dict[str, str | bool]:
    """Convert the run drivers' flat command-line lists to a mapping."""
    result: dict[str, str | bool] = {}
    i = 0
    while i < len(tokens):
        flag = tokens[i]
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            result[flag] = tokens[i + 1]
            i += 2
        else:
            result[flag] = True
            i += 1
    return result


def line_has(audit: Audit, tex: str, name: str, anchor: str,
             tokens: list[str]) -> None:
    """Require all rendered values on the manuscript line containing anchor."""
    lines = [line for line in tex.splitlines() if anchor in line]
    missing = [token for token in tokens if not any(token in line for line in lines)]
    audit.check(
        name,
        bool(lines) and not missing,
        f"anchor={anchor!r}; missing={missing}",
    )


def signed(value: float, digits: int) -> str:
    return f"{value:+.{digits}f}"


def audit_figures(audit: Audit, tex: str) -> None:
    regen = load_module("_publication_regenerate", CODE / "regenerate_all.py")
    sync = load_module("_publication_sync", CODE / "sync_paper_figures.py")
    used = {
        Path(match).stem
        for match in re.findall(
            r"\\includegraphics\[[^]]*\]\{([^}]+)\}", tex
        )
    }
    audit.equal("figure stems equal sync allowlist", used, set(sync.FIGURES))

    # Two generators each emit two manuscript plates.
    produced_by_job = {
        "fig1": {"fig_model", "fig1"},
        "fig5": {"fig_omission", "fig_relocdata"},
    }
    generated: set[str] = set()
    for job in regen.JOBS:
        generated |= produced_by_job.get(job, {job})
    audit.equal("regeneration jobs cover manuscript stems", generated, used)
    for stem, source in sync.FIGURES.items():
        audit.check(f"figure source exists: {stem}", source.exists(), str(source))

    defaults = {
        "fig1": 'REPO_ROOT / "results" / "multi_seed" / "nohead_seed16"',
        "fig2": 'REPO_ROOT / "results" / "multi_seed" / "nohead_seed16"',
        "fig3": 'REPO_ROOT / "results" / "multi_seed_head" / "head_seed28"',
        "fig4": 'REPO_ROOT / "results" / "multi_seed_reloc" / "reloc_seed40"',
    }
    for figure, expression in defaults.items():
        source = (CODE / "figures" / figure / "utils.py").read_text()
        audit.check(
            f"canonical default: {figure}",
            expression in source,
            f"expected {expression}",
        )


def audit_methods(audit: Audit, tex: str) -> None:
    run23 = CODE / "runs" / "run_fig23_extra_seeds.py"
    run34 = CODE / "runs" / "run_fig34_parallel.py"
    nohead = flag_dict(assigned_literal(run23, "FIG2_BASE"))
    head = flag_dict(assigned_literal(run23, "FIG3_BASE"))
    reloc = flag_dict(assigned_literal(run34, "FIG4_BASE"))

    common = {
        "--n_trials": "1000",
        "--timesteps": "100",
        "--phase1_epochs": "100",
        "--max_steps": "20",
        "--interleave_epochs": "20",
        "--interleave_lr": "1e-4",
        "--rnn_rl_lr": "5e-5",
        "--pred_loss_weight": "0.3",
        "--reward_x": "0.3",
        "--reward_y": "0.3",
        "--fixed_goal": True,
    }
    for arm_name, arm in (("no-head", nohead), ("proximity", head), ("relocation", reloc)):
        for flag, expected in common.items():
            audit.equal(f"{arm_name} run argument {flag}", arm.get(flag), expected)
    audit.equal("no-head omits proximity objective", "--reward_pred_head" in nohead, False)
    audit.equal("proximity objective enabled", head.get("--reward_pred_head"), True)
    audit.equal("proximity objective weight", head.get("--reward_pred_weight"), "1.5")
    audit.equal("one-cycle replay episodes", head.get("--max_episodes"), "1000")
    audit.equal("one-cycle replay cycles", head.get("--n_cycles"), "1")
    for flag, expected in {
        "--max_episodes": "2000",
        "--n_cycles": "2",
        "--relocate_x": "0.7",
        "--relocate_y": "0.7",
        "--relocate_at_cycle": "1",
    }.items():
        audit.equal(f"relocation argument {flag}", reloc.get(flag), expected)

    # Check the stored exemplar metadata as well as the driver declarations.
    configs = {
        "no-head exemplar": (
            REPO / "results/multi_seed/nohead_seed16/config.json",
            {"seed": 16, "reward_pred_head": False, "n_cycles": 1},
        ),
        "proximity exemplar": (
            REPO / "results/multi_seed_head/head_seed28/config.json",
            {"seed": 28, "reward_pred_head": True, "n_cycles": 1},
        ),
        "relocation exemplar": (
            REPO / "results/multi_seed_reloc/reloc_seed40/config.json",
            {"seed": 40, "reward_pred_head": True, "n_cycles": 2},
        ),
    }
    for label, (path, expected) in configs.items():
        config = load_json(path)
        for key, value in {
            **expected,
            "n_trials": 1000,
            "timesteps": 100,
            "phase1_epochs": 100,
            "max_steps": 20,
            "interleave_epochs": 20,
            "rnn_rl_learning_rate": 5e-5,
            "controller_learning_rate": 1e-4,
            "pred_loss_weight": 0.3,
            "reward_pos": [0.3, 0.3],
        }.items():
            audit.equal(f"{label} config {key}", config.get(key), value)

    # Constants used by the analyses, rather than prose-only duplicates.
    helpers = (CODE / "lib" / "analysis_helpers.py").read_text()
    extinction = (CODE / "lib" / "extinction_analysis.py").read_text()
    reward_relative = (CODE / "lib" / "reward_relative_analysis.py").read_text()
    reward_vector = (CODE / "lib" / "reward_vector_analysis.py").read_text()
    ratemaps = (CODE / "lib" / "ratemaps.py").read_text()
    main_model = (CODE / "model" / "ratinabox_v11.py").read_text()
    relocation_model = (CODE / "model" / "ratinabox_relocation.py").read_text()
    extinction_model = (CODE / "model" / "ratinabox_extinction.py").read_text()
    implementation_fragments = {
        "entropy gate implementation": (helpers, "MIN_ENTROPY = 5.0"),
        "main Skaggs gate implementation": (helpers, "SKAGGS_THRESHOLD = 0.3"),
        "omission Skaggs gate implementation": (extinction, "SKAGGS_THR = 0.15"),
        "omission near radius implementation": (extinction, "NEAR_RADIUS = 0.15"),
        "relocation Skaggs gate implementation": (reward_relative, "SKAGGS_THR = 0.15"),
        "reward-centred window implementation": (reward_relative, "FRAME_HALF = 0.3"),
        "rate-map smoothing implementation": (ratemaps, "sigma=1.5"),
        "reward-vector grid implementation": (reward_vector, "M_GRID = 32"),
        "reward-vector ring bins": (reward_vector, "n_dist_bins: int = 6"),
        "reward-vector angle bins": (reward_vector, "n_angle_bins: int = 8"),
        "reward-vector null count": (reward_vector, "n_null: int = 100"),
        "bearing bins": (reward_vector, "nbins: int = 16"),
        "effect floors": (
            reward_vector,
            "effect_floor_ramp: float = 0.1, effect_floor_mrl: float = 0.1",
        ),
        "ring MRL ceiling": (reward_vector, "effect_floor_ring: float = 0.15"),
        "main episode warm-up": (main_model, "WARMUP_STEPS = 10"),
        "relocation episode warm-up": (relocation_model, "WARMUP_STEPS = 10"),
        "omission episode warm-up": (extinction_model, "WARMUP_STEPS = 10"),
        "main replay length honors CLI": (main_model, "REPLAY_STEPS = int(ARGS.replay_steps)"),
        "relocation replay length honors CLI": (
            relocation_model,
            "REPLAY_STEPS = int(ARGS.replay_steps)",
        ),
        "omission replay length honors CLI": (
            extinction_model,
            "REPLAY_STEPS = int(ARGS.replay_steps)",
        ),
        "decoder learning rate implementation": (
            main_model,
            '_rl_param_groups.append({"params": reward_pred_head.parameters(), "lr": 1e-4})',
        ),
        "online prediction is MSE only": (
            main_model,
            "total_loss = rl_loss + PRED_LOSS_WEIGHT * pred_loss",
        ),
        "interleaved prediction includes L1": (
            main_model,
            "loss = mse_loss + l1_loss",
        ),
        "interleaved policy is greedy": (
            main_model,
            "action_idx = torch.argmax(logits, dim=1).item()",
        ),
        "analysis-probe policy temperature": (
            main_model,
            "Categorical(logits=logits / 0.3).sample()",
        ),
        "omission indexing exception implementation": (
            extinction_model,
            "h_temp = b_hs[0].unsqueeze(0).detach()",
        ),
    }
    for name, (source, fragment) in implementation_fragments.items():
        audit.check(name, fragment in source, f"expected {fragment!r}")

    methods_fragments = [
        "$H = 512$", "$M=24$", "$32 \\times 32$", "$B = 1000$",
        "$T = 100$", "$\\lambda_{L1}=10^{-4}$", "batch size 64",
        "$T_{\\text{replay}}=50{,}000$", "$\\sigma_R = 0.15$",
        "$\\lambda_{\\text{rew}}=1.5$", "$\\lambda_{\\text{pred}}=0.3$",
        "$\\alpha_{\\text{RNN}}^{\\text{B}}=5\\times10^{-5}$",
        "$\\alpha_{\\text{RNN}}^{\\text{B}}=2\\times10^{-6}$",
        "H = -\\sum_{i} p_i \\log_2 p_i \\;\\geq\\; 5.0",
        "$K=20$", "2000 random cell-pair draws", "100 independent cyclic shifts",
        "omission simulator retained an earlier indexing convention",
        "Returns were bootstrapped from the critic",
        "$(0.3125,0.3125)$", "$(0.6875,0.6875)$",
        "final post-interleaving goal-directed probe used for tuning analyses",
        "entropy gate uses the $0.3$ cell threshold for all families",
        "10 uniformly random, unoptimized warm-up moves",
        "one-step-lagged action convention",
        "Skaggs information was computed on this unsmoothed",
        "no second entropy filter was applied",
        "Proximity-decoder LR",
        "online PPO update did not include the L1 term",
        "generated greedily by the current policy without early termination",
        "policy logits by $0.3$",
        "$F^A\\!\\to\\!F^B$ includes the intervening A-policy predictive update",
        "learned-to-withdrawn contrast therefore includes that intervening",
    ]
    for fragment in methods_fragments:
        audit.check(f"Methods states {fragment}", fragment in tex, fragment)


def audit_cohorts(audit: Audit) -> dict:
    cohorts = load_json(REPO / "results" / "aggregate" / "cohorts.json")
    headline = load_json(REPO / "results" / "aggregate" / "headline_stats.json")
    extinction = load_json(REPO / "results" / "aggregate" / "extinction_summary.json")
    reward_relative = load_json(
        REPO / "results/multi_seed_reloc/aggregate/reward_relative_summary.json"
    )
    expected = {
        "paired": (109, 55),
        "reloc": (72, 52),
        "extinction": (67, 51),
    }
    audit.close("cohort entropy threshold", cohorts["threshold_bits"], 5.0)
    for family, (launched, retained) in expected.items():
        audit.equal(f"{family} launched", cohorts[family]["n_launched"], launched)
        audit.equal(f"{family} retained", cohorts[family]["n"], retained)
        audit.equal(
            f"{family} seed count",
            len(cohorts[family]["seeds"]),
            retained,
        )
    audit.equal("headline paired cohort", headline["paired"]["cohort"]["n"], 55)
    audit.equal("headline relocation cohort", headline["reloc"]["cohort"]["n"], 52)
    audit.check(
        "omission statistics have one authoritative cache",
        "extinction" not in headline,
        "headline_stats.json must not duplicate extinction_summary.json",
    )
    audit.equal("omission usable cohort", extinction["k1"]["n_seeds"], 51)
    audit.equal(
        "reward-relative cohort",
        reward_relative["summary"]["n_seeds"],
        52,
    )
    return {
        "headline": headline,
        "extinction": extinction,
        "reward_relative": reward_relative,
    }


def audit_results(audit: Audit, tex: str, caches: dict) -> None:
    headline = caches["headline"]
    extinction = caches["extinction"]
    reward_relative = caches["reward_relative"]
    paired = headline["paired"]["cohort"]
    reloc = headline["reloc"]["cohort"]

    # Main behavioural and fixed-trajectory effects.
    x = paired["peak"]["nh_beh"]
    line_has(audit, tex, "behavioural peak result", "distance of field peaks", [
        signed(x["mean"], 3), f"{x['sd']:.3f}", f"{x['sem']:.3f}",
        f"{x['pct_toward_mean']:.1f}\\%", "55 networks",
    ])
    x = paired["com"]["nh_beh"]
    line_has(audit, tex, "behavioural COM result", "Field centres of mass gave", [
        signed(x["mean"], 3), f"{x['sd']:.3f}", f"{x['sem']:.3f}",
        f"{x['pct_toward_mean']:.1f}\\%",
    ])
    x = paired["peak"]["nh_frz"]
    retained = paired["peak"]["nh_retained"]
    line_has(audit, tex, "fixed-replay peak result", "Field peaks moved by", [
        signed(x["mean"], 3), f"{x['sd']:.3f}", f"{x['sem']:.3f}",
        f"{round(100 * retained['ratio_of_means']):d}\\%",
        f"{retained['n_replay_smaller']} of {retained['n']}",
    ])
    x = paired["com"]["nh_frz"]
    line_has(audit, tex, "fixed-replay COM result", "Centres of mass behaved", [
        signed(x["mean"], 3), f"{x['sd']:.3f}", f"{x['sem']:.3f}",
        f"{round(100 * paired['com']['nh_retained']['ratio_of_means']):d}\\%",
    ])
    line_has(audit, tex, "fixed-replay direction counts", "residual replay shift", [
        f"{paired['peak']['nh_frz']['pct_toward_mean']:.1f}\\%",
        f"{paired['peak']['nh_frz']['n_negative']} of 55",
        f"{paired['com']['nh_frz']['pct_toward_mean']:.1f}\\%",
    ])

    # Proximity teaching, plasticity, and the interaction.
    peak_hd, com_hd = paired["peak"]["hd_frz"], paired["com"]["hd_frz"]
    line_has(audit, tex, "proximity-teaching result", "field peaks moved toward reward", [
        signed(peak_hd["mean"], 3), f"{peak_hd['sd']:.3f}",
        f"{peak_hd['sem']:.3f}", f"{peak_hd['pct_toward_mean']:.1f}\\%",
        signed(com_hd["mean"], 3), f"{com_hd['pct_toward_mean']:.1f}\\%",
    ])
    rigid = paired["peak"]["rg_frz"]
    line_has(audit, tex, "near-rigid result", "Restricting recurrent plasticity", [
        signed(rigid["mean"], 4), f"{rigid['sd']:.4f}", f"{rigid['sem']:.4f}",
    ])
    absolute = [
        paired["peak"][name]["abs_disp"]
        for name in ("rg_frz", "nh_frz", "hd_frz")
    ]
    line_has(audit, tex, "absolute displacement result", "Mean absolute peak displacement", [
        *(f"{item['mean']:.3f}" for item in absolute),
        *(f"{item['sd']:.3f}" for item in absolute),
    ])
    interaction = paired["peak"]["interaction"]
    line_has(audit, tex, "condition by readout interaction", "condition-by-readout", [
        signed(interaction["mean"], 3), f"{interaction['sd']:.3f}",
        f"{interaction['sem']:.3f}", signed(paired["com"]["interaction"]["mean"], 3),
    ])

    # Reward relocation field geometry.
    peak_a, peak_b = reloc["peak"]["toward_A"], reloc["peak"]["toward_B"]
    com_a, com_b = reloc["com"]["toward_A"], reloc["com"]["toward_B"]
    line_has(audit, tex, "relocation toward A", "field peaks shifted toward A", [
        signed(peak_a["mean"], 3), f"{peak_a['sd']:.3f}",
        f"{peak_a['pct_toward_mean']:.1f}\\%", f"{peak_a['n_negative']} of 52",
        signed(com_a["mean"], 3), f"{com_a['pct_toward_mean']:.1f}\\%",
    ])
    line_has(audit, tex, "relocation toward B", "fields reorganized toward B", [
        signed(peak_b["mean"], 3), f"{peak_b['pct_toward_mean']:.1f}\\%",
        f"{peak_b['n_negative']} of 52", signed(com_b["mean"], 3),
        f"{com_b['pct_toward_mean']:.1f}\\%",
    ])
    line_has(audit, tex, "relocation step comparison", "Field peaks moved more toward A", [
        f"{reloc['peak']['A_minus_B']['n_A_larger']} of 52",
    ])

    # Omission and the Krishnan comparison.
    k1 = extinction["k1"]
    line_has(audit, tex, "omission field result", "Learning moved field peaks", [
        signed(k1["delta_learn"]["mean"], 3), f"{k1['delta_learn']['sd']:.3f}",
        signed(k1["delta_ext"]["mean"], 3), f"{k1['delta_ext']['sd']:.3f}",
    ])
    recovery = k1["recovery_pooled"]
    near = k1["near_frac_by_period"]
    line_has(audit, tex, "omission recovery result", "endpoint passed the original baseline", [
        f"{recovery['point']:.2f}", f"{recovery['ci_lo']:.2f}",
        f"{recovery['ci_hi']:.2f}", f"{100 * near['pred']['mean']:.1f}\\%",
        f"{100 * near['postLearn']['mean']:.1f}\\%",
        f"{100 * near['postExt']['mean']:.1f}\\%",
    ])
    contrast = extinction["k3"]["contrast_near_over_far"]
    line_has(audit, tex, "omission decoder result", "median near-to-far output ratio", [
        f"{contrast['postLearn']['median']:.0f}",
        f"{contrast['postExt']['median']:.1f}",
    ])

    krishnan = load_json(REPO / "data/krishnan2022/cache/krishnan_data.json")
    axon = krishnan["extended"]["axon_ramp"]
    rewarded_ratio = np.median(axon["rewarded"]["end_over_start"])
    omitted_ratio = np.median(axon["re_low"]["end_over_start"])
    line_has(audit, tex, "Krishnan axon result", "VTA--CA1 axons", [
        f"{rewarded_ratio:.1f}\\to{omitted_ratio:.1f}",
    ])
    fields = krishnan["extended"]["end_fraction_n12"]
    line_has(audit, tex, "Krishnan field fraction", "fraction of fields in the reward region", [
        f"{np.mean(fields['rewarded']):.1f}\\%",
        f"{np.mean(fields['re_low']):.1f}\\%",
        f"{fields['chance_pct']:.0f}\\%",
    ])
    model_rel = np.array([
        100 * (row["k2"]["postExt"]["all"] - row["k2"]["postLearn"]["all"])
        / row["k2"]["postLearn"]["all"]
        for row in extinction["per_seed"]
    ])
    line_has(
        audit,
        tex,
        "omission reliability near-radius caption",
        "Split-half map reliability",
        [f"$\\leq{extinction['params']['near_radius']:.2f}$"],
    )
    data_rel = 100 * np.asarray(krishnan["extended"]["reliability_n12"]["frac_change"])
    line_has(audit, tex, "reliability comparison", "Map reliability declined", [
        signed(model_rel.mean(), 1) + "\\%",
        signed(data_rel.mean(), 1) + "\\%",
        f"{model_rel.std(ddof=1) / np.sqrt(len(model_rel)):.1f}",
        f"{data_rel.std(ddof=1) / np.sqrt(len(data_rel)):.1f}",
    ])

    # Reward-relative classification and public Sosa data.
    sosa = load_json(REPO / "data/sosa2025/cache/sosa_tests.json")
    ss = sosa["summary"]
    rr = reward_relative["summary"]
    line_has(audit, tex, "reward-relative fractions", "Place-anchored cells", [
        f"{100 * ss['place_anchored_frac']['mean']:.1f}\\%",
        f"{100 * rr['controlled']['frac_place_anchored']['mean']:.1f}\\%",
        f"{100 * ss['rr_frac']['mean']:.1f}\\%",
        f"{100 * rr['controlled']['frac_reward_relative']['mean']:.1f}\\%",
    ])
    line_has(audit, tex, "behavioural place-anchored fraction", "goal-directed trajectories reduced", [
        f"{100 * rr['behavioural']['frac_place_anchored']['mean']:.1f}\\%",
    ])
    line_has(audit, tex, "Sosa latency result", "anticipatory licking", [
        f"{ss['lick_latency']['mean']:.1f}", f"{ss['lick_latency']['sd']:.1f}",
    ])
    line_has(audit, tex, "Sosa ordering result", "density of fields near the new zone", [
        f"{ss['map_latency']['mean']:.1f}",
        "five of six",
        f"{ss['wilcoxon_lick_vs_map_latency']['p']:.4f}",
    ])

    # Occupancy-matched estimator.
    occupancy = load_json(REPO / "results/aggregate/occupancy_matched_estimator.json")
    osum = occupancy["summary"]["cycle_1_goal"]
    line_has(audit, tex, "matched behavioural estimates", "Matching position and heading", [
        signed(osum["nohead_joint"]["mean"], 3),
        f"{osum['nohead_joint']['sd']:.3f}",
        signed(osum["head_joint"]["mean"], 3),
        f"{osum['head_joint']['sd']:.3f}",
    ])
    line_has(audit, tex, "same-parameter matching control", "identical network parameters", [
        signed(osum["nohead_null_naive"]["mean"], 3),
        signed(osum["head_null_naive"]["mean"], 3),
        signed(osum["nohead_null_joint"]["mean"], 3),
        signed(osum["head_null_joint"]["mean"], 3),
    ])
    line_has(audit, tex, "matched ordering result", "matched behavioural recordings", [
        signed(osum["nohead_joint"]["mean"], 3),
        signed(osum["head_joint"]["mean"], 3),
    ])

    # Bearing rotation summaries.  The CI resamples networks, preserving the
    # statistical unit used by the figure generator.
    for filename, anchor in (
        ("reward_vector_rotation_behavioral.json", "On goal-directed trajectories"),
        ("reward_vector_rotation_occupancy_controlled.json", "On the fixed replay trajectory"),
    ):
        rows = load_json(REPO / "results/multi_seed_reloc/aggregate" / filename)
        per_network = [
            np.asarray(row["dphi_goal_arr"], dtype=float)
            for row in rows if row["dphi_goal_arr"]
        ]
        pooled = np.concatenate(per_network)
        point = np.angle(np.mean(np.exp(1j * pooled)))
        rng = np.random.default_rng(0)
        boot = np.array([
            np.angle(np.mean(np.exp(1j * np.concatenate([
                per_network[k]
                for k in rng.integers(0, len(per_network), len(per_network))
            ]))))
            for _ in range(2000)
        ])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        geom = np.median(np.abs([row["delta_geom"] for row in rows]))
        line_has(audit, tex, f"bearing rotation: {filename}", anchor, [
            signed(np.degrees(point), 1) + "^\\circ",
            signed(np.degrees(lo), 1) + "^\\circ",
            signed(np.degrees(hi), 1) + "^\\circ",
            f"{np.degrees(geom):.0f}^\\circ",
        ])

    # Relocation displacement and success values are inexpensive to derive
    # from the retained cohort and protect two caption/text-only claims.
    sys.path.insert(0, str(CODE / "lib"))
    from analysis_helpers import load_period_centers_cached  # noqa: PLC0415

    reward_a = np.array([0.3, 0.3])
    reward_b = np.array([0.7, 0.7])
    direction = (reward_b - reward_a) / np.linalg.norm(reward_b - reward_a)
    model_displacement = []
    curves = []
    for seed in load_json(REPO / "results/aggregate/cohorts.json")["reloc"]["seeds"]:
        run = REPO / "results/multi_seed_reloc" / f"reloc_seed{seed}"
        sk_a, _, pk_a = load_period_centers_cached(
            str(run / "data"), "replay_reloc_postA"
        )
        sk_b, _, pk_b = load_period_centers_cached(
            str(run / "data"), "replay_reloc_postB"
        )
        gate = (sk_a >= 0.15) & (sk_b >= 0.15)
        model_displacement.append(
            (pk_b[gate] - pk_a[gate]) @ direction
            / np.linalg.norm(reward_b - reward_a)
        )
        curves.append(np.load(run / "learning_curves.npz")["rl_success"])
    model_displacement = np.concatenate(model_displacement)

    def wrap_cm(values):
        return (np.asarray(values, dtype=float) + 225.0) % 450.0 - 225.0

    data_displacement = []
    for row in sosa["per_session"]:
        if "skipped" in row:
            continue
        cells = row["cells"]
        reward_shift = wrap_cm(cells["zone_post_start"] - cells["zone_pre_start"])
        data_displacement.append(
            wrap_cm(np.asarray(cells["pk_post"]) - np.asarray(cells["pk_pre"]))
            / reward_shift
        )
    data_displacement = np.concatenate(data_displacement)
    line_has(audit, tex, "relocation displacement fractions", "Within $0.2$ of $0$", [
        f"{100 * np.mean(np.abs(model_displacement) <= 0.2):.0f}\\%",
        f"{100 * np.mean(np.abs(data_displacement) <= 0.2):.0f}\\%",
        f"{100 * np.mean(np.abs(model_displacement - 1) <= 0.2):.0f}\\%",
        f"{100 * np.mean(np.abs(data_displacement - 1) <= 0.2):.0f}\\%",
    ])
    curves = np.asarray(curves, dtype=float)
    smooth = np.ones(10) / 10
    smoothed = np.asarray([np.convolve(curve, smooth, mode="valid") for curve in curves])
    line_has(audit, tex, "model relocation success", "Model success at B", [
        f"{smoothed[:, 1000].mean():.2f}", f"{smoothed[:, 0].mean():.2f}",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print machine-readable results")
    args = parser.parse_args()

    audit = Audit()
    required = [
        PAPER,
        REPO / "results/aggregate/headline_stats.json",
        REPO / "results/aggregate/cohorts.json",
        REPO / "results/aggregate/extinction_summary.json",
        REPO / "results/aggregate/occupancy_matched_estimator.json",
        REPO / "results/multi_seed_reloc/aggregate/reward_relative_summary.json",
        REPO / "data/krishnan2022/cache/krishnan_data.json",
        REPO / "data/sosa2025/cache/sosa_tests.json",
    ]
    for path in required:
        audit.check(f"required input exists: {path.relative_to(REPO)}", path.exists(), str(path))
    if audit.failures:
        # Avoid secondary tracebacks when a required publication input is absent.
        pass
    else:
        tex = PAPER.read_text()
        audit_figures(audit, tex)
        audit_methods(audit, tex)
        caches = audit_cohorts(audit)
        audit_results(audit, tex, caches)

    if args.json:
        print(json.dumps({
            "ok": not audit.failures,
            "n_checks": len(audit.rows),
            "failures": audit.failures,
        }, indent=2))
    else:
        for row in audit.failures:
            print(f"FAIL: {row['name']}: {row['detail']}")
        print(
            f"paper/code audit: {len(audit.rows) - len(audit.failures)}/"
            f"{len(audit.rows)} checks passed"
        )
    return 1 if audit.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
