#!/usr/bin/env python3
"""
Extinction analysis: Krishnan-analogue metrics on the frozen-replay triple.

Runs produced by Code/ratinabox_extinction.py: head-ON learning at reward A
(cycle 1, bit-identical to the head condition -- verified), then a cycle with
reward delivery and the head objective OFF, with one frozen trajectory replayed
through three weight snapshots: replay_ext_pred / postLearn / postExt.

Metrics (all on the frozen triple, occupancy identical by construction):
  K1  over-representation collapse: near-A field-peak fraction and mean
      peak-distance-to-A per snapshot; delta_learn (pred->postLearn, expected
      < 0 = toward A) and delta_ext (postLearn->postExt, expected > 0 = decay
      back toward baseline). Pooled recovery fraction -sum(delta_ext)/
      sum(delta_learn) with a seed-level bootstrap CI (per-seed ratios are
      unstable; never average ratios).
  K2  reliability analogue (Krishnan: trial-to-trial reliability 0.47 -> 0.32
      under extinction): split each 50k-step replay into interleaved 1k-step
      blocks, odd/even half rate maps, per-cell split-half Pearson r;
      per-seed mean, postLearn vs postExt, near-A vs far-A strata.
  K3  dopamine proxy: offline reward-head output vs distance-to-A, using the
      cycle-boundary checkpoint (postLearn head) vs the final checkpoint
      (postExt head) on the SAME replay hidden states.

Conventions: Delta-d NEGATIVE = toward reward. Skaggs gate 0.15 required in
all three periods (triple-gated contrast precedent). Entropy gate 5.5 on the
triple. Paired Wilcoxon across seeds for every contrast (pairing: same seed).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_NEW_CODE = _REPO / "paper_code" / "lib"
if str(_NEW_CODE) not in sys.path:
    sys.path.insert(0, str(_NEW_CODE))

from occupancy_matched_estimator import (  # noqa: E402
    HIDDEN, M, load_period_arrays, position_bins, rate_maps_from,
    skaggs_weighted,
)
from occupancy_matched_estimator import centres_from  # noqa: E402

REWARD_A = np.array([0.3, 0.3])
SKAGGS_THR = 0.15
NEAR_RADIUS = 0.15
PERIODS = ["replay_ext_pred", "replay_ext_postLearn", "replay_ext_postExt"]
BLOCK = 1000


def triple_maps(data_dir: Path):
    out = []
    for p in PERIODS:
        pos, _, act = load_period_arrays(data_dir, p)
        gc = position_bins(pos)
        rm, occ = rate_maps_from(act, gc)
        out.append({"pos": pos, "act": act, "gc": gc, "rm": rm, "occ": occ,
                    "sk": skaggs_weighted(rm, occ)})
    return out


def k1_overrepresentation(triple) -> dict:
    """Near-A fraction + mean distance per snapshot; deltas; gated 3-way."""
    gate = np.ones(HIDDEN, bool)
    for t in triple:
        gate &= t["sk"] >= SKAGGS_THR
    if gate.sum() < 10:
        return {}
    peaks = []
    for t in triple:
        pk, _ = centres_from(t["rm"])
        peaks.append(pk[gate])
    d = [np.linalg.norm(p - REWARD_A, axis=1) for p in peaks]
    return {
        "n_cells": int(gate.sum()),
        "near_frac": [float((di < NEAR_RADIUS).mean()) for di in d],
        "mean_dist": [float(di.mean()) for di in d],
        # negative = toward A
        "delta_learn": float((d[1] - d[0]).mean()),
        "delta_ext": float((d[2] - d[1]).mean()),
        "delta_net": float((d[2] - d[0]).mean()),
    }


def k2_reliability(data_dir: Path, triple) -> dict:
    """Split-half rate-map reliability per snapshot, near-A vs far-A cells."""
    out = {}
    # near/far strata defined by postLearn peak distance (the learned state)
    gate = np.ones(HIDDEN, bool)
    for t in triple:
        gate &= t["sk"] >= SKAGGS_THR
    pk_learn, _ = centres_from(triple[1]["rm"])
    dist_learn = np.linalg.norm(pk_learn - REWARD_A, axis=1)
    near = gate & (dist_learn < 2 * NEAR_RADIUS)
    far = gate & (dist_learn >= 2 * NEAR_RADIUS)

    for label, t in zip(["pred", "postLearn", "postExt"], triple):
        n = len(t["pos"])
        block_id = (np.arange(n) // BLOCK) % 2      # interleaved halves
        rel = np.full(HIDDEN, np.nan)
        rms = []
        for h in (0, 1):
            m_ = block_id == h
            rm, _ = rate_maps_from(t["act"][m_], t["gc"][m_])
            rms.append(rm.reshape(HIDDEN, -1))
        a, b = rms
        for i in np.where(gate)[0]:
            if a[i].std() > 1e-9 and b[i].std() > 1e-9:
                rel[i] = float(np.corrcoef(a[i], b[i])[0, 1])
        out[label] = {
            "all": float(np.nanmean(rel[gate])),
            "near": float(np.nanmean(rel[near])) if near.sum() >= 5 else None,
            "far": float(np.nanmean(rel[far])) if far.sum() >= 5 else None,
        }
    return out


def k3_dopamine_proxy(run_dir: Path, triple) -> dict | None:
    """Head output vs distance-to-A on the frozen trajectory, postLearn vs
    postExt head weights (cycle-boundary vs final checkpoint)."""
    import torch
    run_dir = Path(run_dir)
    ckpts = {"postLearn": run_dir / "checkpoints" / "phase2_cycle1_end.pt",
             "postExt": run_dir / "checkpoints" / "phase2_final.pt"}
    if not all(p.exists() for p in ckpts.values()):
        alt = list((run_dir / "checkpoints").glob("*.pt")) if (run_dir / "checkpoints").exists() else []
        print(f"  k3: missing checkpoints ({[p.name for p in ckpts.values()]}); have {[a.name for a in alt]}")
        return None

    # head architecture: Linear(512,64) ReLU Linear(64,1) Sigmoid
    def head_from(ckpt_path):
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # save_checkpoint stores it under 'reward_pred_head_state_dict'
        sd = payload.get("reward_pred_head_state_dict") or payload.get("reward_pred_head")
        if sd is None:
            return None
        import torch.nn as nn
        head = nn.Sequential(nn.Linear(HIDDEN, 64), nn.ReLU(),
                             nn.Linear(64, 1), nn.Sigmoid())
        # state dict keys are like net.0.weight -> strip "net."
        sd = { (k[4:] if k.startswith("net.") else k): v for k, v in sd.items() }
        head.load_state_dict(sd)
        head.eval()
        return head

    # The head receives no gradient during extinction (lambda_rew -> 0), so its
    # weights are identical in both checkpoints -- the model's analogue of an
    # unchanged dopaminergic readout circuit. What changes is the REPRESENTATION
    # it reads: evaluate the SAME (postLearn) head on the postLearn vs postExt
    # replay activations (identical trajectory), so any profile change reflects
    # the drift of the map under the frozen proximity readout.
    head = head_from(ckpts["postLearn"])
    if head is None:
        return None
    edges = np.linspace(0, 0.9, 10)
    out = {"dist_bin_centres": ((edges[:-1] + np.diff(edges) / 2)).tolist()}
    for label, t in [("postLearn", triple[1]), ("postExt", triple[2])]:
        pos = t["pos"]
        act = torch.tensor(t["act"], dtype=torch.float32)
        dist = np.linalg.norm(pos - REWARD_A, axis=1)
        which = np.digitize(dist, edges) - 1
        with torch.no_grad():
            rho = head(act).squeeze(1).numpy()
        prof = [float(np.mean(rho[which == b])) if (which == b).sum() > 10 else None
                for b in range(len(edges) - 1)]
        out[label] = prof
    # the analytic target for reference
    sig = 0.15
    out["target"] = [float(np.exp(-c**2 / (2 * sig**2))) for c in out["dist_bin_centres"]]
    return out


def per_seed_row(run_dir: Path, seed: int) -> dict | None:
    d = Path(run_dir) / "data"
    try:
        triple = triple_maps(d)
    except Exception as e:
        print(f"  seed {seed}: load failed ({e})")
        return None
    row = {"seed": seed}
    row["k1"] = k1_overrepresentation(triple)
    if not row["k1"]:
        return None
    row["k2"] = k2_reliability(d, triple)
    row["k3"] = k3_dopamine_proxy(run_dir, triple)
    return row
