"""Per-seed extinction metrics on one frozen trajectory replayed through pred / postLearn / postExt weights.

K1 field concentration near reward A, K2 split-half map reliability, K3 the reward head's proximity profile.
Delta-d negative = toward A; Skaggs gate 0.15 in all three periods. Aggregated by aggregate/run_extinction_analysis.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_NEW_CODE = _REPO / "code" / "lib"
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
    """Arrays, rate maps, occupancy and Skaggs scores for each of the three snapshots."""
    out = []
    for p in PERIODS:
        pos, _, act = load_period_arrays(data_dir, p)
        gc = position_bins(pos)
        rm, occ = rate_maps_from(act, gc)
        out.append({"pos": pos, "act": act, "gc": gc, "rm": rm, "occ": occ,
                    "sk": skaggs_weighted(rm, occ)})
    return out


def k1_overrepresentation(triple) -> dict:
    """K1: near-A fraction, mean peak distance and the learn / ext / net shifts; {} if < 10 cells."""
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
    """K2: split-half rate-map correlation per snapshot, all / near-A / far-A cells."""
    out = {}
    # near / far split by the postLearn peak, at 2 * NEAR_RADIUS
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
        rms, seen = [], []
        for h in (0, 1):
            m_ = block_id == h
            rm, occ_h = rate_maps_from(t["act"][m_], t["gc"][m_])
            rms.append(rm.reshape(HIDDEN, -1))
            seen.append(occ_h.reshape(-1) > 0)
        both = seen[0] & seen[1]                    # bins visited in both halves; shared zeros would inflate r
        a, b = rms[0][:, both], rms[1][:, both]
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
    """K3: the postLearn reward head's output vs distance to A, on the postLearn and postExt replays."""
    import torch
    run_dir = Path(run_dir)
    ckpts = {"postLearn": run_dir / "checkpoints" / "phase2_cycle1_end.pt",
             "postExt": run_dir / "checkpoints" / "phase2_final.pt"}
    if not all(p.exists() for p in ckpts.values()):
        alt = list((run_dir / "checkpoints").glob("*.pt")) if (run_dir / "checkpoints").exists() else []
        print(f"  k3: missing checkpoints ({[p.name for p in ckpts.values()]}); have {[a.name for a in alt]}")
        return None

    def head_from(ckpt_path):
        """Rebuild the head Linear(512,64)-ReLU-Linear(64,1)-Sigmoid from a checkpoint."""
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = payload.get("reward_pred_head_state_dict") or payload.get("reward_pred_head")
        if sd is None:
            return None
        import torch.nn as nn
        head = nn.Sequential(nn.Linear(HIDDEN, 64), nn.ReLU(),
                             nn.Linear(64, 1), nn.Sigmoid())
        sd = { (k[4:] if k.startswith("net.") else k): v for k, v in sd.items() }   # drop "net."
        head.load_state_dict(sd)
        head.eval()
        return head

    # The head gets no gradient during extinction, so one head serves both replays.
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
    sig = 0.15                     # the head's training target, for reference
    out["target"] = [float(np.exp(-c**2 / (2 * sig**2))) for c in out["dist_bin_centres"]]
    return out


def per_seed_row(run_dir: Path, seed: int) -> dict | None:
    """K1-K3 for one seed; None if loading fails or K1 has < 10 cells."""
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
