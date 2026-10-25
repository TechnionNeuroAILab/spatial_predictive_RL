"""
Successor-representation supplementary figure — analysis.

Why this figure exists
----------------------
The previous SR panel (the retired Figure 5 panel f)
reported that head-ON and no-head networks were indistinguishable
(0.2554 vs 0.2556, p = 0.83) — erasing the paper's central effect. It was
wrong. Its metric projected each cell's displacement onto a toward-reward
unit vector computed from that cell's PRE position:

    proj_i = (post_i - pre_i) . normalise(REWARD - pre_i)

With an off-centre reward, any drift toward the middle of the arena scores
as "toward reward" for the many cells on the far side, so the statistic
measures regression to the centre. A shuffled-cell null (re-pair post cells
with pre cells at random) returns more than the observed value, which is the
signature of a metric with no signal in it.

The correct statistic is the change in distance to reward,
    dd_i = ||post_i - R|| - ||pre_i - R||        (negative = toward reward)
which is what code/lib/analysis_helpers.frozen_replay_delta computes and what
Figure 5 panel b already uses.

Choosing nulls
--------------
A cell-identity shuffle is analytically inert for mean dd, because

    mean_i ||post_sigma(i) - R|| == mean_i ||post_i - R||

for any permutation sigma — the mean is a difference of two means, and
permuting one of them changes nothing. So the shuffle is:
  - a valid null for the paired statistics (frac_toward, signed-rank), which
    it does destroy, and
  - useless for the mean, for which we instead use a reward-location null:
    recompute dd toward other arena locations. If the shift is genuinely
    reward-directed, dd toward the true reward should stand out against dd
    toward its mirror images and random points.

Both are reported, so the failure mode that produced the old panel cannot
recur silently.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
_NEW_CODE = _REPO / "code" / "lib"
if str(_NEW_CODE) not in sys.path:
    sys.path.insert(0, str(_NEW_CODE))

from analysis_helpers import load_period_centers, seed_is_clean  # noqa: E402
from sr_common import REWARD, SKAGGS_THR  # noqa: E402

PRE_PERIOD = "replay_c1_pred"
POST_PERIOD = "replay_c1_postrl"

# Mirror images of the off-centre reward, plus the far corner. If the effect
# were generic centralisation these would score similarly to the true reward.
MIRROR_LOCATIONS = {
    "A (0.3,0.3)": np.array([0.3, 0.3]),
    "(0.7,0.3)": np.array([0.7, 0.3]),
    "(0.3,0.7)": np.array([0.3, 0.7]),
    "B (0.7,0.7)": np.array([0.7, 0.7]),
}


# ---------------------------------------------------------------------------
# Per-seed metrics
# ---------------------------------------------------------------------------

def load_pair(run_dir: Path, skaggs_thr: float = SKAGGS_THR):
    """Gated pre/post peak field centres for the frozen-replay pair."""
    d = Path(run_dir) / "data"
    try:
        sk_a, _, pk_a = load_period_centers(str(d), PRE_PERIOD)
        sk_b, _, pk_b = load_period_centers(str(d), POST_PERIOD)
    except Exception:
        return None
    gate = (sk_a >= skaggs_thr) & (sk_b >= skaggs_thr)
    if gate.sum() < 10:
        return None
    return pk_a[gate], pk_b[gate]


def delta_distance(pre: np.ndarray, post: np.ndarray, reward=REWARD) -> np.ndarray:
    """Per-cell change in distance to `reward`. Negative = moved toward it."""
    reward = np.asarray(reward, dtype=float)
    return np.linalg.norm(post - reward, axis=1) - np.linalg.norm(pre - reward, axis=1)


def projection_toward_reward(pre: np.ndarray, post: np.ndarray, reward=REWARD) -> np.ndarray:
    """The retired, biased metric — kept only as a labelled negative control.

    Positive = "toward reward" by its own convention. See module docstring for
    why this is not a usable measure of remapping.
    """
    reward = np.asarray(reward, dtype=float)
    d = reward - pre
    hat = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-10)
    return np.sum((post - pre) * hat, axis=1)


def shuffle_null(pre: np.ndarray, post: np.ndarray, fn, n_perm: int = 200,
                 seed: int = 0, reward=REWARD) -> np.ndarray:
    """Distribution of mean(fn) when post cells are re-paired with pre cells."""
    rng = np.random.default_rng(seed)
    out = np.empty(n_perm)
    for i in range(n_perm):
        out[i] = fn(pre, post[rng.permutation(len(post))], reward).mean()
    return out


def per_seed_metrics(run_dir: Path, n_perm: int = 200, seed: int = 0) -> dict | None:
    """Every shift metric and its null for one seed.

    Reports both the correct metric (`dd_*`, change in distance to reward) and
    the retired `proj_*` projection alongside the null that exposes its bias,
    so the two can be compared in the same figure rather than argued about.

    Parameters
    ----------
    run_dir : Path
        One seed's run directory.
    n_perm : int
        Permutations for each shuffle null.
    seed : int
        rng seed for the nulls.

    Returns
    -------
    dict or None
        Per-seed metrics, or None if the pre/post centre pair will not load.

    Notes
    -----
    `disp_mean` is the per-cell displacement magnitude and exists for the rigid
    control: a near-zero `dd_mean` is ambiguous between "fields did not move"
    and "fields moved in directions that cancel", and only the former supports
    the claim that plasticity is the gate.

    `dd_at::<name>` repeats the distance metric against each mirrored reward
    location -- same cells, different target -- as a location null.
    """
    pair = load_pair(run_dir)
    if pair is None:
        return None
    pre, post = pair

    dd = delta_distance(pre, post)
    proj = projection_toward_reward(pre, post)

    out = {
        "n": int(len(pre)),
        # correct metric
        "dd_mean": float(dd.mean()),
        "dd_frac_toward": float((dd < 0).mean()),
        # per-cell displacement magnitude. Essential for the rigid control:
        # a near-zero dd_mean is ambiguous between "fields did not move" and
        # "fields moved in directions that cancel". Only the former supports
        # the claim that plasticity is the gate.
        "disp_mean": float(np.linalg.norm(post - pre, axis=1).mean()),
        # retired metric + the null that exposes it
        "proj_mean": float(proj.mean()),
        "proj_null_mean": float(shuffle_null(pre, post, projection_toward_reward,
                                             n_perm, seed).mean()),
        # the shuffle applied to the correct metric: invariant for the mean,
        # informative for frac_toward
        "dd_null_mean": float(shuffle_null(pre, post, delta_distance, n_perm, seed).mean()),
    }
    rng = np.random.default_rng(seed + 1)
    fracs = [ (delta_distance(pre, post[rng.permutation(len(post))]) < 0).mean()
              for _ in range(n_perm) ]
    out["dd_frac_toward_null"] = float(np.mean(fracs))

    # reward-location null: same cells, different target locations
    for name, loc in MIRROR_LOCATIONS.items():
        out[f"dd_at::{name}"] = float(delta_distance(pre, post, loc).mean())
    return out


# ---------------------------------------------------------------------------
# Condition loaders
# ---------------------------------------------------------------------------

def discover_seeds(cond_dir: Path, prefix: str) -> list[int]:
    """Seed numbers under `cond_dir` matching `<prefix>seed<N>` that have a data/ subdirectory."""
    out = []
    for p in sorted(Path(cond_dir).glob(f"{prefix}seed*")):
        if p.is_dir() and (p / "data").exists():
            try:
                out.append(int(p.name.replace(prefix + "seed", "")))
            except ValueError:
                pass
    return sorted(out)


def load_condition(cond_dir: Path, prefix: str, seeds=None, min_entropy: float | None = None,
                   n_perm: int = 200) -> list[dict]:
    """Per-seed metrics for one condition, skipping seeds that fail the entropy-collapse gate."""
    cond_dir = Path(cond_dir)
    if not cond_dir.exists():
        return []
    if seeds is None:
        seeds = discover_seeds(cond_dir, prefix)
    rows = []
    for s in seeds:
        run_dir = cond_dir / f"{prefix}seed{s}"
        if not seed_is_clean(str(run_dir / "data"), [PRE_PERIOD, POST_PERIOD], min_entropy):
            continue
        r = per_seed_metrics(run_dir, n_perm=n_perm, seed=s)
        if r:
            r["seed"] = s
            rows.append(r)
    return rows
