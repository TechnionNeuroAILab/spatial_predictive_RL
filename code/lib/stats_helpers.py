"""
Shared statistical helpers for the code pipeline.

Generic tests and corrections that belong to no single analysis. Everything
here is pure numpy/scipy and takes plain arrays, so it has no dependency on run
directories, periods or the dataframe layer.

not a home for the per-analysis null models -- the shuffle and bootstrap
routines in reward_vector_analysis, successor_representation and
reward_relative_analysis are specific to their own statistics and stay there.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def fdr_bh(p_values: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg step-up FDR correction; returns q-values in input order."""
    p = np.asarray(p_values, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    adj = np.empty_like(ranked)
    prev = 1.0
    # Step up from the largest p: each q is capped by the next-larger one,
    # which enforces monotonicity.
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        adj[i] = prev
    out = np.empty_like(adj)
    out[order] = np.clip(adj, 0.0, 1.0)
    return out
