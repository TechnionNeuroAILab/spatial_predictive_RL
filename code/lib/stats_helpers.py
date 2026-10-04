"""Shared statistics that belong to no single analysis: plain arrays in, plain arrays out.
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
    # step up from the largest p; capping by the next q keeps q monotone in p
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        adj[i] = prev
    out = np.empty_like(adj)
    out[order] = np.clip(adj, 0.0, 1.0)
    return out
