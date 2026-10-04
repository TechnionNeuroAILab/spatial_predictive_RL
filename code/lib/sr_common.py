"""SR constants and the random-walk SR baseline, built once per process (a 900 x 900 inversion).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from successor_representation import (  # noqa: E402
    build_T_random,
    compute_sr_matrix,
    state_centers,
)

# must match the simulator: m = 32, movement restricted to [1, m-2]
M, WB = 32, 1
GAMMA = 0.99

SKAGGS_THR = 0.3
REWARD = np.array([0.3, 0.3])

_T_RANDOM = None
_SR_RANDOM = None
_STATE_XY = None


def get_shared_random():
    """Cached random-walk transition matrix, its SR and the state centres."""
    global _T_RANDOM, _SR_RANDOM, _STATE_XY
    if _T_RANDOM is None:
        _T_RANDOM = build_T_random(M, WB)
        _SR_RANDOM, _, _ = compute_sr_matrix(_T_RANDOM, GAMMA, "auto", 500, 1e10)
        _STATE_XY = state_centers(M, WB)
    return _T_RANDOM, _SR_RANDOM, _STATE_XY
