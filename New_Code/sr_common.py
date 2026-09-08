"""
Shared successor-representation constants and the cached random-walk SR.

Extracted from the former New_Code/sr_vs_model_panel.py, which is being
retired because its headline metric was biased (see
code_analysis/figure_sr/analysis.py::projection_toward_reward for the details
and the null that exposes it). Two independent consumers need these symbols:

  - code_analysis/figure_sr/analysis.py          (the corrected SR figure)
  - code_analysis/figure 5/sr_direction_analysis.py  (SR shift-direction test)

Keeping them here rather than in either figure directory avoids a
figure-imports-figure dependency and the sys.modules["utils"] shadowing hazard
that the figure directories already have to work around.
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

# Grid geometry — must match the simulator (Code/ratinabox_v11.py: m = 32,
# movement restricted to [1, m-2]).
M, WB = 32, 1
GAMMA = 0.99

SKAGGS_THR = 0.3
REWARD = np.array([0.3, 0.3])
GOAL_RADIUS = 0.15

_T_RANDOM = None
_SR_RANDOM = None
_STATE_XY = None


def get_shared_random():
    """Random-walk transition matrix, its SR, and state centres (cached).

    Building the SR is the expensive step, and every caller wants the same
    unbiased-walk baseline, so it is computed once per process.
    """
    global _T_RANDOM, _SR_RANDOM, _STATE_XY
    if _T_RANDOM is None:
        _T_RANDOM = build_T_random(M, WB)
        _SR_RANDOM, _, _ = compute_sr_matrix(_T_RANDOM, GAMMA, "auto", 500, 1e10)
        _STATE_XY = state_centers(M, WB)
    return _T_RANDOM, _SR_RANDOM, _STATE_XY


# Backwards-compatible alias for the old private name.
_get_shared_random = get_shared_random
