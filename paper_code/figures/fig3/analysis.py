"""
Figure 3 analysis helpers.

The data structure (phase names, file layout, rate-map computation) is identical
to Figure 2. We load Figure 2's analysis module by absolute path to avoid circular
imports, then re-export its symbols.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# Figure 2's analysis.py (and figure 1's rate_maps.py, which it imports) do a
# bare `from utils import ...`, which resolves against whatever `utils`
# module is currently cached in sys.modules. Since figure 2 and figure 3 each
# have their own utils.py, we must pop the cache before and after exec'ing
# figure 2's module so that this figure's own `from utils import ...` (in
# generate_figure3.py) picks up figure 3's utils.py, not a stale figure-2
# one left behind in sys.modules (same pattern as figure 5/analysis.py).
_saved_utils = sys.modules.pop("utils", None)

# Load Figure 2's analysis.py by absolute path (avoids circular import via sys.path)
_FIG2_ANALYSIS = Path(__file__).parent.parent / "fig2" / "analysis.py"
_spec = importlib.util.spec_from_file_location("fig2_analysis", _FIG2_ANALYSIS)
_fig2 = importlib.util.module_from_spec(_spec)

# Figure 2's analysis.py imports from figure 1 via utils; ensure figure 1 is on path
_FIG1 = Path(__file__).parent.parent / "fig1"
_FIG2_DIR = _FIG2_ANALYSIS.parent
for _p in (str(_FIG1), str(_FIG2_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

sys.modules.pop("utils", None)
_spec.loader.exec_module(_fig2)

# Restore (or clear) the utils cache so this figure's own `from utils import ...`
# (in generate_figure3.py) picks up figure 3's utils.py, not figure 2's.
sys.modules.pop("utils", None)
if _saved_utils is not None:
    sys.modules["utils"] = _saved_utils
sys.path.insert(0, str(_HERE))

# Re-export everything Figure 2 exports
load_period = _fig2.load_period
load_phases = _fig2.load_phases
fig1_panel_d_neuron_ids = _fig2.fig1_panel_d_neuron_ids
shared_place_cell_ids = _fig2.shared_place_cell_ids
paired_distance_stats = _fig2.paired_distance_stats
fig_replay_remap_grid = _fig2.fig_replay_remap_grid
fig_centers_abc = _fig2.fig_centers_abc
fig_delta_violins = _fig2.fig_delta_violins
fig_distance_histograms = _fig2.fig_distance_histograms
FIG1_PANEL_D_NEURONS = _fig2.FIG1_PANEL_D_NEURONS
