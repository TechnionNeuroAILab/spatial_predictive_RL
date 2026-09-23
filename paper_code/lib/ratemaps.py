"""Rate-map primitives shared by the analysis pipeline and the figure generators.

Pure numpy/scipy: no matplotlib, no seaborn and no project imports, so this
module can be imported from anywhere on the path without dragging in a plotting
stack.

convention. Rate maps are indexed [x_bin, y_bin] throughout. `field_peak` and
`field_center_of_mass` both return their two components in the same, mutually
consistent, swapped order relative to their (x, y) naming -- the project's
documented (y, x) transposition, which is distance-preserving for the
on-diagonal rewards used throughout (see `occupancy_matched_estimator`).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def build_rate_maps(df_beh, df_act, m=32, hidden_dim=512):
    """Bin hidden-unit activity into spatial rate maps plus an occupancy map.

    Parameters
    ----------
    df_beh : pandas.DataFrame
        One row per timestep, columns `x` and `y` in normalized arena
        coordinates. Row order defines timestep order.
    df_act : pandas.DataFrame
        Column `activity`, flattened row-major to (n_steps * hidden_dim,).
        Reshaped here without a length check.
    m : int
        Arena is binned m x m.
    hidden_dim : int
        Number of RNN hidden units.

    Returns
    -------
    rate_maps : ndarray (hidden_dim, m, m), float64
        Mean activity per spatial bin, indexed [x_bin, y_bin]. Never-visited bins are 0.
    occupancy : ndarray (m, m), float64
        Fraction of timesteps spent in each bin, same indexing; sums to 1.
    """
    n_steps = len(df_beh)
    act_mat = df_act["activity"].values.reshape(n_steps, hidden_dim)
    x_bin = np.clip((df_beh["x"].values * m).astype(int), 0, m - 1)
    y_bin = np.clip((df_beh["y"].values * m).astype(int), 0, m - 1)
    gc = x_bin * m + y_bin
    n_bins = m * m
    bin_sum = np.zeros((n_bins, hidden_dim), dtype=np.float64)
    np.add.at(bin_sum, gc, act_mat)
    bin_count = np.bincount(gc, minlength=n_bins).astype(float)

    rate_flat = np.divide(bin_sum, bin_count[:, None],out=np.zeros_like(bin_sum),where=bin_count[:, None] > 0)
    rate_maps = rate_flat.T.reshape(hidden_dim, m, m)  # (N, m, m)
    occupancy = (bin_count / n_steps).reshape(m, m)
    return rate_maps, occupancy


def compute_skaggs(rate_maps, occupancy):
    """Skaggs spatial information per unit, in bits per unit of activity.

    Parameters
    ----------
    rate_maps : ndarray (hidden_dim, m, m)
        Per-unit mean activity per spatial bin
    occupancy : ndarray (m, m)
        Fraction of timesteps per bin.
        Must share `rate_maps`' binning; both are flattened the same way, so
        the [x_bin, y_bin] indexing does not matter here.

    Returns
    -------
    skaggs : ndarray (hidden_dim,), float64
        sum_i p_i * (r_i / r_bar) * log2(r_i / r_bar), where `r_bar` is the
        unweighted mean of that unit's map over all m*m bins -- including
        never-visited bins, which hold 0 -- rather than the occupancy-weighted
        mean sum_i p_i * r_i of the standard Skaggs formula. Units with
        `r_bar` <= 1e-10 score 0.

        This is the project's canonical Skaggs. The occupancy-weighted form is
        implemented separately in `occupancy_matched_estimator`, which
        deliberately does not use this one; see its module docstring.
    """
    hidden_dim = rate_maps.shape[0]
    occ_flat = occupancy.reshape(-1)                    # (m*m,)
    rate_flat = rate_maps.reshape(hidden_dim, -1).T     # (m*m, N)
    mean_rate = rate_flat.mean(axis=0)                  # (N,)
    ratio = np.where(mean_rate[None, :] > 1e-10,rate_flat / mean_rate[None, :], 0.0)
    log_ratio = np.where(ratio > 0, np.log2(np.maximum(ratio, 1e-300)), 0.0)
    return (occ_flat[:, None] * ratio * log_ratio).sum(axis=0)


def smooth_and_norm(rm, sigma=1.5):
    s = gaussian_filter(rm, sigma=sigma)
    if s.max() > 0:
        s = (s - s.min()) / (s.max() - s.min())
    return s


def field_center_of_mass(rm):
    """Activity-weighted centre of mass of one unit's field, in arena coords."""
    m = rm.shape[0]
    total = float(rm.sum())
    if total <= 0:
        return (0.5, 0.5)             # empty/silent map -> arena centre
    xs = np.arange(m) / m + 0.5 / m   # bin centres
    ys = np.arange(m) / m + 0.5 / m
    # rm is indexed [x_bin, y_bin], so xs broadcasts over y: cx/cy come back
    # swapped. field_peak swaps identically, so the two stay consistent.
    cx = float((rm * xs[None, :]).sum() / total)
    cy = float((rm * ys[:, None]).sum() / total)
    return (cx, cy)


def field_peak(rm2d):
    """Peak location of one unit's field, in arena coords."""
    m = rm2d.shape[0]
    row, col = np.unravel_index(int(np.argmax(rm2d)), rm2d.shape)
    # (row, col) is (x_bin, y_bin); emitted in the opposite order, matching the
    # same swap in field_center_of_mass.
    return np.array([col / m + 0.5 / m, row / m + 0.5 / m], dtype=float)
