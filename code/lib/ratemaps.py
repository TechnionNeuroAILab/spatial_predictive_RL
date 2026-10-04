"""Rate maps, Skaggs spatial information and field centres, shared by every analysis.

Maps are indexed [x_bin, y_bin]; field centres are returned as (x, y) at the grid
points k/m the simulator's agent occupies.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def build_rate_maps(df_beh, df_act, m=32, hidden_dim=512):
    """Mean activity per m x m spatial bin (unvisited bins = 0) and occupancy fraction."""
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
    """Skaggs information per unit; r_bar is the unweighted mean over all m*m bins."""
    hidden_dim = rate_maps.shape[0]
    occ_flat = occupancy.reshape(-1)# (m*m,)
    rate_flat = rate_maps.reshape(hidden_dim, -1).T # (m*m, N)
    mean_rate = rate_flat.mean(axis=0)# (N,)
    ratio = np.divide(rate_flat, mean_rate[None, :], out=np.zeros_like(rate_flat), where=mean_rate[None, :] > 1e-10)
    log_ratio = np.where(ratio > 0, np.log2(np.maximum(ratio, 1e-300)), 0.0)
    return (occ_flat[:, None] * ratio * log_ratio).sum(axis=0)


def smooth_and_norm(rm, sigma=1.5):
    """Gaussian-smooth a rate map and rescale it to [0, 1]."""
    s = gaussian_filter(rm, sigma=sigma)
    if s.max() > 0:
        s = (s - s.min()) / (s.max() - s.min())
    return s


def field_center_of_mass(rm):
    """Activity-weighted field centre, returned as (x, y)."""
    m = rm.shape[0]
    total = float(rm.sum())
    if total <= 0:
        return (0.5, 0.5)             # silent map -> arena centre
    xs = np.arange(m) / m             # bin k holds the grid point k/m
    ys = np.arange(m) / m
    # rm is [x_bin, y_bin]: axis 0 carries x, axis 1 carries y.
    cx = float((rm * xs[:, None]).sum() / total)
    cy = float((rm * ys[None, :]).sum() / total)
    return (cx, cy)


def field_peak(rm2d):
    """Location of the map's maximum, returned as (x, y)."""
    m = rm2d.shape[0]
    row, col = np.unravel_index(int(np.argmax(rm2d)), rm2d.shape)
    # (row, col) is (x_bin, y_bin).
    return np.array([row / m, col / m], dtype=float)
