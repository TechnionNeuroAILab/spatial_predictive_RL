from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from plot_goal_directed_maps import (
    build_rate_maps,
    compute_skaggs,
    smooth_and_norm,
    field_center_of_mass,
)
from plot_replay_v11 import field_peak

def period_paths(data_dir: str, period: str) -> tuple[str, str]:
    beh = os.path.join(data_dir, f"df_behavior_{period}.pkl")
    act = os.path.join(data_dir, f"df_activations_{period}.pkl")
    return beh, act

def has_period(data_dir: str, period: str) -> bool:
    beh, act = period_paths(data_dir, period)
    return os.path.exists(beh) and os.path.exists(act)


def load_period_centers(data_dir: str, period: str, m: int = 32, hidden_dim: int = 512):
    beh_path, act_path = period_paths(data_dir, period)
    if not (os.path.exists(beh_path) and os.path.exists(act_path)):
        raise FileNotFoundError(f"Missing period '{period}' in {data_dir}")
    beh = pd.read_pickle(beh_path)
    act = pd.read_pickle(act_path)
    rm, occ = build_rate_maps(beh, act, m, hidden_dim)
    sk = compute_skaggs(rm, occ)
    com = np.array(
        [field_center_of_mass(smooth_and_norm(rm[n])) for n in range(hidden_dim)],
        dtype=float,
    )
    peak = np.array(
        [field_peak(smooth_and_norm(rm[n])) for n in range(hidden_dim)],
        dtype=float,
    )
    del rm
    return sk, com, peak


def load_period_activity(data_dir: str, period: str, hidden_dim: int = 512):
    beh_path, act_path = period_paths(data_dir, period)
    if not (os.path.exists(beh_path) and os.path.exists(act_path)):
        raise FileNotFoundError(f"Missing period '{period}' in {data_dir}")
    beh = pd.read_pickle(beh_path)
    act = pd.read_pickle(act_path)
    n_steps = len(beh)
    if len(act) != n_steps * hidden_dim:
        raise ValueError(
            f"Activation rows ({len(act)}) != n_steps*hidden_dim "
            f"({n_steps}*{hidden_dim}); cannot reshape safely for '{period}'."
        )
    X = act["activity"].to_numpy(dtype=float).reshape(n_steps, hidden_dim)
    x = beh["x"].to_numpy(dtype=float)
    y = beh["y"].to_numpy(dtype=float)
    theta = beh["theta"].to_numpy(dtype=float) if "theta" in beh.columns else np.zeros(n_steps)
    return X, x, y, theta

def dist_to_reward(centers: np.ndarray, reward) -> np.ndarray:
    reward = np.asarray(reward, dtype=float)
    return np.linalg.norm(centers - reward[None, :], axis=1)