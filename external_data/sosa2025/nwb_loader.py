"""Loader for Sosa/Plitt/Giocomo 2025 reward-relocation NWB files (DANDI 001361).

Each NWB (one per subject x session) contains:
  - processing/behavior/BehavioralTimeSeries: position, speed, lick, reward_zone (binary
    zone-entry indicator), Reward (delivery events), trial number, trial_start, teleport,
    environment, autoreward, scanning  -- all per imaging frame, shared timestamps.
  - processing/ophys: Deconvolved/plane0 (events), Fluorescence/plane0 (raw F),
    Neuropil/plane0, ImageSegmentation/PlaneSegmentation (iscell) -- (T, n_rois) @ 15.5078 Hz.

The session "scene" (e.g. Env1_LocationA_to_C) is the tail of the NWB identifier.
Reward zones: A=[80,130], B=[200,250], C=[320,370] cm; in switch ("_to_") sessions the zone
moves at ~trial 30 (recover per-trial zone from reward_zone x position, as done here).

Usage:
    from nwb_loader import load_session
    d = load_session("nwb/sub-m11/sub-m11_ses-05_behavior+ophys.nwb")
"""

import re

import h5py
import numpy as np
import pandas as pd

ZONES = {"A": (80.0, 130.0), "B": (200.0, 250.0), "C": (320.0, 370.0)}
ITI_POS = -500.0  # position value used during the inter-trial "teleport" gap


def _zone_label(pos_cm, tol=25.0):
    """Nearest canonical zone label for a reward-zone-entry position."""
    if not np.isfinite(pos_cm):
        return ""
    for lab, (start, _) in ZONES.items():
        if abs(pos_cm - start) <= tol:
            return lab
    return "?"


def load_session(path, dff_neuropil_coeff=0.7, cells_only=True):
    """Load one Sosa 2025 NWB session.

    Parameters
    ----------
    path : str
        Path to a sub-*_ses-*_behavior+ophys.nwb file.
    dff_neuropil_coeff : float
        F_corr = F - c * Fneu used for dF/F (suite2p convention, c=0.7).
    cells_only : bool
        If True, keep only ROIs with suite2p iscell==1.

    Returns
    -------
    dict with keys:
      t            (T,) timestamps in s (behavior & imaging share this clock)
      pos          (T,) position in cm (-500 during inter-trial interval)
      lick         (T,) licks per frame
      speed        (T,) cm/s
      trial        (T,) trial (lap) number, -1 outside trials
      trial_df     DataFrame, one row per trial: trial, start_time, stop_time,
                   start_idx, stop_idx, reward_zone_pos (cm, NaN if never entered),
                   zone_label (A/B/C), n_rewards, is_post_switch
      events       (n_cells, T) deconvolved events
      dff          (n_cells, T) (F - c*Fneu) / robust baseline, per cell
      F, Fneu      (n_cells, T) raw and neuropil fluorescence
      cell_ids     (n_cells,) ROI indices into the original PlaneSegmentation
      iscell       (n_rois_total, 2) suite2p iscell array (all ROIs)
      rate         imaging rate (Hz)
      scene        e.g. "Env1_LocationA_to_C"
      is_switch    bool, True if "_to_" in scene
      switch_trial first trial number at the new reward zone (None if not a switch)
      subject, session, identifier
    """
    with h5py.File(path, "r") as f:
        ident = f["identifier"][()].decode()
        scene = ident.rstrip("/").split("/")[-1]
        subject = f["general/subject/subject_id"][()].decode()
        m = re.search(r"ses-(\d+)", str(path))
        session = int(m.group(1)) if m else None

        bts = f["processing/behavior/BehavioralTimeSeries"]
        t = bts["position/timestamps"][:].astype(np.float64)
        pos = bts["position/data"][:].astype(np.float64)
        lick = bts["lick/data"][:].astype(np.float64)
        speed = bts["speed/data"][:].astype(np.float64)
        trial = bts["trial number/data"][:].astype(np.int64)
        rzone = bts["reward_zone/data"][:].astype(np.float64)
        trial_start = bts["trial_start/data"][:].astype(np.float64)
        teleport = bts["teleport/data"][:].astype(np.float64)
        reward_t = bts["Reward/timestamps"][:].astype(np.float64)

        ophys = f["processing/ophys"]
        events = ophys["Deconvolved/plane0/data"][:].T.astype(np.float64)
        F = ophys["Fluorescence/plane0/data"][:].T.astype(np.float64)
        Fneu = ophys["Neuropil/plane0/data"][:].T.astype(np.float64)
        rate = float(ophys["Deconvolved/plane0/starting_time"].attrs["rate"])
        iscell = ophys["ImageSegmentation/PlaneSegmentation/iscell"][:]

    # --- ROI filtering -------------------------------------------------------
    cell_ids = np.arange(F.shape[0])
    if cells_only:
        keep = iscell[:, 0] > 0
        cell_ids = cell_ids[keep]
        events, F, Fneu = events[keep], F[keep], Fneu[keep]

    # --- dF/F ----------------------------------------------------------------
    Fc = F - dff_neuropil_coeff * Fneu
    f0 = np.percentile(Fc, 20, axis=1, keepdims=True)
    f0 = np.where(f0 <= 0, np.nanmean(np.abs(Fc), axis=1, keepdims=True) + 1e-9, f0)
    dff = (Fc - f0) / f0

    # --- per-trial table -----------------------------------------------------
    rows = []
    trial_ids = np.unique(trial[trial >= 0])
    for tr in trial_ids:
        idx = np.where(trial == tr)[0]
        i0, i1 = idx[0], idx[-1]
        in_rz = idx[rzone[idx] > 0]
        rz_pos = pos[in_rz[0]] if len(in_rz) else np.nan
        n_rew = int(np.sum((reward_t >= t[i0]) & (reward_t <= t[i1])))
        rows.append(
            dict(trial=int(tr), start_time=t[i0], stop_time=t[i1],
                 start_idx=int(i0), stop_idx=int(i1),
                 reward_zone_pos=rz_pos, zone_label=_zone_label(rz_pos),
                 n_rewards=n_rew)
        )
    trial_df = pd.DataFrame(rows)

    # fill zone label for un-entered trials from neighbors (design is blockwise)
    lab = trial_df["zone_label"].replace("", np.nan)
    trial_df["zone_label"] = lab.ffill().bfill()

    # --- switch detection ----------------------------------------------------
    is_switch = "_to_" in scene
    switch_trial = None
    if is_switch:
        labs = trial_df["zone_label"].values
        change = np.where(labs[1:] != labs[:-1])[0]
        if len(change):
            switch_trial = int(trial_df["trial"].iloc[change[0] + 1])
    trial_df["is_post_switch"] = (
        trial_df["trial"] >= switch_trial if switch_trial is not None else False
    )

    return dict(
        t=t, pos=pos, lick=lick, speed=speed, trial=trial,
        trial_start=trial_start, teleport=teleport, reward_times=reward_t,
        trial_df=trial_df,
        events=events, dff=dff, F=F, Fneu=Fneu,
        cell_ids=cell_ids, iscell=iscell, rate=rate,
        scene=scene, is_switch=is_switch, switch_trial=switch_trial,
        subject=subject, session=session, identifier=ident,
    )


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "nwb/sub-m11/sub-m11_ses-05_behavior+ophys.nwb"
    d = load_session(path)
    print(f"identifier   : {d['identifier']}")
    print(f"subject/ses  : {d['subject']} ses-{d['session']}  scene={d['scene']}")
    print(f"rate         : {d['rate']:.4f} Hz")
    print(f"t            : {d['t'].shape}  [{d['t'][0]:.2f}, {d['t'][-1]:.2f}] s")
    for k in ("pos", "lick", "speed", "events", "dff", "F", "Fneu", "cell_ids"):
        print(f"{k:13s}: {d[k].shape}")
    print(f"n trials     : {len(d['trial_df'])}")
    print(f"is_switch    : {d['is_switch']}  switch_trial={d['switch_trial']}")
    with pd.option_context("display.width", 120):
        print(d["trial_df"].head(4))
        print(d["trial_df"][d["trial_df"].is_post_switch].head(3))
    print("zone counts  :", d["trial_df"].zone_label.value_counts().to_dict())
    print("mean rz pos pre/post:",
          d["trial_df"][~d["trial_df"].is_post_switch].reward_zone_pos.mean(),
          d["trial_df"][d["trial_df"].is_post_switch].reward_zone_pos.mean())
