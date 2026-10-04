"""Reanalysis of Sosa, Plitt & Giocomo 2025 (DANDI 001361) switch sessions.

Tests (per within-environment reward-relocation session):
  i   Apparent vs occupancy-matched shift of place-field peaks toward the new
      reward zone (raw vs per-(position-bin x speed-tercile) frame-matched).
  ii  Reward-relative fraction using their criterion (circular rel-peak within
      50 cm of unity + circular-roll null; simplified: 500 rolls of the post
      map, RR iff |rel_pre - rel_post| < 50 cm and < 5th pct of the cell's
      null distances), plus place-anchored fraction.
  iii Latencies: behavioural retargeting (anticipatory licking) vs map
      transfer (place-cell peaks near the new zone), in trials after switch.

Sessions: all downloaded NWBs whose scene contains "_to_" within a single
environment (env-switch sessions like Env1_B_to_Env2_C are skipped).
That is m11 ses-03/05/07/10/12/14 + m19 ses-05 (NB: m11 lacks ses-01/02, so
its Env2 switch days fall on even session numbers).

All outputs cached to cache/sosa_tests.json. Run from this directory:
    python analysis_sosa.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.ndimage import gaussian_filter1d
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
# Code lives here; the NWB recordings, MANIFEST.json and JSON cache live in
# data/sosa2025/ (NWB recordings are git-ignored, fetched from DANDI 001361).
DATA = HERE.parents[2] / "data" / "sosa2025"
sys.path.insert(0, str(HERE))
from nwb_loader import ZONES, load_session  # noqa: E402

# ---------------------------------------------------------------- parameters
TRACK_LEN = 450.0          # cm, circular
BIN_CM = 5.0
NBINS = int(TRACK_LEN / BIN_CM)          # 90
SPEED_THR = 2.0            # cm/s, frame inclusion
SMOOTH_SIGMA = 1.0         # bins, circular Gaussian
WARMUP_PRE = 10            # skip first 10 trials of the session (warmup)
ADAPT_POST = 5             # skip 5 trials right after the switch
N_SI_SHUF = 100            # spatial-information circular-roll shuffles
SI_PCT = 95.0
N_RR_NULL = 500            # circular rolls of post map (their style, n_perm)
RR_DIST_CM = 50.0          # their reward_dist_inclusive
NEAR_CM = 50.0             # test-iii "near zone" distance (to zone start)
ANTICIP_CM = 50.0          # lick window: 50 cm before zone start
MIN_PLACE_CELLS = 30
N_MATCH_REPEATS = 10       # subsample repeats for occupancy matching
OCC_FLOOR_S = 1.0          # min occupancy (s) per bin for aggregate rate maps;
                           # sub-floor bins are NaN'd and circularly interpolated
MIN_WINDOW_TRIALS = 5      # test-iii sliding map window minimum
SEED = 0

EXAMPLE_SESSION = ("m11", 5)   # figure panel (a)

# Model reward-relative fraction (mean, SD over networks), read from the model cache so the
# data-vs-model test always uses the current model run; the constants are only a fallback.
_RR_CACHE = HERE.parents[2] / "results" / "multi_seed_reloc" / "aggregate" / "reward_relative_summary.json"


def _model_rr_band(fallback=(0.064, 0.031)):
    try:
        rows = json.loads(_RR_CACHE.read_text())["per_seed"]
        v = np.array([r["controlled"]["frac_reward_relative"] for r in rows], dtype=float)
        return float(v.mean()), float(v.std(ddof=1))
    except Exception as e:
        print(f"[warn] model reward-relative cache unreadable ({e}); using the fallback {fallback}", file=sys.stderr)
        return fallback


MODEL_RR_FRAC, MODEL_RR_SD = _model_rr_band()

NOTE_TEST_I = (
    "On a stereotyped 1D VR track, laps are nearly identical trajectories, so "
    "pre/post occupancy (position x speed) is already nearly matched. The model "
    "therefore PREDICTS attenuation A ~ small here, unlike in the open field "
    "where marginal occupancy matching removed most of the naive sampling bias. "
    "A small A in this dataset is consistent with the model, not evidence "
    "against occupancy confounds in 2D."
)

NOTE_ATTENUATION = (
    "The per-session attenuation index A = 1 - delta_matched/delta_raw is "
    "ill-conditioned whenever delta_raw ~ 0 (which is itself the predicted "
    "outcome on stereotyped laps); prefer attenuation_pooled = "
    "1 - mean(delta_matched)/mean(delta_raw) and the paired raw-vs-matched "
    "Wilcoxon."
)

NOTE_MATCHING = (
    "Caveat discovered during matching: pre vs post speed distributions differ "
    "most near the reward zones (anticipatory slowing at the NEW zone appears "
    "only post-switch), so (position-bin x speed-tercile) min-count matching "
    "leaves almost no frames precisely there (see n_lowocc_bins_*). Without an "
    "occupancy floor, near-zero-occupancy bins produce spuriously high rate "
    "estimates and artifactual 'toward-reward' matched shifts (e.g. m11 ses-05: "
    "delta_matched -0.14 without floor vs -0.04 with a 1-s floor). Aggregate "
    "maps therefore NaN bins with < OCC_FLOOR_S seconds of occupancy per "
    "condition and interpolate them circularly. This is a structural limitation "
    "of marginal occupancy matching on tasks where behaviour covaries with the "
    "variable of interest at the measurement location."
)


# ------------------------------------------------------------- circular math
def circ_signed(a, b, L=TRACK_LEN):
    """Signed circular difference a-b in (-L/2, L/2]."""
    return (np.asarray(a) - np.asarray(b) + L / 2.0) % L - L / 2.0


def circ_dist(a, b, L=TRACK_LEN):
    return np.abs(circ_signed(a, b, L))


def smooth_circ(maps, sigma=SMOOTH_SIGMA):
    """Circular Gaussian smoothing along the last axis; NaNs interpolated."""
    m = np.array(maps, dtype=np.float64)
    nan = ~np.isfinite(m)
    if nan.any():
        # occupancy-weighted NaN fill: smooth values & weights with wrap mode
        w = (~nan).astype(np.float64)
        m0 = np.where(nan, 0.0, m)
        num = gaussian_filter1d(m0, 2.0, axis=-1, mode="wrap")
        den = gaussian_filter1d(w, 2.0, axis=-1, mode="wrap")
        m = np.where(nan, num / np.maximum(den, 1e-12), m)
    return gaussian_filter1d(m, sigma, axis=-1, mode="wrap")


BIN_CENTERS = (np.arange(NBINS) + 0.5) * BIN_CM


def peaks_cm(maps):
    """Peak position (bin centre, cm) along last axis."""
    return BIN_CENTERS[np.argmax(maps, axis=-1)]


# ------------------------------------------------------------- map machinery
def bin_index(pos):
    return (np.floor(pos / BIN_CM).astype(np.int64)) % NBINS


def one_hot(binidx):
    n = len(binidx)
    return sparse.csr_matrix(
        (np.ones(n), (np.arange(n), binidx)), shape=(n, NBINS))


def rate_from_counts(evsum, occ, dt, occ_floor_s=0.0):
    """Rate map from binned event sums + occupancy counts.

    Bins with less than occ_floor_s seconds of occupancy are NaN'd (to be
    circularly interpolated by smooth_circ) instead of producing unstable
    events/tiny-occupancy estimates.
    """
    floor = max(occ_floor_s / dt, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.asarray(evsum) / (occ[None, :] * dt)
    rate[:, occ <= floor] = np.nan
    return rate


def aggregate_map(events, binidx, dt, occ_floor_s=0.0):
    """Occupancy-normalised event-rate map from a set of frames.

    events: (n_cells, n_frames); returns (n_cells, NBINS), NaN below floor.
    """
    occ = np.bincount(binidx, minlength=NBINS).astype(np.float64)
    evsum = events @ one_hot(binidx)                     # (n_cells, NBINS)
    return rate_from_counts(evsum, occ, dt, occ_floor_s), occ


def per_trial_maps(events, pos, speed, trial, trial_numbers, dt):
    """(n_trials, n_cells, NBINS) rate maps, NaN in unvisited bins."""
    out = np.full((len(trial_numbers), events.shape[0], NBINS), np.nan,
                  dtype=np.float32)
    for i, tr in enumerate(trial_numbers):
        sel = (trial == tr) & (pos >= 0) & (speed >= SPEED_THR)
        if sel.sum() < 3:
            continue
        m, _ = aggregate_map(events[:, sel], bin_index(pos[sel]), dt)
        out[i] = m
    return out


def trial_avg_map(ptm, rows):
    """nanmean over trial rows, then circular smooth."""
    with np.errstate(invalid="ignore"):
        m = np.nanmean(ptm[rows], axis=0)
    return smooth_circ(m)


# ------------------------------------------------- spatial information gate
def spatial_information(evsum, occ, dt):
    """Skaggs bits/event per cell from binned event sums + occupancy counts."""
    p = occ / occ.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        r_b = np.where(occ[None, :] > 0, evsum / (occ[None, :] * dt), 0.0)
    r = (r_b * p[None, :]).sum(axis=1)
    ratio = np.where(r[:, None] > 0, r_b / np.maximum(r[:, None], 1e-300), 0.0)
    term = np.where(ratio > 0, p[None, :] * ratio * np.log2(ratio,
                    where=ratio > 0, out=np.zeros_like(ratio)), 0.0)
    return term.sum(axis=1)


def si_gate(events_set, binidx, dt, rng):
    """True SI + boolean pass mask vs N_SI_SHUF within-set circular rolls."""
    n_frames = binidx.shape[0]
    occ = np.bincount(binidx, minlength=NBINS).astype(np.float64)
    B = one_hot(binidx)
    si_true = spatial_information(np.asarray(events_set @ B), occ, dt)

    n_cells = events_set.shape[0]
    min_off = max(int(0.05 * n_frames), 16)
    si_null = np.zeros((N_SI_SHUF, n_cells))
    base = np.arange(n_frames, dtype=np.int64)
    for s in range(N_SI_SHUF):
        off = rng.integers(min_off, n_frames - min_off, size=n_cells)
        idx = (base[None, :] + off[:, None]) % n_frames
        ev_shuf = np.take_along_axis(events_set, idx, axis=1)
        si_null[s] = spatial_information(np.asarray(ev_shuf @ B), occ, dt)
    thr = np.percentile(si_null, SI_PCT, axis=0)
    return si_true, (si_true > thr)


# ------------------------------------------------------- occupancy matching
def matched_frame_masks(binidx0, speed0, binidx1, speed1, rng):
    """Min-count matching in (position-bin x speed-tercile) strata.

    Returns boolean masks over pre frames and post frames."""
    pooled = np.concatenate([speed0, speed1])
    edges = np.percentile(pooled, [100 / 3, 200 / 3])
    t0 = np.digitize(speed0, edges)      # 0/1/2 tercile
    t1 = np.digitize(speed1, edges)
    s0 = binidx0 * 3 + t0
    s1 = binidx1 * 3 + t1
    keep0 = np.zeros(len(s0), dtype=bool)
    keep1 = np.zeros(len(s1), dtype=bool)
    for stratum in np.union1d(s0, s1):
        i0 = np.where(s0 == stratum)[0]
        i1 = np.where(s1 == stratum)[0]
        m = min(len(i0), len(i1))
        if m == 0:
            continue
        keep0[rng.choice(i0, size=m, replace=False)] = True
        keep1[rng.choice(i1, size=m, replace=False)] = True
    return keep0, keep1


# ------------------------------------------------------------------- licking
def corrected_binary_licks(lick, trial_df, trial):
    """Binarise licks; NaN out trials with a putatively stuck sensor
    (their behavior.correct_lick_sensor_error, thr=0.5)."""
    licks = lick.astype(np.float64).copy()
    bad_trials = []
    for _, row in trial_df.iterrows():
        i0, i1 = int(row.start_idx), int(row.stop_idx) + 1
        seg = licks[i0:i1]
        if len(seg) and np.mean(seg > 2) > 0.5:
            licks[i0:i1] = np.nan
            bad_trials.append(int(row.trial))
    licks = np.where(np.isnan(licks), np.nan, (licks > 0).astype(np.float64))
    return licks, bad_trials


def anticipatory_lick_rate(licks, pos, trial, tr, zone_start, dt):
    """Lick rate (Hz) in the 50 cm before zone_start on trial tr; NaN if the
    trial's sensor was flagged or the window was never visited."""
    sel = (trial == tr) & (pos >= 0)
    if not sel.any():
        return np.nan
    d = circ_signed(zone_start, pos[sel])           # cm ahead to zone start
    win = (d > 0) & (d <= ANTICIP_CM)
    if win.sum() == 0:
        return np.nan
    lw = licks[sel][win]
    if np.isnan(lw).any():
        return np.nan
    return float(np.nansum(lw) / (win.sum() * dt))


# ------------------------------------------------------------------ sessions
def find_switch_sessions():
    manifest = json.loads((DATA / "MANIFEST.json").read_text())
    sessions, skipped = [], []
    for f in manifest["files"]:
        scene = f["scene"]
        if "_to_" not in scene:
            continue
        if scene.count("Env") > 1:
            skipped.append(dict(path=f["local_path"], scene=scene,
                                reason="environment-switch session"))
            continue
        sessions.append(DATA / f["local_path"])
    return sorted(sessions), skipped


# ------------------------------------------------------------- per session
def analyze_session(path, sess_seed):
    rng = np.random.default_rng(SEED + 1000 * sess_seed)
    d = load_session(str(path))
    events, pos, speed, lick = d["events"], d["pos"], d["speed"], d["lick"]
    trial, tdf, dt = d["trial"], d["trial_df"], 1.0 / d["rate"]
    scene = d["scene"]

    # ---- sanity ------------------------------------------------------------
    in_trial = pos >= 0
    assert 440.0 < pos[in_trial].max() < 470.0, "unexpected track length"
    assert d["is_switch"] and d["switch_trial"] is not None
    n_trials = len(tdf)
    assert 50 <= n_trials <= 130, f"odd trial count {n_trials}"

    trial_numbers = tdf["trial"].to_numpy()
    switch_ord = int(np.where(tdf["is_post_switch"].to_numpy())[0][0])
    assert 20 <= switch_ord <= 45, f"odd switch ordinal {switch_ord}"

    zone_pre = tdf["zone_label"].iloc[0]
    zone_post = tdf["zone_label"].iloc[-1]
    assert zone_pre != zone_post and zone_pre in ZONES and zone_post in ZONES
    rz_pre = ZONES[zone_pre]          # (start, stop) cm
    rz_post = ZONES[zone_post]
    new_center = 0.5 * (rz_post[0] + rz_post[1])

    pre_ord = np.arange(WARMUP_PRE, switch_ord)
    post_ord = np.arange(switch_ord + ADAPT_POST, n_trials)
    pre_trials = trial_numbers[pre_ord]
    post_trials = trial_numbers[post_ord]
    assert len(pre_trials) >= 10 and len(post_trials) >= 10

    row = dict(subject=d["subject"], session=d["session"], scene=scene,
               zone_pre=zone_pre, zone_post=zone_post,
               switch_trial=int(trial_numbers[switch_ord]),
               n_trials=int(n_trials),
               n_pre_trials=int(len(pre_trials)),
               n_post_trials=int(len(post_trials)),
               n_cells=int(events.shape[0]))

    # ---- frame sets ----------------------------------------------------------
    valid = in_trial & (speed >= SPEED_THR)
    sel_pre = valid & np.isin(trial, pre_trials)
    sel_post = valid & np.isin(trial, post_trials)
    bi_pre = bin_index(pos[sel_pre])
    bi_post = bin_index(pos[sel_post])
    ev_pre = events[:, sel_pre]
    ev_post = events[:, sel_post]

    # ---- 1. tuning curves + place-cell gate ---------------------------------
    ptm = per_trial_maps(events, pos, speed, trial, trial_numbers, dt)
    map_pre = trial_avg_map(ptm, pre_ord)     # (n_cells, NBINS), smoothed
    map_post = trial_avg_map(ptm, post_ord)

    si_pre, pass_pre = si_gate(ev_pre, bi_pre, dt, rng)
    si_post, pass_post = si_gate(ev_post, bi_post, dt, rng)
    active_both = (np.nanmax(map_pre, axis=1) > 0) & \
                  (np.nanmax(map_post, axis=1) > 0)
    place = (pass_pre | pass_post) & active_both
    n_place = int(place.sum())
    row.update(n_place_cells=n_place,
               n_pass_pre=int(pass_pre.sum()), n_pass_post=int(pass_post.sum()))
    if n_place < MIN_PLACE_CELLS:
        row["skipped"] = f"only {n_place} gated place cells (<{MIN_PLACE_CELLS})"
        return row, None

    pk_pre = peaks_cm(map_pre[place])
    pk_post = peaks_cm(map_post[place])

    # ---- test i: raw vs occupancy-matched shift toward new zone -------------
    # raw + matched both use aggregate (frame-level) maps with the same
    # occupancy floor, so the matching is the only difference between the two
    # estimates. Matched counts are pooled across subsample repeats before
    # normalising (mean-of-noisy-maps / per-repeat peaks are both biased).
    agg_pre, occ_pre = aggregate_map(ev_pre, bi_pre, dt, OCC_FLOOR_S)
    agg_post, occ_post = aggregate_map(ev_post, bi_post, dt, OCC_FLOOR_S)
    apre = peaks_cm(smooth_circ(agg_pre[place]))
    apost = peaks_cm(smooth_circ(agg_post[place]))
    halfL = TRACK_LEN / 2.0
    shift_raw = (circ_dist(apost, new_center) -
                 circ_dist(apre, new_center)) / halfL   # neg = toward new rz
    delta_raw = float(np.mean(shift_raw))

    evp_pre, evp_post = ev_pre[place], ev_post[place]
    evs0 = np.zeros((n_place, NBINS)); mocc0 = np.zeros(NBINS)
    evs1 = np.zeros((n_place, NBINS)); mocc1 = np.zeros(NBINS)
    kept0 = kept1 = 0
    for _ in range(N_MATCH_REPEATS):
        k0, k1 = matched_frame_masks(bi_pre, speed[sel_pre],
                                     bi_post, speed[sel_post], rng)
        kept0 += k0.sum(); kept1 += k1.sum()
        evs0 += evp_pre[:, k0] @ one_hot(bi_pre[k0])
        evs1 += evp_post[:, k1] @ one_hot(bi_post[k1])
        mocc0 += np.bincount(bi_pre[k0], minlength=NBINS)
        mocc1 += np.bincount(bi_post[k1], minlength=NBINS)
    # pooled counts over repeats -> floor is OCC_FLOOR_S per repeat
    floor_pooled = OCC_FLOOR_S * N_MATCH_REPEATS
    mpre = peaks_cm(smooth_circ(rate_from_counts(evs0, mocc0, dt, floor_pooled)))
    mpost = peaks_cm(smooth_circ(rate_from_counts(evs1, mocc1, dt, floor_pooled)))
    n_lowocc0 = int(np.sum(mocc0 * dt / N_MATCH_REPEATS < OCC_FLOOR_S))
    n_lowocc1 = int(np.sum(mocc1 * dt / N_MATCH_REPEATS < OCC_FLOOR_S))
    shift_matched = (circ_dist(mpost, new_center) -
                     circ_dist(mpre, new_center)) / halfL
    delta_matched = float(np.mean(shift_matched))
    atten = (1.0 - delta_matched / delta_raw) if abs(delta_raw) > 1e-3 else np.nan

    # trial-averaged-map cross-check of the raw shift
    delta_raw_trialavg = float(np.mean(
        (circ_dist(pk_post, new_center) - circ_dist(pk_pre, new_center)) / halfL))

    assert abs(delta_raw) <= 1.0 and abs(delta_matched) <= 1.0
    row.update(
        delta_raw=delta_raw, delta_matched=delta_matched,
        attenuation=None if np.isnan(atten) else float(atten),
        delta_raw_trialavg=delta_raw_trialavg,
        match_retention_pre=float(kept0 / (N_MATCH_REPEATS * len(bi_pre))),
        match_retention_post=float(kept1 / (N_MATCH_REPEATS * len(bi_post))),
        n_lowocc_bins_pre=n_lowocc0, n_lowocc_bins_post=n_lowocc1,
    )

    # ---- test ii: reward-relative fraction (their criterion, simplified) ----
    rel_pre = circ_signed(pk_pre, rz_pre[0])
    rel_post = circ_signed(pk_post, rz_post[0])
    d_rel = circ_dist(rel_post, rel_pre)

    # null: circular roll of the post map by k bins moves its peak by exactly
    # k bins, so the rolled-map peak is (peak_bin + k) % NBINS (their n_perm
    # roll null, computed analytically).
    shifts = rng.integers(0, NBINS, size=(N_RR_NULL, len(pk_post)))
    pk_null = BIN_CENTERS[
        ((np.argmax(map_post[place], axis=1)[None, :] + shifts) % NBINS)]
    d_null = circ_dist(circ_signed(pk_null, rz_post[0]), rel_pre[None, :])
    null_p5 = np.percentile(d_null, 5, axis=0)

    is_rr = (d_rel < RR_DIST_CM) & (d_rel < null_p5)
    is_rr50 = d_rel < RR_DIST_CM
    is_anchored = circ_dist(pk_pre, pk_post) < RR_DIST_CM
    rr_frac = float(np.mean(is_rr))
    row.update(rr_frac=rr_frac,
               rr_frac_50cm_only=float(np.mean(is_rr50)),
               place_anchored_frac=float(np.mean(is_anchored)),
               n_rr=int(is_rr.sum()), n_rr_50cm=int(is_rr50.sum()),
               n_anchored=int(is_anchored.sum()))
    for k in ("rr_frac", "rr_frac_50cm_only", "place_anchored_frac"):
        assert 0.0 <= row[k] <= 1.0

    # ---- test iii: lick latency vs map latency ------------------------------
    licks_bin, bad_lick_trials = corrected_binary_licks(lick, tdf, trial)
    lick_lat = None
    lick_rates = []
    for j in range(switch_ord, n_trials):
        tr = trial_numbers[j]
        r_new = anticipatory_lick_rate(licks_bin, pos, trial, tr, rz_post[0], dt)
        r_old = anticipatory_lick_rate(licks_bin, pos, trial, tr, rz_pre[0], dt)
        lick_rates.append(dict(trial=int(tr), rate_new=None if np.isnan(r_new)
                               else r_new, rate_old=None if np.isnan(r_old)
                               else r_old))
        if lick_lat is None and np.isfinite(r_new) and np.isfinite(r_old) \
                and r_new > r_old:
            lick_lat = j - switch_ord + 1

    # map transfer: sliding window switch..t (>= MIN_WINDOW_TRIALS trials)
    pre_ref = float(np.mean(circ_dist(pk_pre, rz_pre[0]) < NEAR_CM))
    map_lat = None
    map_fracs = []
    place_idx = np.where(place)[0]
    for j in range(switch_ord + MIN_WINDOW_TRIALS - 1, n_trials):
        win = np.arange(switch_ord, j + 1)
        wmap = trial_avg_map(ptm[:, place_idx, :], win)
        act = np.nanmax(wmap, axis=1) > 0
        if act.sum() == 0:
            map_fracs.append(dict(trial_end=int(trial_numbers[j]), frac=None))
            continue
        frac = float(np.mean(
            circ_dist(peaks_cm(wmap[act]), rz_post[0]) < NEAR_CM))
        map_fracs.append(dict(trial_end=int(trial_numbers[j]), frac=frac))
        if map_lat is None and frac > pre_ref:
            map_lat = j - switch_ord + 1

    if lick_lat is not None:
        assert lick_lat >= 1
    if map_lat is not None:
        assert map_lat >= MIN_WINDOW_TRIALS
    row.update(lick_latency=lick_lat, map_latency=map_lat,
               map_pre_ref_frac=pre_ref,
               n_bad_lick_trials=len(bad_lick_trials))

    # ---- example-session payload for the figure ------------------------------
    payload = None
    if (d["subject"], d["session"]) == EXAMPLE_SESSION:
        lickmat = np.full((n_trials, NBINS), np.nan)
        for j, tr in enumerate(trial_numbers):
            sel = (trial == tr) & (pos >= 0)
            if not sel.any():
                continue
            lw = licks_bin[sel]
            if np.isnan(lw).any():
                continue
            bi = bin_index(pos[sel])
            occ = np.bincount(bi, minlength=NBINS).astype(float)
            cnt = np.bincount(bi, weights=lw, minlength=NBINS)
            with np.errstate(invalid="ignore", divide="ignore"):
                lr = cnt / (occ * dt)
            lr[occ == 0] = np.nan
            lickmat[j] = lr
        payload = dict(subject=d["subject"], session=d["session"], scene=scene,
                       lick_matrix=np.round(np.nan_to_num(lickmat, nan=-1.0),
                                            3).tolist(),
                       bin_centers=BIN_CENTERS.tolist(),
                       switch_ordinal=switch_ord,
                       zone_pre=list(rz_pre), zone_post=list(rz_post),
                       zone_pre_label=zone_pre, zone_post_label=zone_post)

    row["lick_rates_post"] = lick_rates
    row["map_fracs_post"] = map_fracs
    return row, payload


# ------------------------------------------------------------------ summary
def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, float) and np.isnan(x):
        return None
    return x


def summarize(rows):
    ok = [r for r in rows if "skipped" not in r]
    out = dict(n_sessions_analyzed=len(ok),
               n_sessions_skipped=len(rows) - len(ok))

    def stat(key):
        v = np.array([r[key] for r in ok if r.get(key) is not None],
                     dtype=float)
        return dict(mean=float(v.mean()) if len(v) else None,
                    sd=float(v.std(ddof=1)) if len(v) > 1 else None,
                    n=int(len(v)),
                    values=v.tolist())

    for key in ("delta_raw", "delta_matched", "attenuation",
                "delta_raw_trialavg", "rr_frac", "rr_frac_50cm_only",
                "place_anchored_frac", "lick_latency", "map_latency"):
        out[key] = stat(key)

    def paired_wilcoxon(k1, k2):
        pairs = [(r[k1], r[k2]) for r in ok
                 if r.get(k1) is not None and r.get(k2) is not None]
        if len(pairs) < 6:
            return dict(n=len(pairs), stat=None, p=None,
                        note="n<6, Wilcoxon not run")
        a, b = map(np.array, zip(*pairs))
        if np.allclose(a, b):
            return dict(n=len(pairs), stat=None, p=None, note="all ties")
        s, p = wilcoxon(a, b)
        return dict(n=len(pairs), stat=float(s), p=float(p))

    def onesample_wilcoxon(key, mu=0.0):
        v = np.array([r[key] for r in ok if r.get(key) is not None]) - mu
        if len(v) < 6:
            return dict(n=len(v), stat=None, p=None, note="n<6")
        s, p = wilcoxon(v)
        return dict(n=int(len(v)), stat=float(s), p=float(p), mu=mu)

    out["wilcoxon_raw_vs_matched"] = paired_wilcoxon("delta_raw",
                                                     "delta_matched")
    out["wilcoxon_raw_vs_zero"] = onesample_wilcoxon("delta_raw")
    out["wilcoxon_lick_vs_map_latency"] = paired_wilcoxon("lick_latency",
                                                          "map_latency")
    out["wilcoxon_rrfrac_vs_model"] = onesample_wilcoxon("rr_frac",
                                                         MODEL_RR_FRAC)
    mr = out["delta_raw"]["mean"]
    mm = out["delta_matched"]["mean"]
    out["attenuation_pooled"] = (
        None if (mr is None or abs(mr) < 1e-3) else float(1.0 - mm / mr))
    out["model_rr_frac"] = MODEL_RR_FRAC
    out["model_rr_sd"] = MODEL_RR_SD
    out["note_test_i"] = NOTE_TEST_I
    out["note_attenuation"] = NOTE_ATTENUATION
    out["note_matching"] = NOTE_MATCHING
    out["prediction4_lick_before_map"] = dict(
        n_lick_faster=int(sum(1 for r in ok
                              if r.get("lick_latency") is not None
                              and r.get("map_latency") is not None
                              and r["lick_latency"] < r["map_latency"])),
        n_pairs=int(sum(1 for r in ok
                        if r.get("lick_latency") is not None
                        and r.get("map_latency") is not None)),
        caveat=(f"map latency is lower-bounded at {MIN_WINDOW_TRIALS} trials "
                "by the minimum sliding-window size"))
    return out


def main():
    sessions, skipped_env = find_switch_sessions()
    print(f"Found {len(sessions)} within-env switch sessions "
          f"({len(skipped_env)} env-switch sessions excluded)")
    rows, example = [], None
    for i, path in enumerate(sessions):
        print(f"[{i + 1}/{len(sessions)}] {path.name} ...", flush=True)
        row, payload = analyze_session(path, i)
        rows.append(row)
        if payload is not None:
            example = payload
        msg = row.get("skipped") or (
            f"place={row['n_place_cells']}/{row['n_cells']} "
            f"d_raw={row['delta_raw']:+.3f} d_match={row['delta_matched']:+.3f} "
            f"rr={row['rr_frac']:.3f} lick_lat={row['lick_latency']} "
            f"map_lat={row['map_latency']}")
        print("   ", row["subject"], f"ses-{row['session']:02d}",
              row["scene"], "|", msg, flush=True)

    result = dict(
        params=dict(track_len_cm=TRACK_LEN, bin_cm=BIN_CM, nbins=NBINS,
                    speed_thr=SPEED_THR, smooth_sigma_bins=SMOOTH_SIGMA,
                    warmup_pre_trials=WARMUP_PRE, adapt_post_trials=ADAPT_POST,
                    n_si_shuffles=N_SI_SHUF, si_pct=SI_PCT,
                    n_rr_null=N_RR_NULL, rr_dist_cm=RR_DIST_CM,
                    near_cm=NEAR_CM, anticip_cm=ANTICIP_CM,
                    min_place_cells=MIN_PLACE_CELLS,
                    n_match_repeats=N_MATCH_REPEATS,
                    min_window_trials=MIN_WINDOW_TRIALS, seed=SEED),
        sessions_excluded_env_switch=skipped_env,
        per_session=rows,
        summary=summarize(rows),
        example_session=example,
    )
    out = DATA / "cache" / "sosa_tests.json"
    out.write_text(json.dumps(_clean(result), indent=1))
    print(f"\nWrote {out}")

    s = result["summary"]
    print(f"\nTEST i : delta_raw = {s['delta_raw']['mean']:+.4f} "
          f"± {s['delta_raw']['sd']:.4f}, delta_matched = "
          f"{s['delta_matched']['mean']:+.4f} ± {s['delta_matched']['sd']:.4f}, "
          f"attenuation A = {s['attenuation']['mean']:.3f}")
    print(f"TEST ii: RR fraction = {s['rr_frac']['mean']:.3f} "
          f"± {s['rr_frac']['sd']:.3f} (model {MODEL_RR_FRAC}±{MODEL_RR_SD}); "
          f"50cm-only = {s['rr_frac_50cm_only']['mean']:.3f}; "
          f"anchored = {s['place_anchored_frac']['mean']:.3f}")
    print(f"TEST iii: lick latency = {s['lick_latency']['mean']:.1f} "
          f"± {s['lick_latency']['sd']:.1f} trials, map latency = "
          f"{s['map_latency']['mean']:.1f} ± {s['map_latency']['sd']:.1f}; "
          f"Wilcoxon p = {s['wilcoxon_lick_vs_map_latency']['p']}")


if __name__ == "__main__":
    main()
