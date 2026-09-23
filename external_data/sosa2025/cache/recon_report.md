# Recon report — DANDI 001361 (Sosa, Plitt & Giocomo 2025, Nat Neurosci)

Reward-relocation dataset: VR linear track, 2P CA1 imaging, GCaMP7f.
Paper DOI: 10.1038/s41593-025-01985-4. Code: github.com/GiocomoLab/Sosa_et_al_2024
(cloned in `code_sosa/`, HEAD `adf0f9ed3defa22275d92fa0ed996f47008e05b3`).

Dandiset version pinned: **0.251124.0550** (published 2025-11-24).

## 1. Asset inventory (`cache/asset_list.json`)

- 152 assets, **92.45 GB** total. One NWB per subject x session: `sub-<id>/sub-<id>_ses-XX_behavior+ophys.nwb`.
- 11 subjects: m3, m4, m7, m11, m12, m13, m14, m15, m17, m18, m19. All have ses-01..ses-14
  except **m11 which has only ses-03..ses-14** (no scanning on its first 2 days — confirmed in
  `code_sosa/src/reward_relative/sessions_dict.py`, GCAMP11 entry: "no scanning these days").
- Per-subject totals: m11 **1.38 GB** (smallest), m19 4.36 GB, m14 4.78 GB, m13 5.70 GB,
  m15 6.04 GB, m12 7.07 GB, m7 8.90 GB, m17 10.73 GB, m3 11.94 GB, m4 14.99 GB, m18 16.55 GB.
- `cache/session_identifiers.json` maps every asset path to its NWB `identifier` and resolved S3 URL.

## 2. NWB file structure (streamed recon: m11 ses-03, m11 ses-05, m11 ses-10, m19 ses-01)

- `session_description` = "processed suite2p data" (uninformative).
- **`identifier` encodes the experimental condition**, e.g.
  `/data/InVivoDA/GCAMP11/25_02_2023/Env1_LocationA_to_C` — the trailing "scene" name is the
  session type (see section 3).
- **No `trials` table, no epochs, no stimulus group.** Everything behavioral lives in
  `processing/behavior/BehavioralTimeSeries` (all sampled per imaging frame, explicit
  `timestamps` in seconds, same clock as ophys):
  - `position` (cm; -500 during inter-trial "teleport" gap; track spans ~0-450)
  - `speed` (cm/s), `lick` (cumulative licks per frame, capacitive sensor)
  - `reward_zone` — **binary reward-zone-entry indicator** (this is how the per-trial reward
    zone location is recovered: position at first `reward_zone>0` sample of each trial)
  - `Reward` — event series (n_rewards,) of delivered volumes (mL) with `timestamps`
  - `trial number` (lap index; -1 outside trials), `trial_start`, `teleport` (binary trial
    start/end markers), `environment`, `autoreward`, `scanning`
- `processing/ophys`:
  - `Deconvolved/plane0` — **deconvolved events, shape (T, n_rois)**, rate 15.5078125 Hz
  - `Fluorescence/plane0` — raw F, same shape; `Neuropil/plane0` — neuropil F
  - `ImageSegmentation/PlaneSegmentation` — n_rois rows, columns `pixel_mask`, `iscell`
    (suite2p [is_cell, prob], e.g. m11 ses-05: 189/358 cells), `planeIdx`
  - `Backgrounds_0` — mean/max projection images
- `acquisition/TwoPhotonSeries` is a (1,1,1) stub (no raw movie). Imaging plane: CA1,
  GCaMP7f, 15.5078125 Hz, single plane.
- ROI counts vary per session (m11: 349-419+; m19 ses-01: 862) — **segmentation is
  independent per session; NWBs contain no cross-day ROI alignment**.

## 3. Session design / how switch sessions were identified

The NWB `identifier` scene name (cross-checked against `code_sosa/src/reward_relative/sessions_dict.py`,
which lists the same scenes by exp_day for every animal) gives the design. All subjects follow:

- ses-01, ses-02: `EnvX_LocationL` — familiar, fixed reward zone
- **ses-03, 05, 07: `EnvX_LocationL1_to_L2` — MID-SESSION reward relocation** (reward zone
  moves at ~trial 30 of ~80; `behavior.get_reward_zones` default `change_trial=30`)
- ses-04, 06: consolidation day at the new location
- **ses-08: `Env1_L1_to_Env2_L2` — environment switch + relocation**
- ses-09..14: same alternation (switch on odd, consolidation on even) in Env2

Reward zones (from `code_sosa/src/reward_relative/behavior.py:14`, keys X/Y/Z used for this
experiment's Locations A/B/C): **A = [80, 130] cm, B = [200, 250] cm, C = [320, 370] cm**
(zone start/stop; reward delivered on lick inside zone, autoreward at zone end if no lick).

Verified in-data (streamed, m11 ses-05 `Env1_LocationA_to_C`): per-trial position at first
`reward_zone>0` sample = ~80.8 cm for trials < 30, ~321.7 cm from trial 30 on. Matches design.

m11 session map (all subjects analogous; see `cache/session_identifiers.json` for the rest):

| ses | scene | type |
|-----|-------|------|
| 03 | Env1_LocationB_to_A | switch B→A |
| 04 | Env1_LocationA | post |
| 05 | Env1_LocationA_to_C | switch A→C |
| 06 | Env1_LocationC | post |
| 07 | Env1_LocationC_to_B | switch C→B |
| 08 | Env1_B_to_Env2_C | env switch |
| 09 | Env2_LocationC | post |
| 10 | Env2_LocationC_to_A | switch C→A |
| 11 | Env2_LocationA | post |
| 12 | Env2_LocationA_to_B | switch A→B |
| 13 | Env2_LocationB | post |
| 14 | Env2_LocationB_to_C | switch B→C |

## 4. What was downloaded (nwb/), and why

Budget: 20 GB cap for all of external_data/. Total downloaded here: **~2.34 GB**.

- **sub-m11: ALL 12 sessions (1.38 GB)** — smallest subject by far; gives every within-env
  switch, both consolidation flanks, and the env-switch day for one animal.
- **sub-m19: ses-04, ses-05, ses-06 (~0.96 GB)** — second-smallest subject with a complete
  ses-01..14 schedule; a clean pre (Env1_LocationB) / switch (Env1_LocationB_to_A) /
  post (Env1_LocationA) triad.

sha256 + sizes in `MANIFEST.json` (dandiset version pinned there too).

## 5. Do the NWBs contain everything needed? (figshare-pickle decision)

**Yes, for within-session reward-relative analyses.** The NWBs contain:
- deconvolved events per frame (the paper's RR criterion runs on these / on dF/F), plus raw F
  and neuropil for recomputing dF/F;
- full behavior (position, licks, speed, rewards) on the same clock;
- per-trial reward-zone position, recoverable exactly (reward_zone binary x position), and the
  switch trial is directly observable.

**Missing from the NWBs:** cross-day ROI alignment (each session segmented independently) and
the paper's precomputed intermediate outputs (place-cell masks, trial matrices, RR cell IDs).
Per `code_sosa/docs/data_download.md`, figshare hosts two products:
`multi_anim_sess` preprocessed pickles (~200 GB, doi 10.25452/figshare.plus.27098065) and
`multiDayData` postprocessed pickles (11 GB switch-days / 22 GB all days,
doi 10.25452/figshare.plus.27138633, **reward-relative cells already identified**). The
11-GB switch-day multiDayData would only be needed for (a) same-cell tracking across days
(it holds the cross-day ROI alignment results) or (b) reproducing paper numbers without
recomputing. For the within-session remap-to-reward analysis it is NOT needed.
Decision: skip for now. (Note: their README links older DANDI version 0.250609.2249; we
pinned the current published 0.251124.0550.)

## 6. Where the reward-relative cell criterion lives (code_sosa)

Package `src/reward_relative/` (installed name `reward_relative`). Key locations:

- **`src/reward_relative/dayData.py::dayData.add_circ_relative_peaks`** (line 923) — the
  criterion. Steps:
  1. Convert linear position to circular (`spatial.pos_cm_to_rad`), build spatially binned
     trial matrices of activity (`ts_key`, default deconvolved 'events'; ~10-cm → π/22.5 bins).
  2. Split trials into `trial_set0` (pre-switch reward zone) / `trial_set1` (post-switch), via
     `behavior.define_trial_subsets` (line 160 of behavior.py; zone labels from
     `behavior.get_reward_zones`).
  3. Peak of trial-averaged map per set; express peaks **relative to reward-zone start**:
     `rel_peaks = circ.wrap(peak - circ_rzone)`.
  4. Candidate RR cells: pre vs post reward-relative peaks within `reward_dist_inclusive`
     = **50 cm** (converted to radians, `rdist_to_rad_inc`) of each other — i.e. near the
     unity line — via helper `_mean_rel_dist` (dayData.py line 1308). Null: post-switch maps
     circularly rolled (n_perm=1000).
  5. Confirmation: `calc_field_xcorr` (dayData.py line 1353) — cross-correlation of pre/post
     fields vs 500 circular shuffles; keep cells whose above-shuffle xcorr peak is within
     `xcorr_thr=5` bins (10-cm bins → 50 cm) of the reward-relative shift.
  6. Result: `self.reward_rel_cell_ids[an]`. Prerequisite: `overall_place_cell_masks`
     (place cells in either trial set; built upstream in the class from spatial-information
     criteria; defaults `reward_dist_exclusive = reward_dist_inclusive = 50` set at
     dayData.py lines 257-258).
- Remap-type taxonomy ('rr', 'track', 'appear', 'disappear', 'reward', 'nonreward_remap'):
  `dayData.get_cell_classes` (line 646) and usage around lines 2101-2169.
- Reward position extraction from raw session: `src/reward_relative/rewardAnalysis.py::
  get_reward_times` (line 37) / `get_reward_pos` (line 64) — same logic our loader mirrors
  against the NWB `reward_zone`/`Reward` series.
- Place-cell prerequisite: `dayData.filter_cell_subset` (line 589) — OR of per-trial-set
  place-cell masks (computed upstream in their preprocessing via spatial-information
  shuffle), excluding putative interneurons (`spatial.is_putative_interneuron`).
- Notebooks: `notebooks/make_multiDayData_class.ipynb` runs the criterion
  (`add_circ_relative_peaks=True`); `notebooks/Fig2_Ext2_identifyRewardRelative.ipynb`
  is the paper figure built on `reward_rel_cell_ids`.

## 7. Loader

`nwb_loader.py` in this directory: `load_session(path) -> dict` with keys
`t, pos, lick, speed, trial_df (per-trial start/stop times + reward_zone_pos + zone label),
dff, events, F, Fneu, cell_ids, iscell, rate, scene, subject, session`.
Verification output at bottom of this file's history / in the final report.
