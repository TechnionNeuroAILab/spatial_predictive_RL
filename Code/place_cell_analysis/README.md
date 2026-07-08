# Place-cell analysis — *Reward-induced reshaping of the hippocampal predictive manifold*

Analysis code for the two-stage model (predictive RNN + RL controller) described in
*Predictive_Manifold_Rewards_Reshaping*. The scripts turn the model's saved
rate-map / behaviour / activation data into the paper's figures and statistics:
whether reward reshapes the cognitive map, and whether that reshaping is a genuine
weight-level change or merely a behavioural sampling overlay.

---

## 1. The experiment in one paragraph

A recurrent network is trained purely to **predict its next sensory input** and
develops place-cell-like tuning. An RL **controller** then learns to navigate to a
reward by reading the RNN's hidden state, while an interleaved predictive objective
holds the map stable. The central control is a **frozen-weight experience replay**:
one identical random-walk trajectory is passed through the pre-RL and post-RL
weights (`replay_c1_pred` → `replay_c1_postrl`), so occupancy is held fixed and any
change in the fields is attributable to the weights alone. There are two runs:

| Run | `reward_pred_head` | reward location | cycles | Role |
|-----|--------------------|-----------------|--------|------|
| **`Positive_Results`** | **on** | off-centre **(0.3, 0.3)** | 1 | reward-prediction objective → expect **genuine remapping** toward reward |
| **`Negative_Results`** | off | centre **(0.5, 0.5)** | 5 | task-RL only → **negative control**, expect **no** weight-level remapping |

The headline result is a **dissociation**: during goal-directed behaviour *both* runs
look like they over-represent reward, but under the frozen replay only the
reward-head run (`Positive`) actually reshapes the map.

---

## 2. Script inventory

All scripts share `pc_common.py` (canonical place-cell definition, reward-position
resolution, `save_all` png/jpg/pdf writer, stats persistence). Run order matters only
where noted (GMM must run first).

| Script | Produces | Notes |
|--------|----------|-------|
| `ratinabox_v11.py` | the run directories themselves (data + checkpoints) | **training / data generation**; needs `torch`, `ratinabox`. Skip if you already have the run dirs. |
| `pc_common.py` | — | shared helpers, imported by everything |
| `gmm_place_metric.py` | `data/df_centers_{state}.pkl` | **PREREQUISITE** — GMM field centres; must run before the shift/recruitment/SI scripts |
| `create_place_metrics.py` | place-cell metric figures / top-Skaggs maps | uses canonical place-cell filter |
| `plot_goal_directed_maps.py` | population maps, occupancy, over-representation index (ORI), cross-cycle drift | includes random-walk-vs-goal-directed ORI test |
| `plot_replay_v11.py` | **core replay figures** — field shift, empty-cell threshold sweep, recruitment fate, example galleries | the paper's Fig. 2/3 machinery |
| `recuritment_place_cells.py` | recruitment-zone enrichment figures/stats | |
| `shift_toward_reward.py` | per-cell shift-toward-reward figures/stats | |
| `analyze_activity.py` | **Selectivity Index** (spatial + angular) and SI-vs-shift correlation | empty-cell filtered |
| `analyze_activty_lineardecoder.py` | **linear decoder** of distance-to-reward & gaze-alignment; **cross-decoding** pred↔postrl | population-level remapping test |
| `reward_phenomenology.py` | **the three literature phenomena**: over-representation, distance ramping, goal-vector tuning | new; frozen-replay + head-direction control |
| `reward_phenomenology_summary.py` | cross-condition **dissociation summary figure** | reads both runs' stats |
| `analyze_filtered_place_cell_shifts.py` | detailed per-cell field-shift analysis (many options) | optional deep-dive |
| `analyze_cycle_shift_matrix.py` | cross-state shift matrices (needs multiple cycles) | most useful for `Negative` (5 cycles) |

---

## 3. Requirements

```bash
pip install -r requirements.txt
```

Analysis needs only `numpy scipy pandas scikit-learn matplotlib seaborn`.
`torch` + `ratinabox` are needed **only** to regenerate data with `ratinabox_v11.py`.

---

## 4. Data required

Each analysis reads a **run directory** (`Positive_Results/` or `Negative_Results/`)
with this layout:

```
<run_dir>/
├── config.json                       # reward_pos, replay_cycle, n_cycles, … (authoritative hyperparameters)
├── data/
│   ├── df_behavior_{state}.pkl       # columns: timestep, x, y, theta
│   ├── df_activations_{state}.pkl    # columns: timestep, neuron_id, activity  (512 neurons)
│   ├── df_skaggs_{state}.pkl         # columns: neuron_id, skaggs_score
│   └── df_centers_{state}.pkl        # columns: neuron_id, k, center, icl_score  (produced by gmm_place_metric.py)
└── figures/                          # analysis output lands here
```

**States** (`{state}` tokens) present in a v11 run:
`before`, `after`, `cycle_1 … cycle_N`, `cycle_N_goal`, `goal_conditioned`,
and the frozen-replay pair `replay_c1_pred`, `replay_c1_postrl`.
`Positive` has `n_cycles = 1`; `Negative` has `n_cycles = 5`.

The two key comparisons everywhere are:
* **`replay_c1_pred` → `replay_c1_postrl`** = the frozen-trajectory (occupancy-matched) test → *genuine* change.
* **`goal_conditioned`** (reward-biased occupancy) = the behavioural read-out → *apparent* change.

### (Optional) regenerating the data

```bash
# Positive (reward head on, off-centre reward, 1 cycle)
python ratinabox_v11.py --output_dir Positive_Results/ --reward_pred_head \
    --reward_x 0.3 --reward_y 0.3 --n_cycles 1 --fixed_goal --replay_cycle 1
# Negative (task-RL control, centre reward, 5 cycles)
python ratinabox_v11.py --output_dir Negative_Results/ \
    --reward_x 0.5 --reward_y 0.5 --n_cycles 5 --fixed_goal --replay_cycle 1
```
See each run's `config.json` for the exact hyperparameters actually used
(e.g. `reward_pred_weight`, `rnn_rl_learning_rate`, `pred_loss_weight`).

---

## 5. How to run (in order)

Reward position is auto-read from each run's `config.json`, so you rarely need
`--reward`. Replace `<DIR>` with `Positive_Results` or `Negative_Results` (use an
absolute path if you run from inside `place_cell_analysis/`). Every figure is written
as a **png + jpg + pdf** triple; every test is appended to
`<DIR>/data/statistics_summary.json` and rendered to `statistics_report.txt`.

```bash
# 0. (once per run dir) GMM field centres — PREREQUISITE for shift/SI scripts.
#    Must include the replay pair as well as before/after.
python gmm_place_metric.py --run_dir <DIR> --states before,after,replay_c1_pred,replay_c1_postrl

# 1. Population maps + over-representation index (random-walk vs goal-directed)
python plot_goal_directed_maps.py --run_dir <DIR> --no_show

# 2. Core frozen-replay figures: field shift, empty-cell threshold sweep,
#    recruitment fate, example galleries
python plot_replay_v11.py --run_dir <DIR> --no_show

# 3. Recruitment-zone enrichment and per-cell shift
python recuritment_place_cells.py --run_dir <DIR>
python shift_toward_reward.py     --run_dir <DIR>

# 4. Selectivity Index (both state pairs)
python analyze_activity.py --run_dir <DIR> --state_before replay_c1_pred --state_after replay_c1_postrl
python analyze_activity.py --run_dir <DIR> --state_before before        --state_after after

# 5. Linear decoder + frozen-replay cross-decoding (population remapping test)
python analyze_activty_lineardecoder.py --run_dir <DIR> --period replay_c1_pred
python analyze_activty_lineardecoder.py --run_dir <DIR> --period replay_c1_postrl
python analyze_activty_lineardecoder.py --run_dir <DIR> --period replay_c1_pred --period_b replay_c1_postrl

# 6. The three reward phenomena (over-representation / ramping / goal-vector)
python reward_phenomenology.py --run_dir <DIR>
```

Then, once **both** run dirs are processed, produce the cross-condition figure.
Defaults assume `Positive_Results/` and `Negative_Results/` in the current directory;
pass `--positive_dir`/`--negative_dir` (and optionally `--positive_label`/
`--negative_label`, `--out_dir`) if your run dirs are named or located differently:

```bash
python reward_phenomenology_summary.py \
    --positive_dir Positive_Results --negative_dir Negative_Results
# -> summary_figures/reward_phenomenology_dissociation.(png|jpg|pdf)
```

Optional deep-dives: `create_place_metrics.py --run_dir <DIR>`,
`analyze_filtered_place_cell_shifts.py --run_dir <DIR> --state_a before --state_b after`,
`analyze_cycle_shift_matrix.py --run_dir <DIR>` (most informative for `Negative`, which has 5 cycles).

---

## 6. Outputs

* **`<DIR>/figures/`** — all figures as png/jpg/pdf triples.
* **`<DIR>/data/statistics_summary.json`** — machine-readable stats (append-only).
* **`<DIR>/data/statistics_report.txt`** — human-readable, with significance stars
  (`*` p<0.05, `**` p<0.01, `***` p<0.001). Regenerate any time with
  `python pc_common.py --run_dir <DIR>`.

> **Note on reruns:** `statistics_summary.json` is append-only, so re-running a
> script adds fresh records rather than replacing them. To avoid duplicates, delete
> the records for that script from the JSON before rerunning (or delete the JSON and
> rerun the whole pipeline).

---

## 7. Expected results — what your colleague should see

All numbers below are on **empty-cell-filtered place cells** (Skaggs ≥ 1.0), which is
the project's canonical de-confounding gate. "Survivors" = place cell in **both**
states of a pair. Negative-control reward is at the arena centre, so its raw
distances are smaller than Positive's — always compare the **pre→post change**, not
raw values across runs.

### 7.1 Field shift toward reward — frozen replay (`plot_replay_v11.py`)

| | **Positive** | **Negative (control)** |
|---|---|---|
| survivors (place in both) | 240 | 384 |
| mean field-distance to reward, pre → post | 0.403 → 0.358 | 0.265 → 0.266 |
| Δ (post−pre), <0 = toward reward | **−0.045** | +0.0009 |
| fraction of cells moving toward reward | **68 %** | 49 % (chance) |
| Wilcoxon p | **5.5e-10 \*\*\*** | 0.04 \* (but Δ≈0; trivial at n=384) |
| **threshold sweep** (Δ as empty-cell gate tightens) | **deepens −0.033 → −0.064**, stays *** | stays ≈ 0, loses significance |

Interpretation: the toward-reward shift in `Positive` is real and *strengthens* when
the emptiest cells are removed (genuine, not an empty-cell artifact); `Negative` is a
flat null throughout.

### 7.2 Recruitment fate (`plot_replay_v11.py` → `recruitment_fate`)

| | **Positive** | **Negative** |
|---|---|---|
| lost / recruited / shared | **77 / 35 / 240** | **0 / 7 / 384** |
| lost cells' Skaggs pre → post | 3.01 → 0.60 (fields **dissolve**) | — (none lost) |
| recruited cells' Skaggs pre → post | 0.51 → 3.77 (fields **emerge**) | 0.99 → 1.04 (borderline) |
| recruited cells near reward? | yes (mean dist 0.27; zone-enrichment z=1.67, p=0.048 \*) | no (0 near reward) |

The map in `Positive` reorganises (cells lost far from reward, recruited near it);
`Negative` is essentially unchanged (only 7 borderline recruits, none dissolve).

### 7.3 Phenomenon 1 — over-representation near reward (`reward_phenomenology.py`)

Field-peak density vs distance-to-reward, occupancy-matched frozen replay.

| | **Positive** | **Negative** |
|---|---|---|
| near-reward over-representation index, pre → post | 0.17 → **0.72** | 0.98 → 1.05 (flat) |
| mean field-peak distance, pre → post | 0.571 → 0.491 | 0.429 → 0.426 |
| Mann–Whitney p (pre vs post distances) | **2.7e-6 \*\*\*** | 0.92 (null) |

Fields migrate inward only in `Positive`. (The index stays <1 because the off-centre
reward sits near a corner; the *change* is the signal.)

### 7.4 Phenomenon 3 — reward-distance ramping (`reward_phenomenology.py`)

| | **Positive** | **Negative** |
|---|---|---|
| ramping index (frozen replay), pre → post | −0.037 → −0.034 | −0.047 → −0.046 |
| Wilcoxon p | 0.66 (**null**) | 0.11 (**null**) |
| *apparent* over-rep. index in goal-directed read-out | 1.19 | 1.09 |

**Null in the frozen replay for both runs** → single-cell distance-ramping is a
*behavioural overlay*, not a weight change. Corroborated by the decoder: distance-to-
reward is already ~0.9 R² decodable everywhere (it is implicit in the position code)
and does not strengthen post-RL.

### 7.5 Phenomenon 4 — goal-direction / goal-vector tuning (`reward_phenomenology.py`)

Mean resultant length (MRL) of the egocentric goal-bearing tuning curve, with
allocentric head-direction as a specificity control.

| | **Positive** | **Negative** |
|---|---|---|
| goal-direction MRL, pre → post (Δ) | 0.491 → 0.528 (**+0.037**) | 0.406 → 0.402 (−0.004) |
| Wilcoxon p | **3.4e-4 \*\*\*** | 8e-11 (tiny Δ) |
| head-direction MRL Δ (control) | +0.029 | −0.006 |

Directional tuning sharpens post-RL only in `Positive`. **Caveat:** head-direction
tuning co-rises almost as much (+0.029), so with a *fixed* goal this is a broad
increase in directional tuning, not proven goal-*specific* (a moving-reward
experiment would be needed to fully dissociate — cf. Sarel 2017).

### 7.6 Selectivity Index (`analyze_activity.py`, frozen-replay pair)

| | **Positive** | **Negative** |
|---|---|---|
| spatial SI, pre → post | −0.703 → −0.675 (p=1.1e-3 \*\*) | −0.412 → −0.404 (p=1e-3; ~3.5× smaller Δ) |
| SI-vs-shift correlation (spatial) | **r=−0.34, p=4.7e-8 \*\*\*** | r=−0.01, p=0.78 (null) |

SI is negative for almost all cells by base-rate geometry ("near reward" is a tiny
slice of the trajectory) — negative ≠ anti-reward. The informative results are that
spatial SI becomes *less* negative post-RL (~3.5× more in `Positive`) and that
less-selective cells shift most — only in `Positive`.

### 7.7 Linear decoder & cross-decoding (`analyze_activty_lineardecoder.py`)

| | **Positive** | **Negative** |
|---|---|---|
| within-state distance R² (pred / postrl) | 0.92 / 0.88 | 0.96 / 0.96 |
| **cross-decoding** distance R² (pred→postrl / postrl→pred) | **−0.22 / 0.72 (collapses)** | 0.95 / 0.95 (stable) |

A decoder trained on pre-RL activity **fails** to read post-RL activity in `Positive`
(the population code moved), but transfers perfectly in `Negative` — the
population-level signature of the same dissociation.

### 7.8 Apparent over-representation during behaviour (`plot_goal_directed_maps.py`)

| mean field-distance, random-walk → goal-directed | **Positive** | **Negative** |
|---|---|---|
| | 0.415 → 0.252 (p=3.8e-23 \*\*\*) | 0.267 → 0.201 (p=2.2e-18 \*\*\*) |

**Both** runs show apparent concentration near reward *during goal-directed
behaviour* — this is the behavioural overlay that the frozen-replay control (§7.1,
§7.3) then dissolves for `Negative` but not `Positive`. This contrast is the paper's
central point, and `reward_phenomenology_summary.py` renders it in one figure.

---

## 8. Methodological notes / caveats

* **Field centres use the smoothed-map PEAK**, not the centre of mass. A diffuse or
  near-empty map has its COM at the arena centre by construction; with an off-centre
  reward that fakes a "shift toward reward". Peak is immune to this and is the metric
  the paper trusts. Shift-based analyses are additionally restricted to survivors
  (place cell in **both** states) so empty cells cannot inject spurious values.
* **Compare deltas, not raw values, across runs** — the two runs have different reward
  locations (corner vs centre), which sets different baseline distances and different
  uniform-density expectations.
* **Large-n significance ≠ effect size.** With 240–384 paired cells, sub-0.01 changes
  reach p<0.05 (e.g. `Negative` SI, `Negative` replay shift). Read the magnitude and
  the threshold sweep, not just the star.
* **Goal-direction tuning is not proven goal-*specific*** with a single fixed goal
  (§7.5). The strongest, cleanest results are the field shift / concentration (§7.1,
  §7.3) and the decoder cross-decoding collapse (§7.7).
* **`statistics_summary.json` is append-only** — clear a script's records before
  rerunning it to avoid duplicate entries.
