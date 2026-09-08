# spatial_predictive_RL

Code to train the predictive-RNN place-cell model and reproduce every run,
statistic, and figure used in the manuscript (`Predictive_Manifold_Rewards_Reshaping`,
maintained in a separate repo — see **Manuscript repo** below).

This file replaces an older README that documented a superseded (`New_Results_v2`)
pipeline; the commands below match what actually produced `New_Results_v3` / `Repro/`.

## Environment

```bash
conda create -n spatial_predictive_RL python=3.10
conda activate spatial_predictive_RL
pip install -r requirements.txt
```
Pinned from the exact env used for all reported runs (`ratinabox==1.15.3`, `torch==2.5.1`).

## Repository layout

- `Code/` — simulation entry points (`ratinabox_v11.py` main model, `ratinabox_relocation.py`
  reward-relocation variant, `ratinabox_extinction.py` extinction variant). `ratinabox_v9.py`/`v10.py`
  are earlier, superseded model versions kept for provenance.
- `New_Code/` — analysis library + figure-generation drivers, imported by `code_analysis/figure */`.
  Also contains one-off scripts used to run/select multi-seed batches (`run_*_seeds.py`,
  `select_best_seed_fig*.py`) — see `CLEANUP_TODO.md` for which of these are still current.
- `code_analysis/figure {1..5}/`, `figure_extinction/`, `figure_discriminator/`,
  `figure_reward_vector/`, `figure_sr/` — one directory per manuscript figure; each has its own
  `generate_figure*.py` entry point and writes to a local `figures/` (main) and `figures_supp/`
  (supplementary) subfolder.
- `New_Results_v3/`, `Repro/` — **current, canonical** run outputs (raw pickles/checkpoints are
  git-ignored; only small `config.json`/`metrics.json`/logs/aggregate JSONs are tracked).
  `Repro/` holds the multi-seed batches; `New_Results_v3/` holds the original single-seed runs.
- `New_Results/`, `New_Results_v2/`, `Results/`, `old_code/` — **legacy, superseded** trees kept
  on disk for provenance only (see Provenance note below); git-ignored, not part of the current
  pipeline.
- `external_data/` — third-party datasets used for the data-confrontation figure (Krishnan et al.
  2022; Sosa, Plitt & Giocomo 2025). Raw NWB files and vendored third-party repos are git-ignored;
  only our own cache/figures/scripts are tracked. Fetch instructions below.
- `Analysis_Plots/` — exploratory/diagnostic plots (not manuscript figures).
- `paper/` — the manuscript LaTeX source; this is a **separate git repository** (Overleaf-synced,
  remote `TechnionNeuroAILab/Predictive_Manifold_Rewards_Reshaping`), git-ignored here. Figure
  scripts write into `paper/paper_figures/v2/` via `New_Code/sync_paper_figures.py`, which is how
  the two repos connect.

## 1. Generate the runs

Exact commands (confirmed against each run dir's `config.json`):

```bash
# Main head-ON run (Fig 3 source)
python Code/ratinabox_v11.py --seed 42 --output_dir New_Results_v3/reward_pred_head_offcenter \
  --n_trials 1000 --timesteps 100 --phase1_epochs 100 --n_cycles 1 --max_episodes 1000 --max_steps 20 \
  --fixed_goal --reward_x 0.3 --reward_y 0.3 --rnn_rl_lr 5e-5 --pred_loss_weight 0.3 \
  --reward_pred_head --reward_pred_weight 1.5 --reward_pred_sigma 0.15 --interleave_epochs 20 \
  --replay_cycle 1 --replay_steps 50000 --checkpoint_every 2000 --no_show

# Null (no-head) run (Fig 1 stability / Fig 2 source)
python Code/ratinabox_v11.py --seed 42 --output_dir New_Results_v3/final_v11_fixed \
  --n_trials 1000 --timesteps 100 --phase1_epochs 100 --n_cycles 5 --max_episodes 5000 --max_steps 100 \
  --fixed_goal --rnn_rl_lr 2e-6 --pred_loss_weight 0.7 --interleave_epochs 60 \
  --replay_cycle 1 --replay_steps 50000 --checkpoint_every 5000 --no_show

# Reward relocation A->B (Fig 4 source)
python Code/ratinabox_relocation.py --seed 42 --output_dir New_Results_v3/reward_relocation \
  --n_trials 1000 --timesteps 100 --phase1_epochs 100 --n_cycles 2 --max_episodes 2000 --max_steps 20 \
  --fixed_goal --reward_x 0.3 --reward_y 0.3 --relocate_x 0.7 --relocate_y 0.7 --relocate_at_cycle 1 \
  --rnn_rl_lr 5e-5 --pred_loss_weight 0.3 --reward_pred_head --reward_pred_weight 1.5 \
  --reward_pred_sigma 0.15 --interleave_epochs 20 --replay_steps 50000 --checkpoint_every 5000 --no_show

# Plasticity (learning-rate) sweep (S1 source)
python New_Code/run_dose_sweep.py --mode lr --lrs 2e-6 1e-5 5e-5 1e-4 3e-4 --seed 42 --prune \
  --root New_Results_v3/dose_sweep_lr

# Matched head vs no-head, 6 seeds (S_matched source)
python New_Code/run_matched_seeds.py --seeds 42 123 7 1 2 99 --lr 1e-4 --root New_Results_v3
```

Multi-seed batches (Fig 2/3/4 exemplar selection, extinction, rigid control) are produced by the
`New_Code/run_*_seeds.py` drivers into `Repro/<batch_name>/<seed_dir>/`, each with its own
`config.json` for provenance. See in-file `--help` on each driver for exact seed lists.

Sweep/matched/multi-seed runs auto-prune large intermediate files to keep per-run disk down.

## 2. Generate the figures

Each figure is self-contained under `code_analysis/`:

```bash
python "code_analysis/figure 1/generate_figure1.py" --run_dir New_Results_v3/final_v11_fixed --no_show
python "code_analysis/figure 2/generate_figure2.py" --run_dir Repro/multi_seed/nohead_seed16 --no_show
python "code_analysis/figure 3/generate_figure3.py" --run_dir Repro/multi_seed_head/head_seed28 --no_show
python "code_analysis/figure 4/generate_figure4.py" --run_dir Repro/multi_seed_reloc/reloc_seed9 --no_show
python "code_analysis/figure 5/generate_figure5_data.py" --no_show   # data-confrontation figure
python "code_analysis/figure 5/generate_figure5.py" --no_show        # -> figS_hypotheses (supplement)
```

**Important:** Fig 2/3's shipped exemplars come from explicit `--run_dir` overrides (above), which
do **not** match each script's internal `DEFAULT_RUN`. Always pass `--run_dir` explicitly rather
than relying on the default. (Tracked as a cleanup item — see `CLEANUP_TODO.md`.)

Supplementary/other figures (`figure_extinction`, `figure_discriminator`, `figure_reward_vector`,
`figure_sr`, and the Sosa reanalysis under `external_data/sosa2025/`) each follow the same
`generate_figure*.py --no_show` pattern in their own directory.

## 3. Sync into the manuscript

```bash
python New_Code/sync_paper_figures.py --dry-run   # verify every source exists first
python New_Code/sync_paper_figures.py             # copies into paper/paper_figures/v2/ + MANIFEST.txt
```
This refuses to run if any figure's source PDF is missing — treat that as a signal a figure
above wasn't (re)generated yet. See `New_Code/sync_paper_figures.py` for the full dest->src map.

## External datasets

- **Krishnan et al. 2022**: published Source Data file, cached under `external_data/krishnan2022/`.
- **Sosa, Plitt & Giocomo 2025**: public on DANDI (accession `001361`); fetch the NWB files into
  `external_data/sosa2025/nwb/` (git-ignored — not redistributed here).

## Provenance note

`New_Results_v2/` predates three fixes that materially change results: an arena object-placement
bug fix (`b78eb35`), the introduction of the peak/COM-split analysis pipeline
(`New_Code/replay_field_analysis.py`, `2278dca`), and an RNG re-seed fix so head/no-head runs
from the same seed are properly desynced (`377728a`). `New_Results_v3/` and `Repro/` were
generated after all three fixes and are the intended-pipeline output; `New_Results_v2/`,
`New_Results/` and `old_code/` are kept only as a record of what earlier manuscript drafts were
based on and should not be used to regenerate current figures.

## Manuscript repo

The LaTeX source and assembled figures live at
`git@github.com:TechnionNeuroAILab/Predictive_Manifold_Rewards_Reshaping.git` (checked out here
at `paper/`, its own git history — not tracked by this repo). This repo is the code source of
truth; that repo is the manuscript source of truth. `New_Code/sync_paper_figures.py` is the only
bridge between them.
