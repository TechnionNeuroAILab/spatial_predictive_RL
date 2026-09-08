# paper_code — everything that reproduces `paper/main_v4_reframed.tex`

All simulation, analysis and figure code for the manuscript lives in this one folder.
Run data and datasets stay outside it (`Repro/`, `New_Results_v3/`, `external_data/`),
and the manuscript itself is a separate git repository (`paper/`, see the bottom).

```
paper_code/
  smoke_test.py            verify the pipeline runs (start here)
  sync_paper_figures.py    copy assembled figure PDFs into the manuscript
  requirements.txt         pinned deps + the poppler system dependency
  model/       the three simulators
  lib/         shared analysis library (imported by the figure code)
  runs/        drivers that produce run directories under Repro/
  aggregate/   drivers that produce the JSON caches figures read
  figures/     one directory per manuscript figure
  external/    readers for the two third-party datasets
  tools/       bibliography helper
```

## 0. Environment

```bash
pip install -r paper_code/requirements.txt
sudo apt install poppler-utils          # provides pdftoppm, used by every assembler
python paper_code/smoke_test.py         # deps + imports + every CLI + data presence
```

The smoke test is the acceptance gate for any change in here. It never regenerates a
figure. Add `--micro-run` to also execute a tiny training run, proving the simulator
works and not just that it imports.

## 1. Generate the runs

Single canonical runs (commands confirmed against each run dir's `config.json`):

```bash
# Main head-ON run
python paper_code/model/ratinabox_v11.py --seed 42 \
  --output_dir New_Results_v3/reward_pred_head_offcenter \
  --n_trials 1000 --timesteps 100 --phase1_epochs 100 --n_cycles 1 \
  --max_episodes 1000 --max_steps 20 --fixed_goal --reward_x 0.3 --reward_y 0.3 \
  --rnn_rl_lr 5e-5 --pred_loss_weight 0.3 --reward_pred_head --reward_pred_weight 1.5 \
  --reward_pred_sigma 0.15 --interleave_epochs 20 --replay_cycle 1 \
  --replay_steps 50000 --checkpoint_every 2000 --no_show

# Null (no-head) run — also the input for figS_fate / figS_decoder
python paper_code/model/ratinabox_v11.py --seed 42 \
  --output_dir New_Results_v3/final_v11_fixed \
  --n_trials 1000 --timesteps 100 --phase1_epochs 100 --n_cycles 5 \
  --max_episodes 5000 --max_steps 100 --fixed_goal --rnn_rl_lr 2e-6 \
  --pred_loss_weight 0.7 --interleave_epochs 60 --replay_cycle 1 \
  --replay_steps 50000 --checkpoint_every 5000 --no_show

# Reward relocation A->B
python paper_code/model/ratinabox_relocation.py --seed 42 \
  --output_dir New_Results_v3/reward_relocation \
  --n_trials 1000 --timesteps 100 --phase1_epochs 100 --n_cycles 2 \
  --max_episodes 2000 --max_steps 20 --fixed_goal --reward_x 0.3 --reward_y 0.3 \
  --relocate_x 0.7 --relocate_y 0.7 --relocate_at_cycle 1 --rnn_rl_lr 5e-5 \
  --pred_loss_weight 0.3 --reward_pred_head --reward_pred_weight 1.5 \
  --reward_pred_sigma 0.15 --interleave_epochs 20 --replay_steps 50000 \
  --checkpoint_every 5000 --no_show
```

**The shipped figures use the multi-seed batches, not the single runs above.** Those are
produced by the drivers in `runs/`, each writing to its own root under `Repro/`:

| driver | writes | feeds |
|---|---|---|
| `runs/run_fig2_seeds.py` | `Repro/multi_seed/nohead_seed{N}` | fig2, figS_pipeline, fig5, figS_sr, figS_discriminator, figS_rewardvector |
| `runs/run_fig23_extra_seeds.py` | same roots, seeds 10-19 (also exports the shared base configs) | fig2, fig3 |
| `runs/run_fig34_parallel.py` | `Repro/multi_seed_head/`, `Repro/multi_seed_reloc/` | fig3, fig4, fig5, figS_head, figS_relocation |
| `runs/run_rigid_seeds.py` | `Repro/multi_seed_rigid/` | figS_rigid |
| `runs/run_extinction_seeds.py` | `Repro/multi_seed_ext/` | figS_extinction, fig5 |

Each takes `--dry_run` (print commands only), `--seeds`, and `--prune` (drop large
intermediates via `runs/prune_run.py`). `runs/select_best_seed_fig{2,3,4}.py` re-derive
the exemplar seeds; `runs/verify_{extinction,rigid}_match.py` check config consistency
across a batch.

## 2. Build the JSON caches

The data-confrontation figures read cached statistics rather than re-scanning every run:

```bash
python paper_code/aggregate/run_extinction_analysis.py        # -> Repro/aggregate/extinction_summary.json
python paper_code/aggregate/run_estimator_validation.py       # -> Repro/aggregate/occupancy_matched_estimator.json
python paper_code/aggregate/run_reward_relative_analysis.py --out /tmp/rr.json
python paper_code/external/krishnan2022/analysis.py           # -> external_data/krishnan2022/cache/krishnan_data.json
python paper_code/external/sosa2025/analysis_sosa.py          # -> external_data/sosa2025/cache/sosa_tests.json
```

`run_reward_relative_analysis.py` was reconstructed during the consolidation (the
committed `reward_relative_summary.json` that fig5 reads had no surviving driver). It
refuses to overwrite the committed cache without `--force` — write to a scratch path and
diff first.

The Sosa reader needs the NWB files from DANDI accession `001361` in
`external_data/sosa2025/nwb/` (git-ignored, not redistributed). Krishnan uses the
published Source Data xlsx, which is committed.

## 3. Generate the figures

Everything at once, with the canonical arguments already encoded (this is the
recommended path -- it is where the exemplar run dirs are recorded):

```bash
python paper_code/regenerate_all.py --list          # what it will build
python paper_code/regenerate_all.py --jobs 8        # all 17 figures
python paper_code/regenerate_all.py --only fig2 fig3
```

Run only ONE batch at a time: two concurrent batches write the same PDFs.
`compare_figures.py --baseline <dir>` diffs a rebuild against a saved set of PDFs by
rendered pixels, which is the only comparison that survives PDF timestamps.

Or one directory at a time, each with a `generate_*.py`. All accept `--no_show`.

```bash
python paper_code/figures/fig1/generate_figure1.py  --run_dir Repro/multi_seed/nohead_seed0 --no_show
python paper_code/figures/fig2/generate_figure2.py  --run_dir Repro/multi_seed/nohead_seed16 --no_show
python paper_code/figures/fig3/generate_figure3.py  --run_dir Repro/multi_seed_head/head_seed28 --no_show
python paper_code/figures/fig4/generate_figure4.py  --run_dir Repro/multi_seed_reloc/reloc_seed9 --no_show
python paper_code/figures/fig5/generate_figure5_data.py --no_show      # the data-confrontation figure
python paper_code/figures/fig5/generate_figure5.py      --no_show      # -> figS_hypotheses

# supplements
python paper_code/figures/fig1/generate_figure1_supp.py --no_show      # figS_pipeline
python paper_code/figures/fig2/generate_figure2_supp.py --no_show      # figS_frozenreplay
python paper_code/figures/fig3/generate_figure3_supp.py --no_show      # figS_head
python paper_code/figures/fig4/generate_figure4_supp.py --no_show      # figS_relocation
python paper_code/figures/fig5/generate_figure5_supp.py --no_show      # figS_dissociation
python paper_code/figures/figS_sr/generate_figure_sr_supp.py --no_show
python paper_code/figures/figS_sr/generate_figure_rigid_supp.py --no_show
python paper_code/figures/figS_extinction/generate_figure_extinction.py --no_show
python paper_code/figures/figS_discriminator/generate_figure_discriminator.py --no_show
python paper_code/figures/figS_reward_vector/generate_figure_reward_vector_supp.py --no_show
python paper_code/figures/figS_sosa/generate_figure_sosa.py --no_show
```

**Pass `--run_dir` explicitly for fig1, fig2 and fig3** -- or just use
`regenerate_all.py`, which does it for you. Each `utils.py` `DEFAULT_RUN` points at a
*different* run from the one the shipped exemplar was made with:

| figure | DEFAULT_RUN | run that reproduces the shipped exemplar |
|---|---|---|
| fig1 | `Repro/nohead_seed42` | `Repro/multi_seed/nohead_seed0` |
| fig2 | `Repro/nohead_seed42` | `Repro/multi_seed/nohead_seed16` |
| fig3 | `Repro/multi_seed_head/head_seed0` | `Repro/multi_seed_head/head_seed28` |
| fig4 | `Repro/multi_seed_reloc/reloc_seed9` | (correct already) |

fig1's true source was previously undocumented; it was identified by re-deriving the
panel-d neuron selection (seed 42, 8 cells) across all 24 candidate run dirs -- only
`nohead_seed0` reproduces the shipped IDs `[509, 352, 439, 324, 222, 227, 45, 391]`.

Outputs land in each figure dir's `figures/` (main) and `figures_supp/` (supplementary).

## 4. Sync into the manuscript

```bash
python paper_code/sync_paper_figures.py --dry-run   # check every source exists first
python paper_code/sync_paper_figures.py             # -> paper/paper_figures/v2/ + MANIFEST.txt
```

Explicit dest→src allowlist (never a glob — several figures have shared the filename
`figure3_raw.pdf`). The MANIFEST records source path, mtime, sha256 and the git HEAD, so
any figure in the PDF traces back to the run that made it. The script refuses to run if a
non-optional source is missing.

## Figure design system

All visual decisions live in one place, `lib/pubstyle.py`, which every figure
directory's `utils.py` re-exports. Before it, each of the ten figure directories
carried its own `apply_nature_style()` and its own colours, and the rate maps were
drawn in `jet` across 37 call sites.

- **Type** — TeX Gyre Heros (a Helvetica clone). Arial/Helvetica are not installed
  here, so the previous font stack silently fell back to DejaVu Sans. 7 pt body,
  6 pt ticks, 5.5 pt annotations, 8 pt bold panel letters.
- **Colour** — the Okabe-Ito colourblind-safe palette, exposed by ROLE
  (`C["pre"]`, `C["head"]`, `C["reward_a"]`, ...) so a condition keeps one colour
  in every panel of every figure. Never name a colour in a title or label: that
  text goes silently wrong the moment the palette changes.
- **Rate maps** — `RATEMAP`, a white→amber→vermillion→maroon ramp, chosen by
  rendering real sparse place fields through eight candidates as an actual 2x8
  tile grid. Its stops are positioned (not evenly spaced) so the low third stays
  near-white and a sparse field still reads as sparse. Monotonic in lightness, so
  it survives greyscale printing.
- **Panel titles name the panel; the caption carries the sentence.** `title()`
  warns above 48 characters. Statistics go inside the axes via `stat_annot()`, or
  above their own violin — never stacked into a multi-line title and never into a
  tick label.
- **Render at display size.** The assemblers contain-fit each panel PDF into its
  slot, so a panel rendered 570 pt wide and shown in a 253 pt slot has all of its
  type scaled to 44%. This was the single biggest source of illegible labels.
  `fig1/utils.py:slot_width_in()` returns a panel's true on-page width; pass it as
  `width_in=` when rendering.

Helpers worth using rather than re-inventing: `arena_axes`, `reward_marker`,
`thin_colorbar`, `hist_outline` (overlapping distributions as fill + step outline,
instead of two alpha-blended solids that go muddy), `style_violin`, `zero_line`,
`fmt_p` (an unavailable p-value prints nothing, not "p = n/a").

Figures are deterministic: regenerating twice produces pixel-identical output.

## Figure coverage of `main_v4_reframed.tex`

19 of the 21 figures are reproducible from this folder. Not reproducible:

- **`figS_fate`, `figS_decoder`** — generators were recovered from the deleted legacy tree
  into `figures/figS_legacy/`, but their exact original invocation was never recorded, so
  the wrappers' `--run_dir` is an inference. They are `OPTIONAL` in the sync map and the
  manuscript still resolves them from `paper/paper_figures/`. See
  `figures/figS_legacy/README.md`.
- **`figS_manifold`, `figS_fourcond`** — **no generator exists anywhere in the repo**. Their
  PDFs are stale July 2026 output whose sources were pruned. Flagged for removal from the
  manuscript in `../CLEANUP_TODO.md`.

The two architecture schematics (`v10_architecture.png`, `v11_architecture.png`) are
hand-drawn assets in `paper/Architecture figures/`, not generated.

## Data provenance

`Repro/` and `New_Results_v3/` are current. The legacy `New_Results/`, `New_Results_v2/`,
`Results/` and `old_code/` trees are superseded and git-ignored: they predate an arena
object-placement fix (`b78eb35`), the peak/COM-split analysis pipeline (`2278dca`), and an
RNG re-seed fix that desyncs head/no-head runs from the same seed (`377728a`). Do not use
them to regenerate anything.

Raw run data (`*.pkl`, `*.pt`, checkpoints) and per-run diagnostic PNGs are git-ignored —
regenerate them with the commands above.

## Known rough edges

Ten figure dirs each ship a module named `utils.py` (and seven an `analysis.py`), so the
generators rely on `sys.path` ordering and `sys.modules` juggling to load the right one.
This works — each generator runs in its own process — but `figure1_supp` and `figure3_supp`
silently pick up fig2's `utils`. Preserved as-is during the consolidation; converting the
tree to a real package would remove the hazard. Tracked in `../CLEANUP_TODO.md`.

## The manuscript repository

`paper/` is a separate, Overleaf-synced git repo
(`TechnionNeuroAILab/Predictive_Manifold_Rewards_Reshaping`) and is git-ignored here.
This repo is the code source of truth; that one is the manuscript source of truth.
`sync_paper_figures.py` is the only bridge between them.
