# code — everything that reproduces `paper_repo/main.tex`

All simulation, analysis and figure code for the manuscript lives in this folder.
Run data and external datasets stay in `results/` and `data/`; the manuscript is
the separate `paper_repo/` repository. Figure-by-figure provenance (stem →
generator → exemplar run → synced PDF) is in
[`../context/FIGURE_REPRODUCTION.md`](../context/FIGURE_REPRODUCTION.md).

```
code/
  verify_paper.py          check manuscript ↔ code ↔ cached results
  smoke_test.py            imports, CLIs, data and publication checks
  regenerate_all.py        canonical manuscript-figure jobs
  sync_paper_figures.py    copy assembled figure PDFs into the manuscript
  requirements.txt         pinned deps + the poppler system dependency
  model/       the three simulators
  lib/         shared analysis library (imported by the figure code)
  runs/        drivers that produce run directories under results/
  aggregate/   drivers that produce the JSON caches figures read
  analysis/    standalone analyses not used as manuscript inputs
  figures/     one directory per manuscript figure
  external/    readers for the two third-party datasets
  tools/       bibliography helper
```

## 0. Environment

```bash
pip install -r code/requirements.txt
sudo apt install poppler-utils mupdf-tools  # pdftoppm assembly + mutool type gate
python code/verify_paper.py       # numerical, Methods and figure consistency
python code/smoke_test.py         # deps + imports + CLIs + data + paper gate
python code/check_figure_type.py  # final-size typography and page-fit gate
```

Neither command retrains a cohort or regenerates a figure. Add `--micro-run` to
the smoke test to execute a tiny simulator run.

## 1. Generate the runs

The paper uses seed-indexed cohorts, not a single illustrative run. The
authoritative base argument lists are `FIG2_BASE` and `FIG3_BASE` in
`runs/run_fig23_extra_seeds.py`, `FIG4_BASE` in
`runs/run_fig34_parallel.py`, `RIGID_BASE` in `runs/run_rigid_seeds.py`, and
`EXT_BASE` in `runs/run_extinction_seeds.py`.

The preferred cohort driver trains until the required number of networks passes
the predeclared entropy gate while preserving seed indices across matched arms:

```bash
python code/runs/run_until_clean.py --family paired --target 50 --gpu 0,1 --jobs 12
python code/runs/run_until_clean.py --family reloc --target 50 --gpu 0,1 --jobs 12
python code/runs/run_until_clean.py --family extinction --target 50 --gpu 0,1 --jobs 12
```

Use `--dry_run` to print commands without training, `--smoke` for a tiny
end-to-end run, and `--skip_existing` to resume. Full training is GPU-intensive.
The released cohort definitions are in `results/aggregate/cohorts.json`: 55
matched triples, 52 relocation networks and 51 omission networks pass the
5.0-bit gate. Failed seeds remain identified and are never renumbered.

The simulators write a `config.json`, `metrics.json`, checkpoints, learning
curves, and period-labelled activity/behaviour files into each run directory.
New configs include replay length and the relocation/omission switch so the
key run-specific protocol choices are recoverable without consulting prose. They also record
the 10-step episode warm-up, decoder learning rate, requested and snapped
reward coordinates, and the action-indexing conventions.

Implementation details needed for exact reproduction:

- Phase A and interleaved offline prediction use the outgoing action from
  observation \(t\) to target \(t+1\). Online control and all probes feed the
  preceding transition's action (the current movement-heading code) into the
  recurrent state; the within-episode RL prediction term retains that lagged
  convention.
- The within-episode RL prediction regularizer is next-observation MSE only;
  Phase A and each interleaved predictive block include the additional L1
  activity penalty. Interleaved trajectories use greedy actions without early
  termination; analysis probes sample with policy logits divided by 0.3.
- Every PPO training episode begins from zero recurrent state and takes 10
  unoptimized random warm-up moves before its at-most-20-step rollout.
- The requested reward coordinate is used by the proximity target, probes and
  analyses; training goal cues and termination use its nearest grid point.
- The omission simulator preserves its historical one-step PPO state
  recomputation offset. This exception is documented in the manuscript and is
  not present in the main or relocation simulator.
- Cross-cycle relocation and omission contrasts retain the standard cycle
  order: the first post-RL snapshot precedes that cycle's interleaved
  predictive block, while the second snapshot follows that intervening block
  and the next RL block. The manuscript states this explicitly.
- Per-run `df_skaggs_*.parquet` files are immediate simulator diagnostics.
  Publication analyses recompute Skaggs information from the activity and
  behaviour tables, including the documented post-reset trimming, and those
  recomputed values are authoritative.

## 2. Build the compact analysis caches

Figures and `verify_paper.py` read compact summaries rather than repeatedly
scanning hundreds of gigabytes of per-timestep activity:

```bash
python code/aggregate/cohorts.py                        # -> results/aggregate/cohorts.json
python code/aggregate/headline_stats.py                 # -> results/aggregate/headline_stats.json
python code/aggregate/run_extinction_analysis.py        # -> results/aggregate/extinction_summary.json
python code/aggregate/run_estimator_validation.py       # -> results/aggregate/occupancy_matched_estimator.json
python code/aggregate/run_reward_relative_analysis.py --out /tmp/rr.json
python code/external/krishnan2022/analysis.py           # -> data/krishnan2022/cache/krishnan_data.json
python code/external/sosa2025/analysis_sosa.py          # -> data/sosa2025/cache/sosa_tests.json
```

`run_reward_relative_analysis.py` refuses to overwrite the released cache
without `--force`; rebuild to a scratch path and compare first. The headline
builder records paired and relocation per-network values and the exact retained
seed lists used by the paper. Reward-omission results use three-period survivor
gates and come exclusively from `extinction_summary.json`; they are deliberately
not duplicated in `headline_stats.json`.

The Sosa analysis needs the NWB files from DANDI accession `001361` under
`data/sosa2025/nwb/` (not redistributed). Run
`external/sosa2025/build_manifest.py` after downloading them to verify sizes
and record checksums. The Krishnan analysis reads the published Source Data
workbook distributed under `data/krishnan2022/`.

The field-size follow-up requested during manuscript review remains deliberately
standalone and does not alter the paper or its figures:

```bash
python code/analysis/field_size_vs_reward.py
```

It writes `results/aggregate/field_size_vs_reward.json` and a companion PDF.
The script states the cell gate, field-size definitions, controls and sign
convention in its module documentation.

## 3. Generate the figures

Use the orchestrator so every generator receives the canonical exemplar run:

```bash
python code/regenerate_all.py --list          # 13 jobs / 15 manuscript stems
python code/regenerate_all.py --jobs 8
python code/regenerate_all.py --only fig2 fig3
python code/regenerate_all.py --plot-only     # redraw from existing figure_data.npz
```

Run only ONE batch at a time: two concurrent batches write the same PDFs.
`compare_figures.py --baseline <dir>` diffs a rebuild against a saved set of PDFs by
rendered pixels, which is the only comparison that survives PDF timestamps.

The canonical examples are `nohead_seed16` for Figures 1–2, `head_seed28`
for Figure 3 and `reloc_seed40` for Figure 4. Their `utils.py` defaults and
`regenerate_all.py` agree. The first and fifth jobs each emit two plates
(`fig_model` + `fig1`, and `fig_omission` + `fig_relocdata`), which is why 13
jobs cover 15 manuscript stems.

Outputs land in each figure dir's `figures/` (main) and `figures_supp/` (supplementary).

## 4. Sync into the manuscript

```bash
python code/sync_paper_figures.py --dry-run   # check every source exists first
python code/sync_paper_figures.py             # -> paper_repo/paper_figures/v2/ + MANIFEST.txt
```

Explicit dest→src allowlist (never a glob — several figures have shared the filename
`figure3_raw.pdf`). The MANIFEST records source path, mtime, sha256 and the git HEAD, so
any figure in the PDF traces back to the run that made it. The script refuses to run if a
source is missing or if either the allowlist or `main.tex` contains an unmatched
stem. Run `python code/verify_paper.py` after updating the manuscript, caches or
figure map.

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
- **Render at display size — now enforced, not advised.** The old assemblers
  contain-fit each panel PDF into its slot, so a panel rendered 570 pt wide and
  shown in a 253 pt slot had all of its type scaled to 44%. Combined with the
  `\includegraphics` scale that was applied on top, printed type across the
  figure set ranged from 2.07 pt to 10.13 pt against Nature's 5 pt floor.

  `lib/figlayout.py` inverts this. A `Layout` owns **both** dimensions of every
  slot; a panel is authored at exactly `LAYOUT.slot_size_in(key)` and saved with
  `save_figure(..., exact_size=True)`; `compose()` only translates. A panel that
  does not fill its slot raises `LayoutScaleError` naming the figsize it should
  have used. Nothing is scaled at assembly, so declared size equals printed size.

  Three consequences worth knowing:
  - `Layout(..., caption_lines=N)` raises `LayoutHeightError` at construction —
    before any data is loaded, which matters for the 19-minute figures.
  - `exact_size=True` clips instead of shrinking, and warns when it has to. A
    colourbar or legend outside the axes needs `tight_layout(rect=...)`.
  - Panel letters come from the layout, in a reserved strip above each row, so
    they cannot land on panel content. There is one convention (8 pt bold,
    lowercase, no parentheses) instead of the four that existed before.

- **Page width is the document's, not Nature's.** `lib/pagegeom.py` holds the
  geometry read off `main.log`: `TEXT_W_PT = 505.69` (177.7 mm), not the 183 mm
  the figures used to be drawn at. Every `\includegraphics` is `width=\linewidth`,
  so the include scale is exactly 1. To target a 183 mm journal, set
  `PAGE_W = NATURE_DOUBLE_PT` — the whole set then rescales uniformly, which is
  the operation the old per-figure scales made impossible.

- **`code/check_figure_type.py` is the gate.** It measures each figure PDF with
  `mutool draw -F stext` (which reports span sizes in final page coordinates),
  multiplies by the include scale parsed from `main.tex`, and fails on: a scale
  other than 1, any glyph below 5 pt, artwork taller than the caption leaves
  room for, or a typeface other than TeX Gyre Heros. Run it after every
  regeneration; `--generated` measures the generators' own output before sync.

Helpers worth using rather than re-inventing: `arena_axes`, `reward_marker`,
`thin_colorbar`, `hist_outline` (overlapping distributions as fill + step outline,
instead of two alpha-blended solids that go muddy), `style_violin`, `zero_line`,
`fmt_p` (an unavailable p-value prints nothing, not "p = n/a").

Figures are deterministic: regenerating twice produces pixel-identical output.
Note that the move to `TEXT_W_PT` invalidated every stored baseline at once, so
`compare_figures.py` will report CHANGED for the whole set against anything
older than that change — re-record the baseline before relying on it.

## Figure coverage of `main.tex`

All 15 PDF stems currently referenced by `main.tex` have a canonical source in
`sync_paper_figures.py` and are covered by the 13 jobs in `regenerate_all.py`.
The equality is enforced by both `verify_paper.py` and the sync script. Retired
exploratory generators may remain under `figures/`, but they are not part of
the publication pipeline.

## Data provenance

Only `results/multi_seed*` and the compact released caches are inputs to the
paper pipeline. Trees under `sandbox/` are archival and must not be used for
manuscript regeneration. They predate fixes to arena object placement,
peak/centre-of-mass analysis and matched-arm random-number seeding.

The simulators regenerate checkpoints, activity tables, configuration files,
metrics, learning curves and logs. None of those products is stored in git.
Assembled figures under `code/figures/` are stored. The publication package
also includes the compact JSON files used by `verify_paper.py`; the gate fails
explicitly if any is missing. Dependency pins describe the tested
analysis/figure environment. Historical training library versions were not
fully recorded, so the released random seeds and run-driver arguments are the
available training provenance.

Each figure directory retains a local `utils.py` for compatibility with the
original scripts. The orchestrator runs generators in separate processes so
these same-named modules cannot leak between figures.

## The manuscript repository

`paper_repo/` is a separate, Overleaf-synced git repo
(`TechnionNeuroAILab/Predictive_Manifold_Rewards_Reshaping`) and is git-ignored here.
This repo is the code source of truth; that one is the manuscript source of truth.
`sync_paper_figures.py` is the only bridge between them. Compile the paper after
syncing:

```bash
cd paper_repo
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```
