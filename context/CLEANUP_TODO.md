# Cleanup status

## Done (branch `repo-cleanup`)

**Commit 1 — get the pipeline into git.** Most of the working pipeline was untracked; a
`git clean` would have destroyed it. Committed `New_Code/`, `code_analysis/`,
`data/` scripts and small provenance metadata; extended `.gitignore` to keep raw
run data, legacy trees and the separate `paper/` repo out.

**Commit 2 — consolidate all code into `code/`.**
- One folder for everything that reproduces `paper/main_v4_reframed.tex`:
  `model/ lib/ runs/ aggregate/ figures/ external/ tools/` (see `code/README.md`).
- Figure dirs renamed to drop spaces (`code_analysis/figure 1/` → `code/figures/fig1/`),
  removing a whole class of shell-quoting bugs.
- Deleted 82 dead files: 31 `New_Code/` modules (18 orphan analyses with zero importers,
  13 superseded), `Code/ratinabox_v9/v10.py` + 7 v9-era scripts, the rest of
  `Code/place_cell_analysis/` (incl. a third copy of `ratinabox_v11.py`), and
  `code_analysis/figure_plasticity/` (its figures appear only in the superseded
  `paper/main.tex`). All recoverable: `git show 95b1edf:<path>`.
- Recovered the `figS_fate` and `figS_decoder` generators from the deleted legacy tree
  into `code/figures/figS_legacy/` and wired them into the sync map as `OPTIONAL`.
- Added `code/aggregate/run_reward_relative_analysis.py`, the missing driver for the
  committed `reward_relative_summary.json` that fig5 reads — previously that panel could
  not be regenerated from scratch at all.
- Added `code/smoke_test.py` (deps / imports / every CLI / data presence / optional
  simulator micro-run) as the acceptance gate.
- **Fixed a wrong `requirements.txt`**: commit 1 pinned the `spatial_predictive_RL` conda
  env, which lacks `pandas`, `seaborn` and `scikit-learn` and cannot run the analysis
  pipeline at all. Now pinned to `base` (Python 3.13.9), derived from the live code's
  actual imports — which also surfaced three previously undocumented deps (`pypdf`,
  `h5py`, `openpyxl`) and the `poppler-utils`/`pdftoppm` system requirement.

## Open

### 1. Two manuscript figures have no generator — decide whether to cut
`figS_manifold` (dimensionality PR 21.7→7.0 + Procrustes 0.5079 panels) and `figS_fourcond`
(four-condition field-centre-distance violins) are included by `main_v4_reframed.tex` but
**no `.py` file anywhere in the repo produces them** — searched by figure stem and by the
verbatim strings in their panels. Their PDFs are stale July 2026 matplotlib output whose
source files were pruned, still resolving from `paper/paper_figures/`.

Agreed direction: cut them from the manuscript. That edit belongs in the separate `paper/`
repo, so it is flagged here rather than done. (The modules that computed the underlying
metrics — `manifold_analysis.py`, `procrustes_analysis.py`, `dimensionality_analysis.py` —
were deleted with the other orphans; recover from `95b1edf` if you'd rather rebuild.)

### 2. Confirm the two recovered legacy figures
`code/figures/figS_legacy/generate_figS_{fate,decoder}.py` encode an **inferred**
`--run_dir` (`sandbox/New_Results_v3/final_v11_fixed`). Run them and compare against the shipped
PDFs; if they match, drop both from `OPTIONAL` in `sync_paper_figures.py`. See
`code/figures/figS_legacy/README.md`.

### 3. Validate the reconstructed reward-relative driver
`run_reward_relative_analysis.py` reproduces the committed cache's schema but has not been
run against it (a full pass is 32 seeds of heavy pickle loads). Write to a scratch path and
diff before trusting it; it refuses to overwrite the committed cache without `--force`.

### 4. Fix the fig2/fig3 `DEFAULT_RUN` mismatch
`figures/fig2/utils.py` defaults to `results/nohead_seed42` but the shipped exemplar is
`results/multi_seed/nohead_seed16`; `figures/fig3/utils.py` defaults to `head_seed0` vs the
shipped `head_seed28`. Until fixed, `--run_dir` must be passed explicitly (the smoke test
warns). fig4 is already correct.

### 5. Delete the legacy data trees (needs an explicit decision — irreversible)
`New_Results/` (21M), `New_Results_v2/` (5.6G), `Results/` (15M), `old_code/` (9.8G) are
superseded and git-ignored, so deleting them is **not** recoverable from git. ~15.5G.
Nothing in `code/` references them (verify with
`grep -rn "New_Results_v2\|old_code" code/`).

Related: with `figure_plasticity/` gone, the only remaining consumer of
`sandbox/New_Results_v3/dose_sweep_*` and `matched_*` is `figures/fig1/plot_panel_h_across_lrs.py`,
a diagnostic that is not assembled into any shipped panel.

### 6. Optional: make the figure tree a real Python package
Ten `figures/*/utils.py` and seven `analysis.py` share module names, so the generators use
`sys.path` ordering plus `sys.modules.pop("utils")` juggling to load the right one. It
works (each generator is its own process) but `figure1_supp` and `figure3_supp` silently
get fig2's `utils`. Converting to a package with relative imports would delete all 37
`sys.path.insert` and 25 `sys.modules` hacks. Deliberately not done in the consolidation —
it can't be validated without regenerating every figure.

### 7. Six scripts take no arguments at all
`figures/figS_sosa/generate_figure_sosa.py`, `runs/verify_{extinction,rigid}_match.py`,
`aggregate/run_extinction_analysis.py`, `external/krishnan2022/analysis.py` and
`external/sosa2025/analysis_sosa.py` have no `argparse`, so they silently ignore every
flag — including `--help`, which makes them *run for real*. That bit this consolidation:
the first version of the smoke test "`--help`-tested" them and thereby regenerated
figS_sosa's PDF, while importing the unguarded `build_manifest.py` overwrote
`data/sosa2025/MANIFEST.json`'s recorded dataset download timestamp. Both were
reverted, and the smoke test now detects both classes (`has_main_guard`/`uses_argparse`)
and import-tests instead. Giving these six a real argparse would remove the trap.

### 8. Clean the manuscript's `\graphicspath`
26 of the 29 entries in `main_v4_reframed.tex` are dead (`Figures/` doesn't exist; the 23
`New_Results/**/figures/` entries and `Results/` contain only diagnostic PNGs, no figure
the tex asks for). Only `paper_figures/v2/`, `paper_figures/` and `Architecture figures/`
are live. Again an edit in the separate `paper/` repo.

### 9. Regenerate and verify the figures end-to-end
Nothing was regenerated during the consolidation, so byte-identical reproduction is
unproven. The manuscript is unaffected (the sync map still points at the unchanged PDFs).
A full regeneration pass is the real test of the moved pipeline.
