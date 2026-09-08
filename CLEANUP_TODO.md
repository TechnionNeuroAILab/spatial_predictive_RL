# Remaining cleanup items (not auto-applied — need a judgment call)

This branch (`repo-cleanup`) commits the currently-untracked pipeline into git and adds
`README.md` / `requirements.txt` / `.gitignore` fixes so figures are reproducible from a clean
clone. It deliberately does **not** delete or move anything, since that's hard to reverse and
some of it may still be wanted. Left for a follow-up pass:

## 1. Physically remove/archive legacy trees (currently just git-ignored, still on disk)
- `New_Results/` (21M), `New_Results_v2/` (5.6G), `Results/` (15M), `old_code/` (9.8G) —
  confirmed superseded per `README.md` provenance note. Safe to delete once you're sure nothing
  in `New_Code/` or `code_analysis/` still reads from them (a quick `grep -rl "New_Results_v2\|old_code" New_Code code_analysis` before deleting is worth running).
- Not touched: `New_Results_v3/` and `Repro/` — these are current/canonical, kept in full on disk
  (only their raw `.pkl`/`.pt`/checkpoints/diagnostic-PNGs are git-ignored, not deleted).

## 2. Orphaned supplementary figures with no known generator
`main_v4_reframed.tex` references `figS_fate`, `figS_manifold`, `figS_decoder`, `figS_fourcond` —
none of these appear in `New_Code/sync_paper_figures.py`'s `FIGURES` dict. Either:
(a) find/write the generator and add an entry, or (b) confirm they're legacy figures no longer
regenerated and drop them from the manuscript, or (c) if they're manually-assembled (not
code-generated), say so explicitly in a comment next to the `\includegraphics` call.

## 3. `New_Code/` duplicate/superseded scripts
Pairs that look like one superseded the other — pick the canonical one, delete or move the other
to an `archive/` subfolder so `New_Code/` only holds what's actually used by `code_analysis/`:
- `dose_response.py` vs `dose_response_comparison.py`
- `reward_relocation_analysis.py` / `reward_relocation_maps.py` vs anything reading from `Repro/multi_seed_reloc/`
- `select_best_seed_fig{2,3,4}.py`, `run_fig2_seeds.py`, `run_fig23_extra_seeds.py`,
  `run_fig34_parallel.py` — one-off exemplar-selection scripts; fine to keep for provenance but
  worth separating from the always-imported library modules (e.g. a `New_Code/seed_selection/`
  subfolder) so it's clear at a glance what's "run once" vs "run every time."

## 4. `DEFAULT_RUN` mismatches
Each `code_analysis/figure {2,3}/utils.py` has a `DEFAULT_RUN` that does **not** match the run
actually used for the shipped exemplar (see README §2). Fix the default so
`generate_figure{2,3}.py` with no flags reproduces the shipped figure, instead of relying on a
remembered `--run_dir` override.

## 5. Directory names with spaces
`code_analysis/figure 1/` … `figure 5/` (vs `figure_extinction/`, `figure_sr/`, etc. which don't
have spaces) require quoting in every command/Makefile. Renaming to `figure1/` … `figure5/` would
remove a class of quoting bugs, but touches every import path in `New_Code`/`code_analysis` that
references them — do as its own reviewed change, not bundled with this one.

## 6. `paper/` repo boundary
Currently just git-ignored from this repo (see README "Manuscript repo" section). If you'd rather
have an explicit pointer (e.g. a git submodule) instead of a silently-ignored directory, that's a
deliberate choice to make — not done here since it touches a live Overleaf-synced repo.

## 7. Environment pin
`requirements.txt` (this branch) is a `pip list --format=freeze` snapshot of the
`spatial_predictive_RL` conda env — good enough to reinstall the same versions via pip, but not a
full conda environment export (some packages, e.g. `gmpy2`, are conda-native and may not resolve
identically via plain pip on every platform). Worth validating with a clean-room `pip install -r requirements.txt` before relying on it.
