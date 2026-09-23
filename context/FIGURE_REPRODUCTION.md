# Figure reproduction map

Canonical map from every figure in [`paper_repo/main.tex`](paper_repo/main.tex) to the
generator that produces it. Prefer this file over the provenance table inside the
manuscript (that table still names the pre-consolidation `code_analysis/` /
`New_Code/` paths).

## One-shot rebuild

```bash
pip install -r code/requirements.txt
sudo apt install poppler-utils

# optional: rebuild aggregate JSON caches first (slow)
python code/regenerate_all.py --with-caches --jobs 8

python code/regenerate_all.py --jobs 8          # 17 reproducible figures
python code/sync_paper_figures.py               # -> paper_repo/paper_figures/v2/
# then compile paper_repo/main.tex
```

`regenerate_all.py` encodes the exemplar `--run_dir` for every figure whose
`utils.py` default points at a different seed. Do not rely on those defaults.

## Sync allowlist

[`code/sync_paper_figures.py`](code/sync_paper_figures.py) copies
assembled PDFs into `paper_repo/paper_figures/v2/` under the stem names used by
`\includegraphics`. Optional entries (`figS_fate`, `figS_decoder`) are skipped
when their source is missing. Two stems have **no generator** anywhere in the
repo (`figS_manifold`, `figS_fourcond`); their PDFs are kept in `v2/` so the
manuscript still builds.

## Main figures

| tex stem | discussed | generator | exemplar `--run_dir` / inputs | assembled PDF | synced as |
|---|---|---|---|---|---|
| `fig1` | Fig. 1 | `code/figures/fig1/generate_figure1.py` | `results/multi_seed/nohead_seed0` | `fig1/figures/figure1_raw.pdf` | `fig1.pdf` |
| `fig2` | Fig. 2 | `code/figures/fig2/generate_figure2.py` | `results/multi_seed/nohead_seed16` | `fig2/figures/figure2_raw.pdf` | `fig2.pdf` |
| `fig3` | Fig. 3 | `code/figures/fig3/generate_figure3.py` | `results/multi_seed_head/head_seed28` | `fig3/figures/figure3_raw.pdf` | `fig3.pdf` |
| `fig4` | Fig. 4 | `code/figures/fig4/generate_figure4.py` | `results/multi_seed_reloc/reloc_seed9` | `fig4/figures/figure4_raw.pdf` | `fig4.pdf` |
| `fig5` | Fig. 5 (data confrontation) | `code/figures/fig5/generate_figure5_data.py` | caches under `results/aggregate/` + `data/*/cache/` | `fig5/figures/figure5_data_raw.pdf` | `fig5.pdf` |

## Supplementary figures

| tex stem | label / section | generator | exemplar inputs | assembled PDF | synced as | status |
|---|---|---|---|---|---|---|
| `figS_head` | Fig. S1 matched proximity-objective | `figures/fig3/generate_figure3_supp.py` | `results/multi_seed_head/` | `fig3/figures_supp/figure3_supp_raw.pdf` | `figS_head.pdf` | reproducible |
| `figS_rigid` | Fig. S1 plasticity control | `figures/figS_sr/generate_figure_rigid_supp.py` | `results/multi_seed_rigid/` | `figS_sr/figures_supp/figure_rigid_supp_raw.pdf` | `figS_rigid.pdf` | reproducible |
| `figS_dissociation` | Fig. S1 robustness | `figures/fig5/generate_figure5_supp.py` | multi-seed aggregates | `fig5/figures_supp/figure5_supp_raw.pdf` | `figS_dissociation.pdf` | reproducible |
| `figS_fate` | S2 recruitment fate | `figures/figS_legacy/generate_figS_fate.py` | `sandbox/New_Results_v3/final_v11_fixed` (inferred) | `figS_legacy/figures_supp/figS_fate_raw.pdf` | `figS_fate.pdf` | optional / unconfirmed |
| `figS_decoder` | S2 cross-decoding | `figures/figS_legacy/generate_figS_decoder.py` | `sandbox/New_Results_v3/final_v11_fixed` (inferred) | `figS_legacy/figures_supp/figS_decoder_raw.pdf` | `figS_decoder.pdf` | optional / unconfirmed |
| `figS_manifold` | S2 PR / Procrustes panel | — | — | — | `figS_manifold.pdf` (shipped only) | **no generator** |
| `figS_fourcond` | S4 four-condition violins | — | — | — | `figS_fourcond.pdf` (shipped only) | **no generator** |
| `figS_sr` | S3 successor representation | `figures/figS_sr/generate_figure_sr_supp.py` | `results/multi_seed/`, `results/multi_seed_head/` | `figS_sr/figures_supp/figure_sr_supp_raw.pdf` | `figS_sr.pdf` | reproducible |
| `figS_pipeline` | S5 Phase-1 aggregates | `figures/fig1/generate_figure1_supp.py` | `results/multi_seed/` | `fig1/figures_supp/figure1_supp_raw.pdf` | `figS_pipeline.pdf` | reproducible |
| `figS_frozenreplay` | S5 per-network contrasts | `figures/fig2/generate_figure2_supp.py` | `results/multi_seed/` | `fig2/figures_supp/figure2_supp_raw.pdf` | `figS_frozenreplay.pdf` | reproducible |
| `figS_relocation` | S5 relocation distributions | `figures/fig4/generate_figure4_supp.py` | `results/multi_seed_reloc/` | `fig4/figures_supp/figure4_supp_raw.pdf` | `figS_relocation.pdf` | reproducible |
| `figS_rewardvector` | S6 distance / bearing | `figures/figS_reward_vector/generate_figure_reward_vector_supp.py` | multi-seed batches | `figS_reward_vector/figures_supp/figure_reward_vector_supp_raw.pdf` | `figS_rewardvector.pdf` | reproducible |
| `figS_hypotheses` | S6 readout vs rewrite | `figures/fig5/generate_figure5.py` | multi-seed aggregates | `fig5/figures/figure5_raw.pdf` | `figS_hypotheses.pdf` | reproducible |
| `figS_extinction` | S7 reward omission | `figures/figS_extinction/generate_figure_extinction.py` | `results/aggregate/extinction_summary.json` + Krishnan cache | `figS_extinction/figures_supp/figure_extinction_supp_raw.pdf` | `figS_extinction.pdf` | reproducible |
| `figS_sosa` | S8 Sosa reanalysis | `figures/figS_sosa/generate_figure_sosa.py` | `data/sosa2025/cache/sosa_tests.json` | `figS_sosa/figures_supp/figure_sosa_supp_raw.pdf` | `figS_sosa.pdf` | reproducible |
| `figS_discriminator` | S9 occupancy matching | `figures/figS_discriminator/generate_figure_discriminator.py` | `results/aggregate/occupancy_matched_estimator.json` | `figS_discriminator/figures_supp/figure_discriminator_supp_raw.pdf` | `figS_discriminator.pdf` | reproducible |

## Architecture plates (hand-drawn)

| tex stem | file | generator |
|---|---|---|
| `v10_architecture.png` | `paper_repo/Architecture figures/v10_architecture.png` | none (hand-drawn) |
| `v11_architecture.png` | `paper_repo/Architecture figures/v11_architecture.png` | none (hand-drawn) |

## Run batches that feed the figures

| driver | writes | feeds |
|---|---|---|
| `code/runs/run_fig2_seeds.py` | `results/multi_seed/nohead_seed{N}` | fig1, fig2, figS_pipeline, figS_frozenreplay, fig5, figS_sr, figS_discriminator, figS_rewardvector |
| `code/runs/run_fig23_extra_seeds.py` | same roots, seeds 10–19 | fig2, fig3 |
| `code/runs/run_fig34_parallel.py` | `results/multi_seed_head/`, `results/multi_seed_reloc/` | fig3, fig4, fig5, figS_head, figS_relocation |
| `code/runs/run_rigid_seeds.py` | `results/multi_seed_rigid/` | figS_rigid |
| `code/runs/run_extinction_seeds.py` | `results/multi_seed_ext/` | figS_extinction, fig5 |

Aggregate caches (rebuild with `regenerate_all.py --with-caches` or the scripts in
`code/aggregate/` and `code/external/`):

- `results/aggregate/extinction_summary.json`
- `results/aggregate/occupancy_matched_estimator.json`
- `results/multi_seed_reloc/aggregate/reward_relative_summary.json`
- `data/krishnan2022/cache/krishnan_data.json`
- `data/sosa2025/cache/sosa_tests.json`

## Status summary

- **19 figures** have a generator under `code/figures/` (17 in
  `regenerate_all.py`, plus the two optional legacy wrappers).
- **`figS_fate` / `figS_decoder`**: generators recovered; exemplar `--run_dir` is
  inferred. Compare against the shipped PDFs before trusting a regenerate.
- **`figS_manifold` / `figS_fourcond`**: no generator; shipped PDFs only. Flagged
  for removal from the manuscript in [`CLEANUP_TODO.md`](context/CLEANUP_TODO.md).
- **Architecture plates**: hand-drawn assets, not regenerated.
