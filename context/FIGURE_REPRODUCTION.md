# Figure reproduction map

Canonical map from every figure in [`paper_repo/main.tex`](../paper_repo/main.tex) to the
generator that produces it. The executable sources of truth are
`code/regenerate_all.py` and `code/sync_paper_figures.py`; this file explains
that map for readers.

## One-shot rebuild

```bash
pip install -r code/requirements.txt
sudo apt install poppler-utils mupdf-tools

python code/verify_paper.py
python code/regenerate_all.py --with-caches --jobs 8
python code/sync_paper_figures.py               # -> paper_repo/paper_figures/v2/
cd paper_repo && pdflatex -interaction=nonstopmode main.tex
```

`regenerate_all.py` and each main figure's `DEFAULT_RUN` agree on the canonical
exemplars. `verify_paper.py` checks these paths, cohort sizes, Methods constants
and the numerical Results against the released caches.

## Sync allowlist

[`code/sync_paper_figures.py`](../code/sync_paper_figures.py) copies
assembled PDFs into `paper_repo/paper_figures/v2/` under the stem names used by
`\includegraphics`. It refuses to run if a source is missing, a manuscript stem
is absent from the allowlist, or an allowlist entry is unused by the manuscript.

## Main figures

| tex stem | discussed | generator | exemplar `--run_dir` / inputs | assembled PDF | synced as |
|---|---|---|---|---|---|
| `fig_model` | model schematic | `code/figures/fig1/generate_figure1.py` | `results/multi_seed/nohead_seed16` | `fig1/figures/figure_model_raw.pdf` | `fig_model.pdf` |
| `fig1` | spatial map | `code/figures/fig1/generate_figure1.py` | `results/multi_seed/nohead_seed16` | `fig1/figures/figure1_raw.pdf` | `fig1.pdf` |
| `fig2` | fixed replay | `code/figures/fig2/generate_figure2.py` | `results/multi_seed/nohead_seed16` | `fig2/figures/figure2_raw.pdf` | `fig2.pdf` |
| `fig3` | proximity teaching | `code/figures/fig3/generate_figure3.py` | `results/multi_seed_head/head_seed28` | `fig3/figures/figure3_raw.pdf` | `fig3.pdf` |
| `fig4` | relocation | `code/figures/fig4/generate_figure4.py` | `results/multi_seed_reloc/reloc_seed40` | `fig4/figures/figure4_raw.pdf` | `fig4.pdf` |
| `fig_omission` | reward omission | `code/figures/fig5/generate_figure5_data.py` | aggregate + Krishnan caches | `fig5/figures/figure_omission_raw.pdf` | `fig_omission.pdf` |
| `fig_relocdata` | recorded relocation | `code/figures/fig5/generate_figure5_data.py` | relocation cohort + Sosa cache | `fig5/figures/figure_relocdata_raw.pdf` | `fig_relocdata.pdf` |

## Supplementary figures

| tex stem | label / section | generator | exemplar inputs | assembled PDF | synced as | status |
|---|---|---|---|---|---|---|
| `figS_pipeline` | map quality and behavioural shifts | `figures/fig1/generate_figure1_supp.py` | paired cohort | `fig1/figures_supp/figure1_supp_raw.pdf` | `figS_pipeline.pdf` | reproducible |
| `figS_dissociation` | gate/threshold robustness | `figures/fig5/generate_figure5_supp.py` | headline aggregates | `fig5/figures_supp/figure5_supp_raw.pdf` | `figS_dissociation.pdf` | reproducible |
| `figS_relocation` | relocation distributions | `figures/fig4/generate_figure4_supp.py` | relocation cohort | `fig4/figures_supp/figure4_supp_raw.pdf` | `figS_relocation.pdf` | reproducible |
| `figS_rewardvector` | distance and bearing tuning | `figures/figS_reward_vector/generate_figure_reward_vector_supp.py` | paired + relocation cohorts | `figS_reward_vector/figures_supp/figure_reward_vector_supp_raw.pdf` | `figS_rewardvector.pdf` | reproducible |
| `figS_sr` | successor-representation comparison | `figures/figS_sr/generate_figure_sr_supp.py` | paired cohort | `figS_sr/figures_supp/figure_sr_supp_raw.pdf` | `figS_sr.pdf` | reproducible |
| `figS_extinction` | reward omission | `figures/figS_extinction/generate_figure_extinction.py` | extinction + Krishnan caches | `figS_extinction/figures_supp/figure_extinction_supp_raw.pdf` | `figS_extinction.pdf` | reproducible |
| `figS_sosa` | Sosa reanalysis | `figures/figS_sosa/generate_figure_sosa.py` | Sosa cache | `figS_sosa/figures_supp/figure_sosa_supp_raw.pdf` | `figS_sosa.pdf` | reproducible |
| `figS_discriminator` | occupancy matching | `figures/figS_discriminator/generate_figure_discriminator.py` | occupancy-matched cache | `figS_discriminator/figures_supp/figure_discriminator_supp_raw.pdf` | `figS_discriminator.pdf` | reproducible |

## Run batches that feed the figures

| driver | writes | feeds |
|---|---|---|
| `code/runs/run_until_clean.py --family paired` | `results/multi_seed/`, `multi_seed_head/`, `multi_seed_rigid/` | fig1–3 and paired-cohort supplements |
| `code/runs/run_until_clean.py --family reloc` | `results/multi_seed_reloc/` | fig4, fig_relocdata and relocation supplements |
| `code/runs/run_until_clean.py --family extinction` | `results/multi_seed_ext/` | fig_omission and figS_extinction |

Aggregate caches (rebuild with `regenerate_all.py --with-caches` or the scripts in
`code/aggregate/` and `code/external/`):

- `results/aggregate/extinction_summary.json`
- `results/aggregate/headline_stats.json`
- `results/aggregate/cohorts.json`
- `results/aggregate/occupancy_matched_estimator.json`
- `results/multi_seed_reloc/aggregate/reward_relative_summary.json`
- `data/krishnan2022/cache/krishnan_data.json`
- `data/sosa2025/cache/sosa_tests.json`

## Publication invariant

Only figures referenced by `main.tex` are regenerated and synced. Retired
exploratory generators are archival code, not manuscript dependencies. Run
`python code/verify_paper.py` to check this invariant and the reported
statistics before release.
