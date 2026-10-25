# spatial_predictive_RL

Predictive-RNN place-cell model and reproduction pipeline for
*Reward changes the use and structure of predictive spatial maps through
distinct mechanisms* ([`paper_repo/main.tex`](paper_repo/main.tex)).

## Reproduce and verify

The detailed protocol is in [`code/README.md`](code/README.md). The shortest
publication check is:

```bash
pip install -r code/requirements.txt
sudo apt install poppler-utils mupdf-tools
python code/verify_paper.py            # manuscript ↔ code ↔ cached statistics
python code/smoke_test.py              # imports, CLIs, data and publication gate
python code/regenerate_all.py --list   # the 13 jobs producing all 15 PDF stems
python code/sync_paper_figures.py --dry-run
python code/check_figure_type.py        # final-size fonts, scale and page fit
```

The authoritative figure map is
[`context/FIGURE_REPRODUCTION.md`](context/FIGURE_REPRODUCTION.md).
`code/regenerate_all.py` defines the generators and exemplar runs;
`code/sync_paper_figures.py` defines exactly the PDF stems referenced by the
manuscript. Both are checked against `main.tex`.

## Repository layout

| path | what it is | tracked? |
|---|---|---|
| `code/` | simulators, run drivers, analyses, figure generators and checks | yes |
| `results/aggregate/` and `results/*/aggregate/*.json` | compact cohort definitions and manuscript statistics | yes |
| `results/multi_seed*/` | simulation outputs, checkpoints, learning curves and logs | no; regenerate locally |
| `data/` | published-source inputs/caches and provenance for Krishnan 2022 and Sosa 2025 | yes, except raw NWB |
| `context/` | figure provenance and maintenance notes | yes |
| `paper_repo/` | separate Overleaf-synced manuscript repository | no in this repository |

Simulation outputs are not stored in git. This includes checkpoints, activity
tables, per-run configuration and metrics, learning curves, logs, and derived
cell tables. The run drivers recreate them with deterministic seed-indexed
commands. Assembled figures under `code/figures/` are tracked. Compact JSON
summaries needed to audit the reported results are also tracked. Downloaded
Sosa NWB recordings are not distributed.
