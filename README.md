# spatial_predictive_RL

Predictive-RNN place-cell model and the full reproduction pipeline for the manuscript
*Predictive manifolds and reward reshaping* (`paper/main_v4_reframed.tex`).

## All code lives in [`paper_code/`](paper_code/)

Start with **[`paper_code/README.md`](paper_code/README.md)** — it documents the whole
chain: environment setup, the exact simulation commands, the JSON-cache builders, one
`generate_*.py` per manuscript figure, and the sync step into the manuscript.

```bash
pip install -r paper_code/requirements.txt
sudo apt install poppler-utils        # pdftoppm, used by every figure assembler
python paper_code/smoke_test.py       # verify the pipeline runs
```

## Everything else at this level is data or output

| path | what it is | tracked? |
|---|---|---|
| `Repro/` | current multi-seed run batches — the source of every shipped figure | metadata only |
| `New_Results_v3/` | current single-seed runs (incl. the `figS_fate`/`figS_decoder` input) | metadata only |
| `external_data/` | Krishnan 2022 Source Data; Sosa 2025 NWB (DANDI `001361`) + caches | caches + xlsx |
| `Analysis_Plots/` | exploratory diagnostic plots, not manuscript figures | yes |
| `paper/` | the manuscript — a **separate** Overleaf-synced git repo | no (git-ignored) |
| `New_Results/`, `New_Results_v2/`, `Results/`, `old_code/` | superseded, kept for provenance only | no (git-ignored) |

Raw run data (`*.pkl`, `*.pt`, checkpoints) and per-run diagnostic PNGs are git-ignored:
they are regenerable from the commands in `paper_code/README.md`, and total several
hundred GB.

The legacy trees predate three fixes that materially change results (arena object
placement `b78eb35`, the peak/COM-split pipeline `2278dca`, and an RNG re-seed fix
`377728a`) — see the provenance section of `paper_code/README.md`.

Open cleanup items are tracked in [`CLEANUP_TODO.md`](CLEANUP_TODO.md).
