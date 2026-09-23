# spatial_predictive_RL

Predictive-RNN place-cell model and the full reproduction pipeline for the manuscript
*Predictive manifolds and reward reshaping* ([`paper_repo/main.tex`](paper_repo/main.tex)).

## All code lives in [`code/`](code/)

Start with **[`code/README.md`](code/README.md)** — it documents the whole
chain: environment setup, the exact simulation commands, the JSON-cache builders, one
`generate_*.py` per manuscript figure, and the sync step into the manuscript.

For a figure-by-figure map (stem → generator → exemplar run → synced PDF), see
**[`context/FIGURE_REPRODUCTION.md`](context/FIGURE_REPRODUCTION.md)**.

```bash
pip install -r code/requirements.txt
sudo apt install poppler-utils        # pdftoppm, used by every figure assembler
python code/smoke_test.py       # verify the pipeline runs
```

## Everything else at this level is data or output

| path | what it is | tracked? |
|---|---|---|
| `results/` | current multi-seed run batches — the source of every shipped figure | metadata only |
| `sandbox/New_Results_v3/` | single-seed runs; only the optional `figS_fate` / `figS_decoder` generators read it | no |
| `data/` | Krishnan 2022 Source Data; Sosa 2025 NWB (DANDI `001361`) + caches | no |
| `paper_repo/` | the manuscript — a **separate** Overleaf-synced git repo (`main.tex`) | no (git-ignored) |
| `sandbox/` | archived leftovers from the paper-slim cleanup (not part of the pipeline) | no (git-ignored) |

Raw run data (`*.pkl`, `*.pt`, checkpoints) and per-run diagnostic PNGs are git-ignored:
they are regenerable from the commands in `code/README.md`, and total several
hundred GB.

Superseded trees (`New_Results/`, `New_Results_v2/`, `Results/`, `old_code/`,
`Analysis_Plots/`) live under `sandbox/project_legacy/`. They predate three fixes that
materially change results (arena object placement `b78eb35`, the peak/COM-split pipeline
`2278dca`, and an RNG re-seed fix `377728a`) — see the provenance section of
`code/README.md`.

Open cleanup items are tracked in [`context/CLEANUP_TODO.md`](context/CLEANUP_TODO.md).
