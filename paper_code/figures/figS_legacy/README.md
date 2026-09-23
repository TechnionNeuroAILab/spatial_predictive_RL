# figS_legacy — recovered generators for `figS_fate` and `figS_decoder`

`main_v4_reframed.tex` includes two supplementary figures whose generators were not
part of the current figure pipeline. They are kept here as standalone copies of the
older analysis scripts that produced the shipped panels.

| tex figure | generator | wrapper |
|---|---|---|
| `figS_fate` | `plot_replay_v11.py` → `fig_recruitment_fate()` | `generate_figS_fate.py` |
| `figS_decoder` | `analyze_activty_lineardecoder.py` → `plot_cross_decoding_results()` | `generate_figS_decoder.py` |

## Why these four files sit together in one directory

`plot_replay_v11.py` and `plot_goal_directed_maps.py` here are **not** the modules of
the same name in `paper_code/lib/` — they are the older, larger variants (916 vs 313
lines, and 744 vs 709 lines respectively), and only these produce the shipped panels.

Both scripts do `sys.path.insert(0, dirname(__file__))` at import time, so keeping all
four files (the two generators plus their `pc_common.py` and `plot_goal_directed_maps.py`
dependencies) in this single directory means each generator resolves its *own* copies
first and can never pick up `lib/`'s same-named modules. Do not move these files apart,
and do not add this directory to `sys.path` from anywhere else.

## Provenance caveat (read before trusting the output)

The shipped `figS_fate.pdf` / `figS_decoder.pdf` in `paper/paper_figures/` are July 2026
matplotlib output; the exact invocations were never recorded, and the generators' own
output files were pruned. The `--run_dir` defaults in both wrappers
(`New_Results_v3/final_v11_fixed`) are an **inference** — that is the null/no-head run
whose `data/` still contains the required `replay_c1_pred` / `replay_c1_postrl` periods.

So: these wrappers give the figures a working, documented generator, but they are not yet
confirmed to reproduce the shipped PDFs. Compare before swapping them into the manuscript.
Until then `sync_paper_figures.py` lists both in `OPTIONAL`, and the tex keeps resolving
them from `paper/paper_figures/`.

Also note `plot_replay_v11.py` regenerates its whole set of replay figures, not just the
one panel, so it is slow and rewrites other files under `<run_dir>/figures/`.

## Not recovered

`figS_manifold` and `figS_fourcond` (also in `main_v4_reframed.tex`) have **no** generator
anywhere in the repo — no `.py` file contains those stems or the strings in their panels.
They are flagged for removal from the manuscript in `CLEANUP_TODO.md`.
