# AGENTS.md

## Cursor Cloud specific instructions

This is a Python computational-neuroscience research codebase (RatInABox place-cell +
predictive/reinforcement-learning simulations). There are **no web services, servers, APIs,
or databases** — everything is batch Python scripts that read/write `.pkl` files and emit
`.png` figures. "Running the application" means running the simulation and analysis scripts.

### Environment
- Dependencies are installed by the startup update script (see `requirements.txt`). PyTorch is
  the CPU-only build (`torch ...+cpu`) because the VM has no GPU; the code auto-falls back to
  CPU (`device = 'cuda' if torch.cuda.is_available() else 'cpu'`).
- Packages install to `~/.local` (user site). No virtualenv is used; just run `python3`.

### Data is Git LFS (must be pulled before running analysis)
- Files under `data/*.pkl` are **Git LFS** objects. A fresh checkout contains ~130-byte
  pointer stubs, and `pd.read_pickle(...)` will fail on them. Run **`git lfs pull`** once to
  fetch the real data (~1 GB total; `df_activations_*.pkl` are ~512 MB each). This is **not**
  in the update script (too heavy/networky to run on every startup) — run it manually when you
  need the analysis data.
- After `git lfs pull`, `git status` may show `data/*.pkl` as "modified" (real content vs.
  pointer). Do **not** commit these — they are LFS-managed. Likewise, the analysis scripts
  overwrite tracked figures under `figures/`; don't commit regenerated output unless intended.

### Running things (all commands from the repo root `/workspace`)
- Scripts use **relative paths** (`data/...`, `figures/...`), so always run from the repo root.
- Full pipeline (`code/code_v1.py`) does 1000-trial simulation + 200-epoch pretraining +
  **300,000 RL timesteps**. On CPU this is effectively multi-hour+ and not suitable for a quick
  check. It regenerates `data/df_behavior_*.pkl` and `data/df_activations_*.pkl`.
- Fast end-to-end analysis (uses the LFS data, ~40 s): `MPLBACKEND=Agg python3 code/shift_toward_reward.py`
  → writes figures to `figures/` and prints place-field shift t-tests.
- Place-field detection: `MPLBACKEND=Agg python3 code/gmm_place_metric.py` (GMM/ICL over all
  512 neurons → `data/df_centers_before.pkl`).
- Set `MPLBACKEND=Agg` for any script that plots — the VM is headless and scripts call
  `plt.show()`.
- `code/analyze_corr_using_metrics` has **no `.py` extension**; run as
  `python3 code/analyze_corr_using_metrics`.
- `code/reward_pdf_concentration.py` does `from shift_toward_reward import ...`, so run it with
  `code/` on the path (e.g. `cd code && MPLBACKEND=Agg python3 reward_pdf_concentration.py`).

### Tests / lint
- There is **no** test suite, linter config, or CI in this repo. "Verification" = running the
  scripts and confirming they produce the expected `.pkl`/`.png` outputs without error.
