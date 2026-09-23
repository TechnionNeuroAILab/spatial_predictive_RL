# Code review — `paper_code/`

> **Part 2 (whole-codebase audit) is the important half** and begins at
> [Whole-codebase audit](#whole-codebase-audit-beyond-the-diff). Part 1 below reviews only the
> working-tree diff, which turned out to be almost entirely documentation.


## How to read the confidence ratings

Every finding carries a **Confidence** line with two parts.

| rating | meaning |
|---|---|
| **Certain** | I ran it, or the code states it unambiguously and I traced every step. If you disagree, one of us has misread a line — check the quoted code. |
| **High** | Traced by reading, not executed. I can name the mechanism and the call path; I have not seen it fire. |
| **Medium** | The mechanism is real; whether it changes a published number depends on data or usage I could not observe from this checkout. |
| **Low** | A judgement call — style, convention, or a risk that needs a future change to materialise. Reasonable people can decline it. |

Where the two differ I say so, e.g. *"mechanism Certain, impact Medium"*: I am sure the code
does the wrong thing, and unsure how much it moved the numbers. That distinction matters more
than the rating itself, because **no raw run data survives in this checkout** (see A12), so
nothing below could be re-measured against real runs. Where I could substitute a reproduction
on synthetic input, I did, and I say so.


## Part 1 — working-tree diff

**Verification performed (analysis only — no model runs).**
`smoke_test.py --phase imports` → 32 ok / 1 warn / 0 fail.
`smoke_test.py --phase cli` → 40 ok / 0 fail.
Every `lib/` module imports standalone. The `ratemaps.py` extraction was checked
line-for-line against the three copies it replaces and is numerically identical.
Findings 1 and 2 were reproduced with a standalone script.

---

## Severity triage: does any of this make a result fake?

**No. Nothing in this diff invalidates a headline claim.** One supplementary panel measures
something other than what its caption says, one cached JSON cannot be regenerated, and the
rest is documentation and hygiene. Evidence for each verdict below.

### Tier 1 — a shipped panel is not measuring what it claims (finding 1)

`figS_discriminator` panel d ("Estimate vs matched sample size", x-axis *"matched timesteps
(per period)"*). The `frac < 1.0` branch un-matches the strata, and at the leftmost point it
does so almost completely. Simulated at the observed scale (2142 matched timesteps,
32×32×8 = 8192 possible joint strata, so 1–4 samples per occupied stratum):

```
mean count/stratum   frac   pre samples with a post counterpart
                 1   0.125                              12.4%
                 2   0.125                              21.3%
                 4   0.125                              38.2%
                 1   0.5                                49.5%
                 4   0.5                                93.7%
   (shared-index fix: 100% at every frac, by construction)
```

At `frac=0.125` the "matched" pre and post maps are built from **nearly disjoint**
(position × heading) strata — precisely the confound the module exists to remove. Panel d's
practical guidance is therefore not a sample-size curve for the matched estimator.

Scope, and why this is not fatal: `if frac < 1.0` is entered **only** from `estimator_row`'s
`size_curve` loop. Panels a/b/c and every number quoted from `summary` use `frac=1.0` and
never touch the branch. Note also that the cached curve drifts *toward zero* with less data
(no-head cycle_1_goal: −0.0465 → −0.0352), whereas un-matching pushes *toward* the naive
estimate (−0.19, i.e. more negative) — the two effects partly cancel, so the true
sample-size dependence is likely steeper than plotted, not shallower.

### Tier 2 — the reproducibility promise is false; the numbers are not (finding 2)

`hash(mode)` makes `estimate()`'s RNG stream differ per process, so re-running
`aggregate/run_estimator_validation.py` does not reproduce
`Repro/aggregate/occupancy_matched_estimator.json`. But the magnitude is small. From the
cached file:

| | within-seed `rep_sd` (20 reps) | across-seed SD | effect being compared |
|---|---|---|---|
| `est_joint` | ≈ 0.004 | 0.019 – 0.036 | −0.028 vs −0.073 (gt no-head vs head) |

The Monte-Carlo noise the RNG controls is ~10× smaller than seed-to-seed variability and
~6–10× smaller than the contrast under test, and each `mean_dd` is an unbiased average over
20 reps whichever stream is drawn. A rerun perturbs the JSON in the third decimal and moves
no conclusion. This is a broken reproduction guarantee — made worse by the docstring this
diff *adds*, which asserts `default_rng([seed, rep, mode])` — not a wrong result.

### Tier 3 — latent, currently inert (finding 6)

The `ords.setdefault("gt", …)` seed-set mismatch does not bite on the current data. Checked
directly in the cache: `est_naive` and `est_joint` are non-null for **every** seed in all
four period × condition blocks (`n_null = 0`), so `ok` is all-True and the `gt` block's means
and its `paired_p` describe the same 34 paired seeds:

```
ordering (goal_conditioned) → gt: nohead −0.02773  head −0.06735  paired_p 0.00090
```

It becomes a wrong number, or a crash, only when a future rerun has a seed where the naive
estimator misses the ≥10-gated-cell floor.

### Everything else changes no number

* The `ratemaps.py` extraction was checked function-by-function against the three copies it
  replaces — identical arithmetic, so no figure moves.
* The `importlib` work-around (finding 9) rebinds symbols that are the *same objects*:
  fig1/fig2/fig3 `utils.py` all re-export from `lib/pubstyle.py`, and all three define
  `DEFAULT_REWARD = (0.3, 0.3)` and `PAGE_MARGIN = 1.0 * MM_TO_PT`. No panel changes.
* The de-capitalisation pass touched only comments, docstrings and Markdown — no axis label,
  title, legend or any other plot-facing string.
* Imports all still resolve: `smoke_test.py --phase imports` 32 ok / 0 fail,
  `--phase cli` 40 ok / 0 fail.

**The finding that actually stops a reader cold is 13**, not any of the above: `ratemaps.py`
and `stats_helpers.py` are untracked, so a fresh clone fails at import across the whole
pipeline. That breaks the repo without touching a single number.

### Worth saying explicitly

`figS_discriminator` is a figure whose job is to report a **negative** result, and it does so
honestly: ground truth gives no-head −0.028 vs head −0.067 (p = 0.0009, ordering recovered)
while the matched estimator gives −0.0214 vs −0.0228 (p = 0.41, ordering not recovered),
under a panel titled *"Only trajectory control recovers the head vs no-head ordering"*. The
module is being used to document its own limitation. Whatever else is wrong here, the intent
is not to oversell.

---

## Findings, most severe first

### 1. `lib/occupancy_matched_estimator.py:295-297` — the sample-size curve silently un-matches the strata


**Confidence:** **Certain.** Reproduced: `match_indices` alignment verified, and the frac-cut histograms measured (pre `[35 54 36]` vs post `[37 47 41]`). The call path to `size_curve` -> panel d is traced. Not in doubt.
```python
i_pre, i_post = match_indices(s_pre, s_post, rng)   # identical stratum histograms
if frac < 1.0:
    k_pre  = rng.choice(len(i_pre),  max(50, int(len(i_pre)  * frac)), replace=False)
    k_post = rng.choice(len(i_post), max(50, int(len(i_post) * frac)), replace=False)
    i_pre, i_post = i_pre[k_pre], i_post[k_post]
```

`match_indices` returns two index arrays that are **positionally aligned** — it walks the
strata in order and appends exactly `target[s]` entries to each list — so element *k* of
`i_pre` and of `i_post` always belong to the same (position × heading) stratum. Drawing
`k_pre` and `k_post` independently throws that alignment away. The two subsets then have
only *expected*-equal histograms, not equal ones, which reintroduces the sampling confound
the whole module exists to remove.

Reproduced (3 strata, `frac=0.125`):

```
matched  pre [300 400 300]  post [300 400 300]   aligned: True
frac-cut pre [ 35  54  36]  post [ 37  47  41]   still matched: False
shared-index fix                                 still matched: True
```

`frac < 1.0` is reached only from `estimator_row`'s `size_curve` loop, which feeds
**panel d of `figS_discriminator` ("Sample-size dependence of the matched estimator —
practical guidance")**. That panel therefore conflates "less data" with "less matching",
and the bias grows as `frac` shrinks — i.e. exactly along the axis the panel plots.

Fix: draw one index set and apply it to both.

```python
k = rng.choice(len(i_pre), max(50, int(len(i_pre) * frac)), replace=False)
i_pre, i_post = i_pre[k], i_post[k]
```

(As a bonus this removes a latent `ValueError`: `rng.choice(n, max(50, …), replace=False)`
raises when `n < 50`.)

### 2. `lib/occupancy_matched_estimator.py:292` — `hash(mode)` makes the estimator non-reproducible across processes


**Confidence:** **Certain** that the RNG is non-reproducible (measured across three processes). **Medium** that it matters: `rep_sd` is ~10x below the across-seed SD, so reruns move the third decimal, not a conclusion.
```python
rng = np.random.default_rng([seed, rep, hash(mode) % (2**31)])
```

Python salts `str.__hash__` per process unless `PYTHONHASHSEED` is pinned, so the same
seed and the same rep draw a different subsample on every invocation:

```
hash('joint') % 2**31 across 3 processes: 892552764 / 777482213 / 1643756848
```

`aggregate/run_estimator_validation.py` caches its output to
`Repro/aggregate/occupancy_matched_estimator.json`, and `figS_discriminator` reads that
cache — so re-running the cache rebuild does not reproduce the published numbers. This
diff *adds* a docstring to `estimate()` that states the opposite:

> `seed : int` — Base seed; each repetition draws from `default_rng([seed, rep, mode])`.

Fix: replace `hash(mode)` with a fixed lookup, e.g.
`{"naive": 0, "pos": 1, "joint": 2, "reweight": 3}[mode]`.

### 3. `figures/fig4/analysis.py:243` — new docstring describes the wrong return value


**Confidence:** **Certain.** The function is six lines and returns `rm, sk`; the docstring says occupancy. Nothing to interpret.
```python
def _load_rate_maps(data_dir, period, m: int = M, hidden_dim: int = HIDDEN_DIM):
    """Rate maps and occupancy for one period's pickles."""
    ...
    rm, occ = build_rate_maps(beh, act, m, hidden_dim)
    sk = compute_skaggs(rm, occ)
    return rm, sk
```

The function returns `(rate_maps, skaggs)`; `occ` is computed and discarded. The docstring
was added by this diff. A caller who writes `rm, occ = _load_rate_maps(...)` gets Skaggs
scores in an "occupancy" variable with no error — and the only current caller does
`rm, sk = …`, so nothing catches it. Should read "Rate maps and Skaggs scores".

### 4. `lib/analysis_helpers.py:47-55` — `smooth_and_norm` is run twice per unit


**Confidence:** **Certain** that the work is duplicated (two list comprehensions, same call). **Low** severity: it is speed, not correctness.
```python
com  = np.array([field_center_of_mass(smooth_and_norm(rm[n])) for n in range(hidden_dim)], …)
peak = np.array([field_peak(smooth_and_norm(rm[n]))          for n in range(hidden_dim)], …)
```

1024 `gaussian_filter` calls per period instead of 512. `load_period_centers` is the
pipeline's workhorse — every seed × period of every multi-seed figure and every gate check
goes through it, and `figures/fig5/generate_figure5_supp.py` builds a whole precompute
layer specifically to avoid repeating this work. The correct pattern already exists twice
in the repo: `occupancy_matched_estimator.centres_from` and
`figures/fig1/rate_maps.compute_field_centers` both smooth once and derive both centres.

```python
sm   = [smooth_and_norm(rm[n]) for n in range(hidden_dim)]
com  = np.array([field_center_of_mass(s) for s in sm], dtype=float)
peak = np.array([field_peak(s)           for s in sm], dtype=float)
```

### 5. `lib/occupancy_matched_estimator.py:196-208` — `centres_from` computes a COM nobody uses


**Confidence:** **Certain.** All four call sites grepped; every one discards the second return value.
Every one of its four call sites discards the second return value:

```
lib/occupancy_matched_estimator.py:326   pk_pre,  _ = centres_from(rm_pre)
lib/occupancy_matched_estimator.py:327   pk_post, _ = centres_from(rm_post)
lib/extinction_analysis.py:64            pk,       _ = centres_from(t["rm"])
lib/extinction_analysis.py:85            pk_learn, _ = centres_from(triple[1]["rm"])
```

`field_center_of_mass` runs 512 times per pass and its result is thrown away. `_single_pass`
is the innermost loop of the estimator validation (per seed: 4 modes + 2 nulls + 4 size
fractions, up to 20 reps each), so this is a large constant factor on the most expensive
cache in the repo. Add `want_com: bool = False`, or split out a `peaks_from`.

### 6. `aggregate/run_estimator_validation.py:127-132` — `setdefault` with an eagerly-built value, on the wrong mask


**Confidence:** **Certain** about the code. **Verified inert on current data**: `est_naive`/`est_joint` are non-null for every seed in all four blocks, so the mask is all-True and the `gt` means and p-value do describe the same 34 seeds. A latent trap, not a wrong number today.
```python
for mode in ("naive", "joint"):
    ...
    ok = np.isfinite(a) & np.isfinite(b)
    ords[mode] = { …, "paired_p": float(wilcoxon(a[ok], b[ok]).pvalue) if ok.sum() > 4 else None }
    ords.setdefault("gt", {
        "nohead_mean": float(g_a.mean()), "head_mean": float(g_b.mean()),
        "head_more_toward": bool(g_b.mean() < g_a.mean()),
        "paired_p": float(wilcoxon(g_a[ok], g_b[ok]).pvalue),
    })
```

Three problems in one statement:

* Python evaluates the default argument on **every** iteration, so the ground-truth
  Wilcoxon is computed twice and the second result discarded.
* The surviving `"gt"` entry is the one built during `mode == "naive"`, so its `paired_p`
  is computed over the seeds where `est_naive` happened to be finite, while its
  `nohead_mean` / `head_mean` / `head_more_toward` use **all** seeds. The reported p-value
  and the reported means describe different seed sets.
* The mode-level call is guarded by `if ok.sum() > 4`; the `gt` call is not. If the naive
  estimator fails on enough seeds, `wilcoxon` on a near-empty (or all-zero-difference)
  array raises and takes down the whole aggregation.

Hoist the `gt` block out of the loop and compute it over all seeds.

### 7. `regenerate_all.py:6`, `sync_paper_figures.py:89` — the de-caps pass rewrote a filename


**Confidence:** **Certain.** `ls` confirms `CLEANUP_TODO.md` at the repo root; `README.md` cites it correctly while the two touched files do not.
```
regenerate_all.py:6     different run from the shipped exemplar (see cleanup_todo item 4).
sync_paper_figures.py:89 # see cleanup_todo.md.
```

The file is `CLEANUP_TODO.md` at the repo root. `README.md:218` and `:240` still cite it
correctly as `../CLEANUP_TODO.md`, and `figures/figS_legacy/README.md:44` as
`CLEANUP_TODO.md`, so the repo now spells the same file three ways and two of them do not
exist on a case-sensitive filesystem. Same class of damage elsewhere in the diff:
`lib/pubstyle.py:246` now reads `"""apa-ish p formatting…` (APA is an acronym), and
`external/krishnan2022/utils.py:162` came out half-converted as
`exact BLOCK parsed:`. A mechanical de-caps should exclude acronyms and file names.

### 8. `figures/figS_legacy/plot_replay_v11.py:17` (+ `:679`, `lib/plot_replay_v11.py:234`, `runs/prune_run.py:16-17`) — touched usage lines point at a tree the reader will not have


**Confidence:** **Certain** that the paths point at a git-ignored tree (`git check-ignore` + README). **Low** severity: documentation, and only bites a reader who copy-pastes.
```
python paper_code/figures/figS_legacy/plot_replay_v11.py --run_dir New_Results_v2/final_v11_fixed --no_show
```

This line was edited by the diff (the script path was corrected) but the run directory was
left as `New_Results_v2/`, which `.gitignore` excludes wholesale (`/New_Results_v2/`) and
which `README.md:225-229` declares superseded: *"Do not use them to regenerate anything."*
`generate_figS_fate.py:42`, the wrapper that actually drives this script, defaults to
`New_Results_v3/final_v11_fixed`. Same in `runs/prune_run.py`'s Usage block (also inside a
touched hunk) and in both `--run_dir` argparse help strings. A reader who clones the repo
cannot run any of these as written.

### 9. `figures/fig1/generate_figure1_supp.py:50-61`, `figures/fig3/generate_figure3_supp.py:54-63` — the `importlib` work-around is a no-op, duplicated twice


**Confidence:** **Certain** that it is a no-op: I compared every rebound symbol across the three `utils.py` modules; all re-export from `lib/pubstyle.py` and the two constants are defined identically. **Certain** that the README paragraph is now stale.
The diagnosis in the new comment is correct (`from analysis import …` binds `utils` to
fig2's module in `sys.modules`, so a later bare `from utils import …` can never reach
fig1's/fig3's). The fix buys nothing, because all three `utils.py` modules re-export the
same objects from the same place:

```
figures/fig{1,2,3}/utils.py:  from pubstyle import (… apply_nature_style, ensure_dir, save_figure,
                                                    PAGE_W, MM_TO_PT …)
figures/fig{1,2,3}/utils.py:  PAGE_MARGIN   = 1.0 * MM_TO_PT
figures/fig{1,2,3}/utils.py:  DEFAULT_REWARD = (0.3, 0.3)
```

Every one of the seven (resp. six) rebound names is the identical function object or the
identical value. What the change actually adds is 12 lines of loader machinery per file, a
second execution of `utils.py`, and a `sys.modules["fig1_utils"]` / `["fig3_utils"]` entry —
copy-pasted into two files. The root cause is that ten figure directories each ship a
module named `utils` and seven ship one named `analysis`; `README.md:236-240` already
records this and `smoke_test.py:106-115` already has to spawn a subprocess per module
because of it. If the shadowing is to be fixed, fix it once — make `figures/` a package, or
give each module a unique name — rather than adding a per-file special case that currently
changes no value.

It also leaves the README stale. `README.md:237-240`, under "Known rough edges", still says:

> …but `figure1_supp` and `figure3_supp` silently pick up fig2's `utils`. **Preserved
> as-is during the consolidation**; converting the tree to a real package would remove
> the hazard.

The diff un-preserves exactly that, in exactly those two files, and did not update the
paragraph — even though `README.md` is itself modified by this diff.

### 10. `lib/plot_goal_directed_maps.py:19` — hard sibling dependency next to an optional one


**Confidence:** **High.** Read, not executed -- I did not try importing the module with `lib/` off the path. The asymmetry with the guarded `pubstyle` import is plain in the source.
```python
from ratemaps import (  # noqa: F401
    build_rate_maps, compute_skaggs, smooth_and_norm, field_center_of_mass,
)

try:
    from pubstyle import RATEMAP as _RATEMAP, …
except ImportError:  # pragma: no cover - stand-alone use
    _RATEMAP, _DIVERGING, _SEQUENTIAL = "magma", "coolwarm", "magma"
```

The module used to define the four primitives inline, so `pubstyle` was its only sibling
and the `try/except` made it genuinely stand-alone (the 916-line `figures/figS_legacy/`
copy exists precisely because that property was relied on). Now an unguarded
`ModuleNotFoundError: ratemaps` fires first and the `# pragma: no cover - stand-alone use`
comment is false. Either guard both or drop the `pubstyle` fallback and say the module
requires `lib/` on the path.

### 11. `lib/plot_goal_directed_maps.py:46, 59-62` — dead parameters documented instead of deleted


**Confidence:** **Certain** about the dead parameters and the wrong "both call sites" claim (one call site in this module). **Low** severity.
The diff removed the unused `title_suffix` parameter from `fig_comparison` but kept
`skaggs_rw` / `skaggs_gc`, adding a docstring that says so:

> Accepted but unused — unpacked in the draw loop and never read. Kept because both call
> sites pass them positionally.

There is exactly one call site in this file (line 569) and it passes them positionally, so
"both call sites" is already wrong; the `figS_legacy` copy is a separate module with its own
`fig_comparison`. The loop variable `sk` in `for ax, rm_src, sk in [...]` is likewise dead.
Since `title_suffix` was deleted outright in the same hunk, the inconsistent treatment is
harder to justify than either option alone.

### 12. `lib/reward_relative_analysis.py:1-16` — the method record was deleted from the only place it was written down


**Confidence:** **Certain** that the content was removed (diffed against `HEAD`) and that the replacement sentence is ungrammatical. **Low-to-Medium** severity: a judgement call about how much a method docstring is worth.
The rewritten module docstring drops the classification rule that `readout()` implements:

```
reward-relative : r_reward > r_place  AND  r_reward > null 95th pct
place-anchored  : r_place  > r_reward AND  r_place  > null 95th pct
remap-other     : neither exceeds the null
RRI = r_reward - r_place
frame window u in [-0.3, 0.3]^2 (~19x19 bins);  Skaggs gate 0.15 in both periods;
teleport transients trimmed from behavioural goal periods
```

and replaces it with "is classified by which wins against a cell-identity shuffle null",
which omits the null threshold, the third class and the index definition. The replacement
sentence is also ungrammatical: *"The latter is primary reward-frame sampling is then
identical by construction."* (a dropped `--`). The same trimming removed, from
`lib/occupancy_matched_estimator.py`, the validation targets and the "inference must use
per-seed rep-means only" rule, and from `lib/extinction_analysis.py` the "per-seed ratios
are unstable; never average ratios" rule, the 1000-step block size and the 5.5 entropy
gate. For a repository whose stated purpose is reproducing published figures, these
docstrings are the method section; deleting them is a real loss even though no code changed.

### 13. `lib/ratemaps.py`, `lib/stats_helpers.py` — new files are untracked while the rest of the refactor is staged


**Confidence:** **Certain.** `git status` shows `??` for both; `git check-ignore` exits 1, so they are untracked rather than ignored.
```
RM paper_code/lib/common.py -> paper_code/lib/dataframe_io.py
?? paper_code/lib/ratemaps.py
?? paper_code/lib/stats_helpers.py
```

The `common.py` → `dataframe_io.py` rename is already in the index; the two genuinely new
modules are not, and neither is matched by `.gitignore` (`git check-ignore` exits 1).
Committing as-is records every `from ratemaps import …` / `from stats_helpers import …`
rewrite without the modules they point at, and the clone breaks at import for
`analysis_helpers`, `plot_goal_directed_maps`,
`plot_replay_v11`, `occupancy_matched_estimator`, `reward_vector_analysis`,
`successor_representation`, `figures/fig1/rate_maps.py` and `figures/fig4/analysis.py` —
i.e. essentially the whole pipeline. `git add paper_code/lib/ratemaps.py
paper_code/lib/stats_helpers.py` before committing.

### 14. `figures/figS_legacy/generate_figS_fate.py:5` — stale line count in a rewritten sentence


**Confidence:** **Certain.** `wc -l` gives 916 / 334, and `git show HEAD:` gives 321. **Low** severity.
> …(the 916-line standalone copy, not the trimmed **313-line** `paper_code/lib/` module of
> the same name)

`figures/figS_legacy/plot_replay_v11.py` is 916 lines (correct). `lib/plot_replay_v11.py`
was 321 lines at `HEAD` and is **334** after this diff. The hunk rewrote the surrounding
prose and left the number. Since the number is the stated way to tell the two files apart,
either update it or drop it in favour of the path.

### 15. `lib/successor_representation.py:57` — default output directory writes into a superseded tree


**Confidence:** **Certain** about the default. **Low** severity -- it is `main()`, which the module's own docstring says nothing reads.
```python
parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=str,
                    default="../New_Results/successor_representation")
```

`New_Results/` is git-ignored (`/New_Results/`) and `README.md:225-229` marks it
superseded. The default is also `..`-relative to the caller's CWD, so it lands somewhere
different depending on where the script is run from. The module docstring added by this
diff audits `main()` honestly ("not wired into `regenerate_all.py` or `smoke_test.py`, and
its `--reward-x/y` default of 0.75 matches no condition in the current runs") but stops
short of this default. Either point it at `Repro/` or, given the docstring's own verdict,
delete `main()` and keep the library surface.

---

## Checked and clear

* **`ratemaps.py` extraction is numerically faithful.** `build_rate_maps`, `compute_skaggs`,
  `smooth_and_norm`, `field_center_of_mass` and `field_peak` were compared line-for-line
  against the three copies they replace (`figures/fig1/rate_maps.py`,
  `lib/plot_goal_directed_maps.py`, `lib/plot_replay_v11.py`). Identical, including the
  `(y, x)` swap in both centre functions — which the new docstring documents accurately,
  and which stays distance-preserving because every reward in the repo is on the diagonal
  ((0.3,0.3), (0.7,0.7), (0.5,0.5)).
* **`compute_skaggs`' non-standard `r_bar`.** The new docstring correctly flags that the
  canonical implementation uses an *unweighted* mean over all m×m bins (including
  never-visited zeros), not the occupancy-weighted mean of the textbook Skaggs formula, and
  points at `occupancy_matched_estimator.skaggs_weighted` for the weighted form.
* **`lib/common.py` deletion.** `DEFAULT_REWARD`, `load_goal_position`,
  `compute_distance_to_reward`, `bin_positions`, `maybe_subsample_indices`,
  `bootstrap_mean_ci`, `cohen_d_independent` and `cohen_d_paired` were dropped and are
  referenced nowhere in `paper_code/`. `successor_representation.py` imported
  `DEFAULT_REWARD` and `cohen_d_independent` but never used them (it has its own
  `load_goal_position` at line 606). `model/*.py` define their own `ensure_dir` / `save_json`
  and never imported `common`.
* **`sr_common.py` removals.** `GOAL_RADIUS` is only used in `model/`, which defines its own
  (`0.05`, not the `0.15` that was deleted). The `_get_shared_random` alias is safe to drop:
  the sole consumer already imports `get_shared_random as _get_shared_random`
  (`figures/fig5/sr_direction_analysis.py:43`).
* **`frozen_replay_gate`'s dropped `min_cells`.** The parameter was accepted and ignored by
  the old body too, and the only caller (`frozen_replay_delta_from_centers`) applies the
  threshold itself.
* **Shebang removals** from `lib/extinction_analysis.py`, `lib/occupancy_matched_estimator.py`
  and `lib/reward_relative_analysis.py` are harmless — none of the three is executable
  (`-rw-r--r--`) and nothing invokes them by path.
* **Legacy import shadowing.** `figures/figS_legacy/plot_replay_v11.py` prepends its own
  directory to `sys.path` before `from plot_goal_directed_maps import …`, so it keeps
  binding the legacy sibling in its own directory (which keeps its own
  `fig_comparison(… title_suffix …)` at
  `figures/figS_legacy/plot_goal_directed_maps.py:78`), not the `lib/` module.
* **No plot-facing strings were changed.** Every `+`/`-` pair in the de-caps pass lands
  inside a comment, a docstring or a Markdown file; no axis label, title or legend text was
  touched.

---
---

<a name="whole-codebase-audit-beyond-the-diff"></a>

# Part 2 — Whole-codebase audit (beyond the diff)

**Scope.** All of `paper_code/` except `model/`: 23,013 lines across 82 Python files
(`lib/` 13, `runs/` 11, `figures/` 39, `aggregate/` 3, `external/` 5, top level 4, `tools/` 1).
Static review plus verification against the three committed caches
(`Repro/aggregate/occupancy_matched_estimator.json`, `Repro/aggregate/extinction_summary.json`,
`Repro/multi_seed_reloc/aggregate/reward_relative_summary.json`,
`external_data/sosa2025/cache/sosa_tests.json`) and standalone reproduction scripts for the
mechanisms below. No code changed; no model run.

**Headline.** The diff was clean. The codebase underneath is not. There is **one finding that
makes a shipped supplementary result wrong in a direction that favours the paper's own
conclusion** (A1), one that makes a core gate non-comparable between the two arms of several
contrasts (A2), and one that would silently convert an honestly-chosen exemplar into a
cherry-picked one (A3). The paper's **primary** H1-vs-H2 frozen-replay result is immune to
A1 and A2, and I say why below.

---

## A1. `figures/fig5/sr_direction_analysis.py:110-121` — the SR direction test compares two different coordinate systems


**Confidence:** **Certain**, and the highest-confidence finding in this report. Both coordinate conventions are traced to the lines that set them, and the mismatch is reproduced numerically (a shift in perfect agreement with SR is scored at 143 deg). The consuming panel is traced. The only thing I could not do is measure the size of the error on the real runs, because no run data survives -- but the test statistic is an angle, so a reflection is not a small perturbation of it.
**This is the most serious finding in the repo.**

The RNN field centres and the SR state grid use opposite axis orders, and
`seed_direction_metrics` compares them directly as vectors.

*RNN side.* `ratemaps.build_rate_maps` indexes maps as `rm[x_bin, y_bin]`
(`gc = x_bin * m + y_bin`). `field_center_of_mass` then weights by `xs[None, :]` — which
broadcasts along **axis 1 = y_bin** — and by `ys[:, None]` — **axis 0 = x_bin**. So it
returns `(mean_y, mean_x)`. `field_peak` swaps identically. Both therefore return **(y, x)**:
a reflection about the line *y = x*. `lib/ratemaps.py`'s own docstring states this.

*SR side.* `pos_to_interior_idx(x, y)` sets `rr` from `x` and `cc` from `y`;
`state_centers` sets `out[idx,0] = r/m`, `out[idx,1] = c/m`. So `state_xy`, and the vectors
`sr_predicted_shift_vectors` returns, are in true **(x, y)**. `transition_from_behavior_positions`
reads `pos[:,0], pos[:,1]` straight from the behaviour frame — also unswapped.

Measured directly:

```
true arena position                                             : (0.25, 0.75)
ratemaps.field_peak / field_center_of_mass                      : (0.766, 0.266)   <- (y, x)
successor_representation.state_centers[pos_to_interior_idx(x,y)]: (0.250, 0.719)   <- (x, y)

angle(true shift, SR prediction = that same true shift)         :   0.0 deg
angle(shift as the code measures it, same SR prediction)        : 143.1 deg
```

Two independent corruptions inside one function:

```python
com_a = phases["A"]["com"][ids]                     # (y, x)
com_b = phases["B"]["com"][ids]                     # (y, x)
actual_shift = com_b - com_a                        # reflected about y = x

state_idx  = nearest_state_for_points(com_a, state_xy)   # (y,x) point looked up in an (x,y) grid
pred_shift = pred_vec[state_idx]                          # true (x, y)

angles = angle_between_vectors(actual_shift, pred_shift)  # reflected vs unreflected
```

1. Every cell not on the diagonal is assigned its **mirror-image SR state**, so it receives
   the wrong SR prediction to begin with.
2. The angle then compares a reflected vector against an unreflected one.
3. `residual_mag` — "the actual shift component orthogonal to the SR prediction" — inherits both.

`dist_to_reward` in the same function is fine, because `REWARD = (0.3, 0.3)` is on the
diagonal and the reflection preserves distances to diagonal points. That is exactly why this
has survived: everything distance-based looks right.

**What it breaks.** `figures/fig5/generate_figure5_supp.py::fig_supp_sr_direction`, i.e.
**figure 5 supplementary panel c** in full: the pooled angle histogram, the `mean R`
Rayleigh statistic ("0 = perfectly aligned with SR prediction"), and the
residual-vs-distance Spearman.

**Why it matters more than a normal bug.** The panel exists to show that SR does *not*
explain the observed remapping. Reflecting one of the two vector fields destroys any true
alignment, so the bug manufactures exactly the negative result the panel reports. This is a
false negative that **supports the paper's thesis** — the most dangerous kind to leave in.

The fix is one line at the boundary (swap `com` to `(x, y)` before the SR comparison), but
the fix must be applied consistently: `nearest_state_for_points` and `angle_between_vectors`
must both receive true `(x, y)`.

## A2. `lib/ratemaps.py:58` — the canonical Skaggs score depends on how much of the arena the trajectory covered


**Confidence:** **Certain** for the mechanism -- reproduced on a synthetic cell where only coverage varies (3.99 -> 7.93 -> 16.03 while the standard form falls). **Medium** for the impact: which contrasts are immune is Certain (same-trajectory replay cancels it exactly, and that covers the primary result), but the realized coverage gap between `cycle_1` and `cycle_1_goal` on real runs is **unmeasured** -- my demo uses an extreme 4x coverage difference that real trajectories will not reach. Treat the direction as established and the magnitude as open.
`compute_skaggs` normalises by `rate_flat.mean(axis=0)` — the **unweighted** mean of the
unit's map over all `m*m = 1024` bins, *including never-visited bins, which hold 0* — rather
than the occupancy-weighted mean `Σ p_i r_i` of the standard Skaggs formula. The less of the
arena a period covers, the more zero bins deflate `r_bar`, and the higher the score.

Measured on a synthetic cell whose field, tuning and activity are held **identical** while
only the trajectory's coverage changes:

```
                              bins visited   repo Skaggs   occupancy-weighted Skaggs
broad coverage (phase A)       1024/1024         3.994              3.986
half coverage                   529/1024         7.925              2.996
narrow coverage (phase B)       256/1024        16.026              1.993
```

The repo's score **rises 4x** as the same cell is sampled over a quarter of the arena; the
standard form correctly falls. So an absolute threshold (`0.05` / `0.15` / `0.3`) is not the
same gate in two periods with different coverage.

**Immune — including the paper's primary result.** Every frozen-replay contrast replays one
fixed trajectory through two weight snapshots, so coverage is *identical in both arms* and
the distortion cancels exactly: `replay_c1_pred` vs `replay_c1_postrl` (figs 2/3/5, the
H1-vs-H2 diagnostic), `replay_reloc_pred/postA/postB` (fig 4), the extinction triple, and
`figS_sr`/`figS_rigid`. This is the single most important thing to say about A2: **the
headline claim does not rest on it.**

**Exposed.** Contrasts whose two arms have different coverage by construction:
`PHASE_A = "cycle_1"` (free exploration, broad) vs `PHASE_B = "cycle_1_goal"` (goal-directed,
concentrated) in `shared_place_cell_ids` (figs 1, 2, 3 and `sr_direction_analysis`); the
`("cycle_1", "cycle_1_goal")` behavioural arm of `readout_dissociation`; and the cell
selection inside `peak_entropy`, which feeds the 5.5-bit collapse gate everywhere. Because
`shared_place_cell_ids` requires the threshold in **both** A and B, the binding constraint is
the broad-coverage phase — the gated population is effectively defined by phase A.

The repo already contains two correct occupancy-weighted implementations
(`occupancy_matched_estimator.skaggs_weighted`, `figS_legacy/pc_common.compute_skaggs_scores_optimized`),
so this is an inconsistency, not a missing capability. Note the diff **deleted** the one
comment that warned about it, in `occupancy_matched_estimator.py`'s old docstring:
*"the canonical compute_skaggs in plot_goal_directed_maps uses an unweighted mean over bins
-- do not use it here."*

## A3. `runs/select_best_seed_fig4.py:143,240-252` — the documented command replaces the honest exemplar with a cherry-picked one


**Confidence:** **Certain** about the code (sort key, `rows[0]`, the `subprocess.run` that regenerates in place) and about the README quoting both the median convention and the command that breaks it. **Medium** on whether it has ever actually happened -- `regenerate_all.py` pins seed 9, so the shipped figure looks correct today. This is a trap, not evidence of past cherry-picking, and I want to be clear I am not alleging the latter.
```python
rows.sort(key=lambda r: r["tracking_score"], reverse=True)   # line 143
...
best = rows[0]
best_run_dir = MULTI_SEED_DIR / f"reloc_seed{best['seed']}"
subprocess.run([sys.executable, str(FIG4_DIR / "generate_figure4.py"),
                "--run_dir", str(best_run_dir), "--output_dir", str(fig4_out), ...])
```

`tracking_score` is the harmonic mean of `frac_toward_A` and `frac_toward_B` — **the very
quantity figure 4 reports**. The script ranks all clean seeds by it, takes rank 1, and
regenerates figure 4 in place.

But the shipped exemplar is `reloc_seed9`, and `figures/fig4/README.md` documents it as *"the
**MEDIAN** entropy-gate-passing seed by tracking quality (`reloc_seed9`, rank 17 of 32 clean
seeds)"* — a deliberate, defensible choice. The same README then instructs:

```
# Pick best seed and regenerate:
python paper_code/runs/select_best_seed_fig4.py
```

Running the repo's own documented command therefore silently upgrades panel a from
*median-of-32* to *best-of-32*, with no warning and no record in the output. `regenerate_all.py`
pins `reloc_seed9` explicitly, so the canonical path is safe — but nothing stops the README
path, and nothing in the script mentions the median convention.

`select_best_seed_fig2.py` / `_fig3.py` have the same auto-regenerate structure but rank by
**COM spatial entropy** (`rows.sort(key=lambda r: r["entropy"])`), which is effect-independent
and therefore legitimate. They still overwrite the pinned exemplars: there are now three
different answers to "which run is figure 2" — `fig2/utils.py` `DEFAULT_RUN = nohead_seed42`,
`regenerate_all.py` pins `nohead_seed16`, and `select_best_seed_fig2.py` writes whichever seed
currently has the highest entropy.

## A4. `figures/fig5/generate_figure5_supp.py:290-292` — pseudoreplicated p-value


**Confidence:** **Certain** that every cell from every seed and both conditions is concatenated before `spearmanr`. **High** that the p-value is therefore uninterpretable -- that is a textbook point, not a judgement call.
```python
all_residual = np.concatenate([r["residual_mag"] for r in rows_nh + rows_hd])
all_dist     = np.concatenate([r["dist_to_reward"] for r in rows_nh + rows_hd])
r_pooled, p_pooled = spearmanr(all_residual, all_dist)
```

Every cell from every seed **and both conditions** is pooled into one correlation, and the
resulting `p` is printed on the panel. Cells within a seed are not independent, so *n* is
inflated by roughly three orders of magnitude and the p-value is not interpretable. This is
out of step with the rest of the repo, which is otherwise careful to aggregate at the seed
level and says so repeatedly ("pairing: same seed"). The per-seed machinery already exists —
`per_seed_R` is computed two lines above and only used for a mean.

`figures/figS_reward_vector/generate_figure_reward_vector_supp.py:186-193` pools cells the
same way for a bootstrap CI, but **declares** it ("the pooled cell-level circular mean +-
bootstrap CI ... far better powered than eyeballing the noisy per-seed scatter") and pairs it
with a proper per-seed Wilcoxon. Declared trade-off, not a hidden one.

## A5. `external/sosa2025/analysis_sosa.py:62-63` — a hand-copied model constant crossing an analysis boundary


**Confidence:** **Certain** that the constant is hard-coded and unguarded, and **Certain** that it currently matches the live cache (measured: 0.0643 vs 0.064, 0.0314 vs 0.031). **Medium** as a live problem: nothing is wrong today, it is a coupling waiting to break. The n=7 equivalence reading is a separate **High**-confidence concern.
```python
MODEL_RR_FRAC = 0.064      # model comparison band (32 seeds)
MODEL_RR_SD   = 0.031
```

`fig5` panel e plots the model mean computed **live** from
`Repro/multi_seed_reloc/aggregate/reward_relative_summary.json`, but the p-value it prints
comes from `so["summary"]["wilcoxon_rrfrac_vs_model"]`, a one-sample Wilcoxon of the 7
sessions against this **frozen constant**.

They currently agree — live mean `0.0643 ± 0.0314` over 32 nets vs the hard-coded
`0.064 ± 0.031` — so no published number is wrong today. But nothing enforces it, and
`aggregate/run_reward_relative_analysis.py`'s own header says the driver is *"not yet
validated against the committed cache -- run with --out to a scratch path and diff before
trusting it"*. Regenerate that cache and the panel will plot one model mean while quoting a
p-value against another, silently. One `assert` (or reading the JSON) closes it.

Related, on the same panel: `p = 0.8125` at *n = 7 sessions* is presented as model/data
agreement. Failing to reject with 7 paired observations is very weak evidence of
equivalence; the panel shows both distributions, which helps, but the text reads as a
positive result.

## A6. `figures/fig2/analysis.py:260-266` — panel b is mirrored and its axes are mislabelled


**Confidence:** **Certain.** `com` is `(y, x)` (proven in A1), the labels are read from source, and I checked `_draw_arena` to rule out a compensating transform -- it only sets `xlim`/`ylim` to (0,1) and the spines. Nothing undoes the reflection.
```python
ax.scatter(com[:, 0], com[:, 1], ...)
axes[0].set_xlabel("Arena x")
axes[0].set_ylabel("Arena y")
```

`com` is `(y, x)` (see A1), so the plotted horizontal axis is arena **y** and the vertical is
arena **x**: the field-centre cloud is reflected about the diagonal and both labels are
wrong. The reward marker drawn by `_draw_arena(ax, R, ...)` uses `R = (0.3, 0.3)`, which is
on the diagonal and therefore lands in the right place — which is precisely what makes the
panel look self-consistent while every cell is mirrored. Any genuine asymmetry in the field
distribution (wall preference, corner clustering) is shown flipped.

## A7. `figures/figS_sr/analysis.py:64-69` — the two mirror nulls are labelled the wrong way round


**Confidence:** **Certain.** Computed the four distances explicitly; the two off-diagonal labels are exchanged. **Low** severity -- I say in the finding why the conclusion survives.
```python
MIRROR_LOCATIONS = {"A (0.3,0.3)": [0.3,0.3], "(0.7,0.3)": [0.7,0.3],
                    "(0.3,0.7)": [0.3,0.7],   "B (0.7,0.7)": [0.7,0.7]}
```

`dd_at::<name>` measures distances from the `(y, x)` centres, so for a cell truly at
`(0.25, 0.75)`:

```
  A (0.3,0.3): true 0.4528   as computed 0.4528   (on-diagonal, fine)
    (0.7,0.3): true 0.6364   as computed 0.0707   <- actually the (0.3,0.7) distance
    (0.3,0.7): true 0.0707   as computed 0.6364   <- actually the (0.7,0.3) distance
  B (0.7,0.7): true 0.4528   as computed 0.4528   (on-diagonal, fine)
```

The two off-diagonal mirror nulls are exchanged with each other. The *conclusion* survives —
the set of mirror values is unchanged, so "the true reward stands out against its mirrors"
still holds — but the two per-location numbers in panel d carry each other's labels.

## A8. `aggregate/run_extinction_analysis.py` + `figS_extinction` panel b — a dissociation claimed from two separate marginal tests


**Confidence:** **Certain** about the statistical structure: two marginal tests, no interaction test, no correction, strata defined by one arm. **Low** that the substantive claim is wrong -- effect sizes differ 10x (0.018 vs 0.180), so running the proper test would most likely confirm it. This is a rigour finding, not a correctness one, and I would not hold up a submission for it.
The panel title asserts *"Reliability drop is concentrated in FAR cells (divergence from
Krishnan's zone-specific drop)"*. From the committed cache:

```
  all : postLearn 0.7473  postExt 0.6178  diff +0.1295  p = 0.0081
  near: postLearn 0.6575  postExt 0.6393  diff +0.0182  p = 0.8926
  far : postLearn 0.7885  postExt 0.6084  diff +0.1801  p = 0.0017
```

The near/far dissociation is inferred from "near is n.s., far is significant" — the classic
error of treating a difference between significant and non-significant as itself
significant. The correct test (a within-seed near-vs-far difference-of-differences, paired
across the 13 seeds) is never run, although the per-seed values needed for it exist. No
correction across the three strata either, though at p = 0.0017 that would not change the
verdict. The effect sizes (0.018 vs 0.180) do look convincingly different — which is why
running the interaction test would likely *strengthen* the claim rather than weaken it.

Two supporting weaknesses in the same panel: the strata are defined by **postLearn** peak
distance (`near = dist_learn < 2*NEAR_RADIUS`), i.e. by one of the two periods being
compared; and `k2_reliability` returns only means, so per-seed near/far **cell counts are
never reported** — a reader cannot tell whether "near, n.s." reflects a small effect or a
small stratum. (Reward A is at `(0.3, 0.3)` with a 0.30 radius, so `near` is roughly a fifth
of the arena.)

## A9. `aggregate/run_extinction_analysis.py:64-75` — `_wstat_ratio` documents a test it never runs, and returns an inconsistent dict


**Confidence:** **Certain** (the function is ten lines). **Low** severity: currently unreachable because no caller asks for the missing key.
```python
def _wstat_ratio(x):
    """... the paired/one-sample Wilcoxon is rank-based and unaffected."""
    if len(x) == 0:
        return {"n": 0, "mean": None, ..., "p": None}        # has "p"
    return {"n": ..., "mean": ..., "sd": ..., "median": ..., "iqr": ...}   # no "p"
```

No p-value is ever computed despite the docstring, and the empty-input branch returns a key
the normal branch omits — so `result["p"]` is a `KeyError` exactly when there is data. Not
currently triggered (`figS_extinction` reads `postLearn_vs_postExt["p"]`, which exists), but
it is a trap. Credit where due: `fig_k3` correctly plots the **median** (19.3x → 1.4x), not
the mean, which is `610.98 ± 2084.67` — the heavy tail the helper was written to handle.

## A10. Three Skaggs implementations, two different formulas


**Confidence:** **Certain.** Four implementations located by grep, formulas read. Severity follows from A2, so it inherits A2's **Medium** impact rating.
| implementation | formula | used by |
|---|---|---|
| `lib/ratemaps.py:58` | unweighted `r_bar` over all bins | everything in `lib/`, `figures/`, `runs/`, `aggregate/` |
| `lib/occupancy_matched_estimator.py:182` | occupancy-weighted (standard) | the matched estimator, extinction, reward-relative |
| `figures/figS_legacy/pc_common.py:72` | occupancy-weighted (standard) | the two legacy supplementary figures |

Plus `figures/figS_legacy/plot_goal_directed_maps.py:50`, a fourth copy of the unweighted
one. Two conventions coexist, gates are quoted as bare numbers (`0.05` / `0.15` / `0.3`)
without saying which formula they belong to, and the diff removed the only warning comment.
See A2 for why the difference is not cosmetic.

## A11. `aggregate/run_reward_relative_analysis.py:113-115` — lexicographic seed ordering


**Confidence:** **Certain** about the ordering difference; **Certain** it is harmless as currently written, since pairing is by position within one list. **Low**.
```python
seed_dirs = sorted((d for d in reloc_dir.iterdir() if ...), key=lambda d: d.name)
```

`reloc_seed0, reloc_seed1, reloc_seed10, reloc_seed11, ..., reloc_seed2, ...`. Harmless as
written — `_paired_p` zips rows from the same list, so pairing survives — and the comment
says it is deliberate ("matching the seed order in the committed cache"). But
`run_extinction_analysis.py` sorts numerically (`sorted(int(p.name.replace(...)))`), so the
two caches order seeds differently, and any future code that zips one against a numerically
ordered seed list will mis-pair silently.

## A12. Reproducibility ceiling: there is no raw data in the tree


**Confidence:** **Certain.** `find` returns zero pickles and zero `data/` directories. Not a defect -- the README declares it -- but it bounds what anyone, including me, can verify.
```
find Repro New_Results_v3 -name "*.pkl"  ->  0
find Repro -maxdepth 3 -type d -name data ->  0
```

Zero pickles and zero `data/` directories survive. `README.md` states this as a design
choice ("Raw run data ... git-ignored — regenerate them with the commands above"), and it is
a reasonable one for a 100 GB+ tree. The practical consequence is worth stating plainly
anyway: **only the figures fed by the four committed JSON caches can be regenerated by a
reader.** Everything that reads `df_behavior_*.pkl` / `df_activations_*.pkl` — figures 1, 2,
3, 4, most of 5, `figS_sr`, `figS_rigid`, `figS_reward_vector`, `figS_legacy` — requires GPU
retraining first. That also means A1 and A2 cannot be re-measured on the real runs from this
checkout; both are demonstrated above on synthetic inputs instead.

---

## Priority order

1. **A1** — fix before the figure is cited; it manufactures the negative result it reports.
2. **A2** — decide one Skaggs convention; re-check the A→B gated populations. The primary
   frozen-replay result is unaffected, so this is important but not urgent.
3. **A3** — add a guard, or change `select_best_seed_fig4.py` to pick the median it claims.
4. **Part 1, finding 1** (`frac < 1.0` un-matching) and **A4** — both make a supplementary
   panel measure the wrong thing.
5. **A5**, **A6**, **A7** — cheap, mechanical, and each removes a way for a correct analysis
   to be reported incorrectly.
6. Everything else is hygiene.

---

# Part 3 — Candidate fixes, measured then reverted

> **The working tree is unmodified.** Nothing in `lib/` or `figures/` carries a review
> change. Each patch below was applied only long enough to measure it against a
> reconstruction driven by the repo's own code, then reverted. They are recorded here as
> proposals for the PI to accept or decline — and **measuring them corrected Part 1 and
> Part 2 on the substance of both findings**, so read the measurement before deciding.
>
> Figures: `paper_code/code_review_figures/` (PDF + PNG). Git-ignored and regenerable;
> they are diagnostics, not paper figures. No existing figure was overwritten, and no
> model was run.

## The candidate patches

### P1 — `lib/occupancy_matched_estimator.py:295` (Part 1, finding 1)

```python
# as it stands
if frac < 1.0:
    k_pre = rng.choice(len(i_pre), max(50, int(len(i_pre) * frac)), replace=False)
    k_post = rng.choice(len(i_post), max(50, int(len(i_post) * frac)), replace=False)
    i_pre, i_post = i_pre[k_pre], i_post[k_post]

# proposed -- match_indices returns the two arrays positionally aligned, so one
# shared draw keeps the matching that two independent draws void
if frac < 1.0:
    k = rng.choice(len(i_pre), max(50, int(len(i_pre) * frac)), replace=False)
    i_pre, i_post = i_pre[k], i_post[k]
```

### P2 — `figures/fig5/sr_direction_analysis.py:111` (A1)

```python
# as it stands
com_a = phases["A"]["com"][ids]
com_b = phases["B"]["com"][ids]

# proposed -- ratemaps emits (y, x); state_xy and every vector in pred_vec are (x, y)
com_a = phases["A"]["com"][ids][:, ::-1]
com_b = phases["B"]["com"][ids][:, ::-1]
```

With both applied, `smoke_test.py` was 32 ok / 1 warn / 0 FAIL (imports) and 40 ok /
0 FAIL (cli) — unchanged from without them.

## Finding 1 — the mechanism is confirmed; the *consequence* was stated wrongly

Reconstruction at real scale (4525 timesteps/period, 32x32 grid x 8 headings, 512 units,
n_reps=30, 6-10 seeds), calling the real `strata_ids`, `match_indices` and `_single_pass`;
only the two lines under test were swapped. Ground truth was planted as **zero** shift —
identical place fields in both periods, only occupancy differs — so the correct curve is
flat at 0 and any departure is bias.

| frac | stratum overlap, as published | fixed | rep SD, as published | fixed | ratio |
|---|---|---|---|---|---|
| 0.125 | **15.8%** | 100% | 0.01197 | 0.00211 | **5.7x** |
| 0.25 | 29.4% | 100% | 0.00701 | 0.00148 | 4.7x |
| 0.5 | 53.9% | 100% | 0.00373 | 0.00100 | 3.7x |
| 1.0 | 100% | 100% | 0.00039 | 0.00039 | 1.0x |

**The point estimate is not biased.** Paired across seeds: p = 0.85, 0.49, 0.70 at
frac = 0.125, 0.25, 0.5. Both index arrays remain *marginally* random subsamples of an
already-matched set, so their stratum histograms stay equal **in expectation**; what is
destroyed is the per-draw guarantee, and that shows up as variance, not bias.

Part 1 claimed the bias "grows exactly along the plotted axis" and that panel d therefore
measures the wrong thing. That is wrong. The published size curve's *shape* is not an
artefact of this bug. What is wrong is its **error bars**, and the fingerprint is already
in the committed JSON:

| frac | observed SD growth | expected from sqrt(n) | excess (nohead / head) |
|---|---|---|---|
| 0.5 | 2.83x | 1.41x | **2.00x** / 1.66x |
| 0.25 | 3.73x | 2.00x | **1.87x** / 1.70x |
| 0.125 | 4.84x | 2.83x | **1.71x** / 1.55x |
| 1.0 | 1.00x | 1.00x | 1.00x / 1.00x |

Across 39 and 40 seeds, every `frac < 1.0` point carries 1.55-2.00x more dispersion than
sampling alone explains, and `frac = 1.0` — the one value that never enters the buggy
branch — sits at exactly 1.00x. Revised severity: **panel d's curve stands; its error bars
are understated by roughly 1.6-2x and the matching guarantee it advertises is void below
frac = 1.** The `figS_discriminator` conclusion does not move.

## A1 — the mismatch is real; "it manufactures the null" is not supported

Reconstruction driving the real `build_T_random`, `compute_sr_matrix`,
`sr_predicted_shift_vectors`, `nearest_state_for_points`, `angle_between_vectors` and
`rayleigh_from_angles` (no run data needed — the random-walk baseline is analytic).
Ground truth was planted so the remapping **is** SR-explained: each cell's A->B shift is
the SR-predicted vector at its own location plus angular scatter. A correct test must
therefore recover alignment; the question is what each code path reports.

| reward | Rayleigh R, as published | fixed | median angle | residual rho |
|---|---|---|---|---|
| **(0.3, 0.3)** — the paper's, on the diagonal | 0.969 | 0.977 | 15.0° vs 13.6° | +0.088 vs +0.104 |
| (0.75, 0.25) — off-diagonal control | 0.938 | 0.977 | 16.4° vs 13.6° | +0.031 vs +0.037 |

**The reflection nearly cancels for this arena.** Reflection about the diagonal is linear
and fixes any on-diagonal point, so for a reward at (0.3, 0.3),
`mirror(reward - p) = reward - mirror(p)`: the mirrored shift and the prediction looked up
at the mirrored state transform together. The residual error is only the part of the SR
field that is not diagonally symmetric. Moving the reward off the diagonal makes the bug
bite (R 0.977 -> 0.938, monotonic in distance from the diagonal: 0.972, 0.967, 0.955,
0.949, 0.936 at (0.3,0.3), (0.35,0.3), (0.5,0.3), (0.7,0.3), (0.8,0.2)).

At the paper's geometry the cost is **~0.9% of R**. A genuine SR alignment planted in the
data still comes back at R = 0.969, overwhelmingly significant. So the bug **cannot**
convert a real alignment into the null panel c reports, and the claim that it "manufactures
the negative result the panel reports" is not supported. Revised severity: a real frame
error worth fixing — it is latent and would become serious under any off-diagonal reward
condition — but **it does not put fig5 supp panel c's conclusion in question.**

## Caveat that applies to both

No raw run data exists in this checkout (A12): zero `.pkl`/`.npy`/`.h5`/`.parquet` files
and no `data/` directories, so neither fix could be re-measured on real runs. Panel c of
the finding-1 figure is the exception — it is computed directly from the committed
`Repro/aggregate/occupancy_matched_estimator.json`. Everything else is reconstruction.
The reconstruction's matched-set size reaches ~1072 against the real median of 2142,
because two independently generated periods overlap less than two periods of one real run;
per-stratum occupancy, which is what governs this bug, is in the real 1-4 sample regime.

---

# Part 4 — Second finding: `hash(mode)` (candidate patch), A2 (measured only)

## Part 1 #2 — candidate patch P3, measured then reverted

`lib/occupancy_matched_estimator.py:292` seeds from `hash(mode) % 2**31`. `str.__hash__`
is salted per process, so that stream differs on every run and the cached results cannot
be regenerated.

```python
# as it stands
rng = np.random.default_rng([seed, rep, hash(mode) % (2**31)])

# proposed -- a fixed code per mode, defined at module level
_MODE_CODE = {"naive": 0, "pos": 1, "joint": 2, "reweight": 3}
rng = np.random.default_rng([seed, rep, _MODE_CODE[mode]])
```

Verified while applied, across three fresh interpreters: `joint -> 2`, first draw
`221737325` every time (previously 892552764 / 777482213 / 1643756848). Now reverted, so
`aggregate/run_estimator_validation.py` remains non-reproducible as the tree stands.

## A2 — measured, deliberately NOT fixed

**Why not fixed.** `compute_skaggs` has 15 callers, three of them in `model/`,
and the gates 0.05 / 0.15 / 0.3 are spread across figs 1-5 and `figS_legacy`.
Switching the normaliser re-tunes every gate in the paper and changes which
cells enter every figure. That is a scientific decision, not a cleanup.

**Reconstruction** (`lib/ratemaps.build_rate_maps` + `compute_skaggs`, 5 seeds x
512 units): every field is planted with the same 0.06 shift toward the reward,
so the correct answer is the same whichever cells are selected. Only the
normaliser is varied.

| coverage of goal period | median Skaggs, repo | weighted | cells passing 0.15 gate, repo | weighted | measured dd, repo | weighted | discrepancy |
|---|---|---|---|---|---|---|---|
| 0.81 | 2.25 | 1.65 | 512 | 510 | -0.0741 | -0.0726 | **2.0%** |
| 0.71 | 2.74 | 1.61 | 512 | 502 | -0.0943 | -0.0866 | **8.9%** |
| 0.61 | 3.35 | 1.52 | 512 | 487 | -0.1278 | -0.1080 | 18.3% |
| 0.51 | 4.11 | 1.41 | 512 | 463 | -0.1737 | -0.1356 | 28.1% |
| 0.41 | 5.11 | 1.22 | 512 | 402 | -0.2302 | -0.1478 | 55.8% |
| 0.31 | 6.90 | 0.92 | 512 | 342 | -0.2823 | -0.1519 | 85.9% |
| 0.20 | 11.99 | 0.45 | 512 | 291 | -0.3109 | -0.1450 | 114.4% |

Three things, in increasing order of importance:

1. The score inflates **5.3x** as coverage falls, while the standard form
   correctly falls. Confirms the mechanism.
2. **The gate stops working.** Under the repo formula all 512 units clear the
   0.15 gate at every coverage level — in the period where sampling is worst,
   the gate excludes nothing. The occupancy-weighted gate correctly tightens
   (510 -> 291).
3. Consequence for the measured effect: the cells the repo gate fails to
   exclude are exactly those whose fields are poorly sampled in the
   goal-directed period, and their centre-of-mass estimates are dragged toward
   the visited region — which is toward the reward. **The artefact points the
   same way as the hypothesis.**

**Tolerance.** The discrepancy is 2% at 0.81 coverage and 8.9% at 0.71; it
passes 18% at 0.61 and doubles the effect below 0.31. So this is a non-issue
while `cycle_1_goal` covers roughly 70% or more of the arena's bins, and a real
problem below ~60%.

**The one number needed.** Bin coverage of `cycle_1` vs `cycle_1_goal` in the
real runs. No committed JSON records coverage or per-period gate counts
(checked all three files in `Repro/aggregate/`), so it can only come from the
behaviour pickles, i.e. from a rerun. Until then A2's severity is bounded, not
known. Note also that the occupancy-weighted curve itself drifts
(-0.073 -> -0.145): fixing the gate reduces the coverage sensitivity but does
not remove it, because low coverage degrades the centre estimates regardless.
