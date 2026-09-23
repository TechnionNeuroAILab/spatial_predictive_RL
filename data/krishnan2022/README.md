# Krishnan et al. 2022 — data confrontation

Krishnan S, Heer C, Cherian C, Sheffield MEJ. *Reward expectation extinction
restructures and degrades CA1 spatial maps through loss of a dopaminergic
reward proximity signal.* Nat Commun 13, 6662 (2022).
DOI: [10.1038/s41467-022-34465-5](https://doi.org/10.1038/s41467-022-34465-5)

Why this dataset: our model's reward-prediction head has target
`exp(-|x - x_rew|^2 / 2 sigma^2)` with `sigma = 0.15` (paper/main_v3.tex) —
the in-silico analogue of the VTA-DA reward-proximity ramp Krishnan et al.
measured, and their reward-extinction (RE) protocol is the experimental
analogue of our extinction simulation (reward + head switched off after
learning).

## Tier-1 vs Tier-2

- **Tier-1 (this directory): confrontation against PUBLISHED SOURCE DATA.**
  Every number here comes from the journal-hosted Source Data file
  `MOESM4_source_data.xlsx` (per-figure summary values released with the
  paper). We did **not** reanalyse any raw data.
- **Tier-2 (not done, requires the authors): reanalysis of raw/processed
  imaging data.** Raw data are available only on request
  (sheffield@uchicago.edu); the authors' analysis code is public at
  [github.com/seethakris/HPCrewardpaper](https://github.com/seethakris/HPCrewardpaper)
  (shallow clone in `code_krishnan/`, HEAD recorded in `MANIFEST.json`).

## Files

| File | Purpose |
| --- | --- |
| `MOESM4_source_data.xlsx` | Published Source Data (18 sheets); sha256 in `MANIFEST.json` |
| `MANIFEST.json` | Provenance: URL, sha256, download date, DOI, availability caveat, code-repo SHA |
| `utils.py` | Defensive loaders for the sheets used (anchor-scanning, magnitude assertions) |
| `analysis.py` | Computes the confrontation numbers, caches to `cache/krishnan_data.json` |
| `code_krishnan/` | Shallow clone of the authors' analysis code (not executed) |

## What was extracted from where

- **Sheet `Figure3`, Fig 3B "Reliability"** → `load_reliability()`:
  per-animal place-field reliability for Rewarded / RE-low / Re-Rewarded.
  **Only n = 3 imaged animals** (the behaviour panels in Fig 1F have n = 12);
  this limits paired non-parametric tests (exact Wilcoxon floor p = 0.25).
- **Sheet `Figure 5`, Fig 5D "Reward Normalized Mean F"** → `load_da_ramp()`:
  per-axon (n = 7) reward-normalized VTA-DA fluorescence at Start / Middle /
  End of the 200 cm track, plus RE-high and RE-low blocks. Quirk in the
  published file: the RE-low block has identical Middle and End values for
  every axon.
- **Sheet `Figure2`, Fig 2D** → `load_field_density_profiles()`:
  per-track-region, per-condition profiles with 95% CIs. After inspection
  these are **spatial-correlation** values for the *same* tracked cells in
  every condition (n fixed at 133/262/207 per region) — **not** field
  density. Kept as the spatial-correlation-by-region confrontation.
- **Sheet `Figure3`, Fig 3C sample sizes** → `load_field_counts_by_region()`:
  per-condition place-field counts by region (these DO vary by condition),
  used for the near-reward field-density enrichment.
- **Sheet `Figure3`, Fig 3E "% placefields"** → `load_pf_fraction_by_region()`:
  per-animal (n = 3) percentage of fields in Middle / End of track per
  condition (Start derived as the remainder).

Track orientation (documented choice): reward at the **end** of the 200 cm
track — the rewarded-condition DA signal ramps Start < Middle < End, matching
the paper's reward-proximity ramp. For the model overlay, region centers map
to normalized distance-to-reward 0.875 / 0.5 / 0.125; `analysis.py` provides
the head target at region centers, bin-averaged over regions, and on a fine
grid.

## Headline numbers (from `cache/krishnan_data.json`)

- Reliability group means: 0.494 (Rewarded) → 0.273 (RE-low) → 0.411
  (Re-Rewarded); group-mean fractional drop **−44.6%**, with **62.7%** of the
  drop recovered on re-reward; direction consistent in 3/3 animals.
  *Annotation:* the paper text suggests ≈ −32%; the per-animal source-data
  values give the larger drop above.
- DA ramp (rewarded): 0.306 / 0.466 / 0.747 (mean at Start/Middle/End,
  n = 7 axons; end vs start paired t p ≈ 0.001, pairing: axon). The
  `sigma = 0.15` head target is much sharper than the measured ramp (data at
  Start is already ~41% of its End value; the model target is ~0 there).
- Near-reward field-density enrichment (fields in 150–200 cm vs uniform 0.25):
  ×1.38 (Rewarded) → ×1.07 (RE-low, indistinguishable from uniform,
  chi-square p = 0.29) → ×1.53 (Re-Rewarded).

## Data-request draft (Tier-2)

> To: sheffield@uchicago.edu
> Subject: Data request — Krishnan et al. 2022 (Nat Commun) extinction sessions
>
> Dear Dr. Sheffield,
>
> I am Stefano Recanatesi (Allen Institute / Technion), and my group has
> developed a spatial reinforcement-learning model whose reward-prediction
> component makes quantitative predictions about how CA1 spatial maps degrade
> when reward expectation is extinguished, which we would like to test against
> the reward-extinction experiments in Krishnan et al. (2022, Nature
> Communications 13:6662). Would you be willing to share the processed
> ΔF/F-by-position matrices (cells × track position, per lap or per session)
> for the extinction sessions, which go beyond the per-figure summary values
> in the published Source Data? We would of course credit the data
> appropriately and are happy to sign any data-sharing agreement you require.
>
> Best regards,
> Stefano Recanatesi
