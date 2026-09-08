# Figure 3 & Supplementary Figure 3 — plasticity enables remapping

No-head runs from `New_Results_v3/dose_sweep_lr_nohead` (seed 42 per LR).

```bash
python "code_analysis/figure 3/generate_figure3.py" --no_show
python "code_analysis/figure 3/generate_figure3_supp.py" --no_show
```

## Data

| Role | Path |
|------|------|
| High-plasticity demo (1e−4) | `dose_sweep_lr_nohead/lr_0.0001` |
| Plasticity sweep | `lr_2e-06`, `lr_1e-05`, `lr_5e-05`, `lr_0.0001`, `lr_0.0003` |

**Note:** A full 20-seed × LR grid is not present in the repo. The sweep uses **seed 42** at each LR. Matched multi-seed folders (`matched_nohead_1e-4_seed*`) keep only frozen-replay pickles, so they are not used for A→B panels.

## Main Figure 3

| Panel | Content |
|-------|---------|
| **a** | Remapping grid at lr=1e−4 (prediction vs goal policy; toward-reward cells) |
| **b** | COM centres for Phases A / B / C at lr=1e−4 |
| **c** | Boxplots of per-cell COM/peak Δd (A→B) vs RL learning rate |

**Conceptual point:** remapping is **allowed by plasticity** — Δd toward reward grows with RL learning rate.

## Supplementary Figure 3

| Panel | Content |
|-------|---------|
| **a** | Mean distance to reward (COM & peak) vs LR |
| **b** | # active place cells vs LR |
| **c** | Mean Skaggs information vs LR |
| **d** | Violins of COM distance-to-reward (A vs B) per LR + means |

Outputs under `figures/` and `figures_supp/`.
