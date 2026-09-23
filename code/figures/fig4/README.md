# Figure 4 — Reward Relocation (Paper Figure 4)

**Claim:** After the reward moves from A to B, place fields leave A and concentrate at B — the map flexibly tracks the current goal.

**Protocol:** ONE frozen random-walk trajectory is recorded before any RL. It is replayed through three weight snapshots:
- `baseline` (pre-RL weights)
- `postA` (after 1000 episodes of RL at A = (0.3, 0.3))
- `postB` (after 1000 more episodes of RL at B = (0.7, 0.7))

Because trajectory and occupancy are identical in all three, any field movement is purely representational.

## Data

- Training: `code/model/ratinabox_relocation.py`, seeds 0–9
- Config: `--n_cycles 2 --max_episodes 2000 --reward_pred_head --rnn_rl_lr 5e-5`
- Reward A: (0.3, 0.3), Reward B: (0.7, 0.7)
- Outputs: `results/multi_seed_reloc/reloc_seed{N}/`
- Seed ranking (entropy gate + tracking score): `code/runs/select_best_seed_fig4.py`
- Displayed network: the MEDIAN entropy-gate-passing seed by tracking quality
  (`reloc_seed9`, rank 17 of 32 clean seeds; entropy 7.55 / 7.60 / 7.47 bits
  across the three periods, gate ≥ 5.5). Set in `utils.py` (`DEFAULT_RUN`).

## Panels

- **a** Example tracking cells: 3×N rate maps (baseline / postA / postB).
  Exemplars must pass the Skaggs gate in all three periods, have a minimum
  smoothed-field area (anti-speckle), and genuinely track A then B
  (`analysis.fig_example_cells_grid`).
- **b** Field-centre (COM) clouds across the three phases, same graphic as
  Figure 3 panel b, with A/B stars
- **c** Mean distance-to-A and distance-to-B across phases — crossing curves
- **d** Δd violin: toward A (pre→postA) and toward B (postA→postB) with Wilcoxon p

## Supplementary Figure 4

- **a** Per-seed fraction toward A (postA)
- **b** Per-seed fraction toward B (postB)
- **c** Pooled violin of both shifts across all 10 seeds
- **d** Scatter: tracking quality (frac→A vs frac→B per seed)

## Scripts

```bash
# Run from repo root

# Training (parallel with Fig 3, ~70 min on 2 GPUs):
python code/runs/run_fig34_parallel.py --only fig4 --gpus 0 1

# Pick best seed and regenerate:
python code/runs/select_best_seed_fig4.py

# Or manually:
python code/figures/fig4/generate_figure4.py \
    --run_dir results/multi_seed_reloc/reloc_seed9 --no_show

python code/figures/fig4/generate_figure4_supp.py \
    --multi_seed_dir results/multi_seed_reloc \
    --no_show
```
