# Figure 3 — Reward-Prediction Head (Paper Figure 3)

**Claim:** An explicit reward-prediction objective causes genuine, occupancy-independent remapping toward the reward.

**Control:** Same frozen-replay paradigm as Figure 2. The ONLY difference is that the network is additionally trained with a reward-prediction head (`--reward_pred_head --reward_pred_weight 1.5`) whose gradients flow into the RNN weights.

**Result:** Under frozen-replay (Phase C_post), fields move significantly toward the reward — the opposite of the Figure 2 null result.

## Data

- Training: `code/model/ratinabox_v11.py` (same script as Fig 2), seeds 0–9
- Head flags: `--reward_pred_head --reward_pred_weight 1.5`
- rnn_rl_lr: 5e-5 (same as Fig 2 for consistency)
- Outputs: `results/multi_seed_head/head_seed{N}/`
- Best seed selected by: `code/runs/select_best_seed_fig3.py`

## Panels

- **a** Frozen-replay remapping grid: 8 example cells, pre vs post-RL weights on same trajectory
- **b** COM scatter for Phases A (prediction), B (RL), C (frozen post-RL)
- **c** Violin of Δd for A→B and A→C (both should now be significant)
- **d** Distance-to-reward histograms for A–B and A–C

## Supplementary Figure 3

- **a** Per-seed mean Δd A→B (all negative, unlike Fig 2's mixed result)
- **b** Per-seed fraction toward reward A→B (all above chance)
- **c** Pooled violin (A→B significant across all seeds)
- **d** Head vs no-head paired comparison: Δd(A→B) with head vs without — the double dissociation

## Scripts

```bash
# Run from repo root

# Training (parallel, ~35 min on 2 GPUs):
python code/runs/run_fig34_parallel.py --only fig3 --gpus 0 1

# Pick best seed and regenerate:
python code/runs/select_best_seed_fig3.py

# Or manually:
python code/figures/fig3/generate_figure3.py \
    --run_dir results/multi_seed_head/head_seed0 --reward_pos 0.3 0.3 --no_show

python code/figures/fig3/generate_figure3_supp.py \
    --multi_seed_dir results/multi_seed_head \
    --nohead_dir results/multi_seed \
    --no_show
```
