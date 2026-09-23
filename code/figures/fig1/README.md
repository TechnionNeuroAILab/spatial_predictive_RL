# Figure 1 — panel generation (self-contained)

Generates Figure 1 panels as **PNG + PDF** under `figures/`.

```bash
python code/figures/fig1/generate_figure1.py --no_show
# defaults: results/nohead_seed42 (no head, lr=5e-5), reward (0.3, 0.3)
```

Default composite layout (**4 columns**, Nature double-column width, equal row heights):

| Row | Layout |
|-----|--------|
| 1 | **a** (architecture cartoon, full width) |
| 2 | **b** (FoV) · **c** (PCA 3D: x / y / heading, 3 cols) |
| 3 | **d** (before/after RL remapping, jet, random cells) · **e** (peak dist) · **f** (COM dist) |

Panel letters are consecutive (`a`–`f`). Panel **d** samples units uniformly at random (`--seed`). Panel **c** uses a shared colormap for allocentric x and y, and a cyclic map (`hsv`) for heading θ.

## Outputs

| File | Description |
|------|-------------|
| `figures/panels/panel_*.pdf` | Individual panels (vector PDF) |
| `figures/figure1_raw.pdf` | Composite assembled from panel PDFs (vector; aspect preserved) |
| `figures/figure1_raw.png` | Raster preview of the composite (via `pdftoppm`) |

The composite PDF is built with `pypdf` by placing each panel PDF into equal-height rows with **contain** scaling (no stretch). Panel **a** uses `v11_architecture_vectorized.pdf`.

## Styling

- Nature-style defaults: Arial/Helvetica, 6–7 pt labels, 183 mm width
- Rate maps use `inferno` colormap (perceptually uniform)
- Panel letters placed on the composite (c, a, b, d, h)

## FAQ (from panel content)

### Panel b — dots on the walls?

The **hollow coloured circles** are **landmark objects** (3 types × 12), placed on **all four walls** only (none in the interior; 3 per wall per type). This is a schematic for the FoV panel — training (`code/model/ratinabox_v11.py`) places objects uniformly in the arena.

- Edge colour = object type (blue / orange / green)
- Filled grey ellipses = egocentric **field-of-view (FoV) cells** tuned to objects
- Right strip = **24-dim observation**: 3×6 object-FoV firing rates (RGB rows) + 6 wall-distance rays (purple row)

### Panel c — 3D PCA manifold

PCA of phase-1 hidden states (3 components) during the random-walk probe. Three views of the same embedding:

- coloured by allocentric **x** (`viridis`)
- coloured by allocentric **y** (same `viridis`)
- coloured by agent heading **θ** (cyclic `hsv`)

### Panels e and f — peak / COM distance histograms

Place cells gated by Skaggs ≥ `--skaggs_threshold` (default 0.05) in both periods. For each cell, field **peak** and activity-weighted **COM** are measured on smoothed rate maps; panels show overlaid histograms of Euclidean distance to the reward. Periods match panel **d**: `cycle_1` (prediction / RW) vs `cycle_1_goal` (goal-driven policy). Frozen replay belongs in Figure 2, not here.

Numeric arrays are also saved to `figures/panels/peak_com_distances.npz`.

### Legacy panels

- **legacy_f**: one neuron, all cycles (prediction row stable, RL row varies)
- **h**: top-N neurons, **cycle 1 only**, prediction vs RL columns
- **legacy_g**: 100-cell goal-conditioned rate-map grid

## Dependencies

`numpy`, `pandas`, `matplotlib`, `scipy`, `scikit-learn`, `Pillow`, `ratinabox` (panel b only), plus training pickles in `sandbox/New_Results_v3/reward_pred_head_offcenter/data/` (default: seed 42, `rnn_rl_lr=5e-5`, reward head ON, reward at (0.3, 0.3)).
