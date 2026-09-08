# Figure 2 — frozen replay & policy-driven remapping

Same run as Figure 1 (`Repro/nohead_seed42`, no head, lr=`5e-5`, reward `(0.3, 0.3)`), with **Phase C** frozen-replay probes.

```bash
python "code_analysis/figure 2/generate_figure2.py" --no_show
```

## Phases

| Phase | Probe | Meaning |
|-------|--------|---------|
| **A** | `cycle_1` | Prediction / open-loop random-walk probe |
| **B** | `cycle_1_goal` | RL / goal-conditioned policy probe |
| **C** | `replay_c1_postrl` (centres); pred/postrl in panel a | Frozen trajectory |

## Layout

| Row | Panels |
|-----|--------|
| 1 | **a** — frozen-replay remapping grid (**same cells as Figure 1d**) |
| 2 | **b** (COM centres A/B/C, 3 cols) · **c** (Δd violins A→B and A→C) |
| 3 | **d** — left: **A–C field peak** (as Fig. 1e); right: **A–B COM** |

**Conceptual point:** remapping of distance-to-reward is driven more by **exploration statistics (A→B)** than by weight updates on a frozen trajectory (**A→C**).

## Outputs

| File | Description |
|------|-------------|
| `figures/panels/panel_{a,b,c,d}.pdf` | Individual panels |
| `figures/panels/phase_abc_summary.json` | Numeric Δd summary |
| `figures/figure2_raw.pdf` | Vector composite |
| `figures/figure2_raw.png` | Raster preview (`pdftoppm`) |
