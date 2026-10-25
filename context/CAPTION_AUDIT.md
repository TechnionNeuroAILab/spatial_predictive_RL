# Caption and Methods audit

Every caption clause in `paper_repo/main.tex` checked against the code that
draws the panel it describes. Sample sizes and simulation counts are out of
scope: they move when the cohort changes, and the figures state them directly.

Findings are ordered by how wrong they were. Everything under **Fixed** is
already applied; everything under **Flagged** needs an author decision.

---

## Fixed

### 1. Figure 4 (proximity objective) described five panels; there are four

The caption opened with

> a) The auxiliary prediction error propagates through the proximity head into
> the recurrent parameters.

— a schematic that has never been in this figure. Every later letter was
therefore off by one against the artwork, and the last two were also
transposed: `fig_delta_violins` is panel **c** and `fig_distance_histograms` is
panel **d**, but the caption had "d) Distribution of field-centre distances"
ahead of "e) Per-cell changes".

Five in-text references cited `\Cref{fig:proximity}a` through `e`. The `a`
reference now points at `\Cref{fig:arch_after}` (Fig. S13), which does show the
reward-prediction head and its gradient path.

### 2. Figure S3 (`fig:s_sr`) was captioned as a figure it is not

The caption read *"Trajectory-derived successor-representation field
concentration. Successor fields estimated from exploratory versus goal-directed
trajectories become more concentrated near reward."*

The artwork (`figS_sr/generate_figure_sr_supp.py`) plots none of that. Its four
panels are metric and null controls: the retired projected-shift metric failing
its own null, the metric actually used in the paper, the cell-identity shuffle
being analytically inert for a difference of means but valid for the paired
statistic, and a reward-location specificity null. The caption now describes
those.

The successor-representation *direction* test is in `fig:s_dissociation`
panel c. The `0.40 -> 0.23` concentration quoted in Supplementary Note S2 is not
plotted in any figure — see **Flagged 1**.

### 3. Figure 2 panel c row labels contradicted the figure's own caption

The two rows were labelled "Before RL" / "After RL", which asserts that the
weights differ. They do not: both are cycle 1 (`cycle_1` and `cycle_1_goal`),
and what differs is the trajectory the probe follows. The caption said so
correctly ("exploratory prediction probe ... post-learning goal-directed
trajectory"), as did the legends of panels d and e ("Prediction" / "Goal
policy"). The rows are now labelled "Prediction" / "Goal policy".

### 4. Methods stated one Skaggs gate; the code uses two

Methods gave 0.3 bits/spike (0.15 for the three-way relocation). That is what
the aggregate analyses use (`fig5`, `figS_sr`, `figS_discriminator`,
`figS_rewardvector`), and the relocation figure does use 0.15 — but every
single-network exemplar panel and its per-network supplement runs at **0.05**
(`fig1`, `fig2`, `fig3`, `figS_pipeline`, `figS_frozenreplay`). Methods now
states both tiers and points at the two threshold-sensitivity panels.

### 5. Ten figures promised panel letters their artwork did not have

`figS_rigid`, `figS_dissociation`, `figS_hypotheses`, `figS_extinction`,
`figS_sosa` and `figS_discriminator` hard-coded `\textbf{a)}` in the caption
while the plate carried no letters at all; `figS_sr`, `figS_pipeline`,
`figS_frozenreplay`, `figS_relocation` and `figS_rewardvector` had no per-panel
caption text at all. All eleven now carry drawn letters and `\subref` captions
with per-panel sentences, so caption and artwork can no longer drift.

### 6. Figure 6 caption said "orange or black"; the data series are black

`generate_figure5_data.py` declared its own palette (`#4c72b0`, `#e76f51`,
`#2a9d8f`, `#8B4513`) instead of importing the Okabe-Ito roles, which is why
reward A was teal there and blue in the relocation figure. On the shared
palette, experimental quantities are black throughout; the caption now says so.

### 7. Two panels were drawn over their own data

- Figure 7 panel c: the "map never reached pre-switch level" note sat at
  `transAxes (0.03, 0.96)`, on top of the triangle markers it describes.
- Figure 2 panel b: the variance-explained string was drawn at `(0.005, 0.995)`
  and ran through the "Arena x" subplot title in every shipped version.

---

## Flagged — needs an author decision

### 1. Supplementary Note S2 cites a figure for a number that is not plotted

> An SR estimated separately from the exploratory and goal-directed
> trajectories concentrates toward reward under the goal-directed policy (mean
> successor-field distance $0.40\to0.23$; \Cref{fig:s_sr}).

`fig:s_sr` does not show successor-field distances (see **Fixed 2**). Either the
number loses its figure citation and is reported as a text-only statistic, or a
panel showing it is added. I left the citation in place rather than silently
deleting a cross-reference.

### 2. The schematic labels its phases "Phase A/B"; the text says "Phase 1/2"

`v11_architecture_vectorized.pdf` is externally authored artwork, left
untouched by request, so the new Figure 1 caption bridges the two namings
explicitly ("Phase 1, marked *Phase A* in the schematic"). A one-line edit to
the SVG would remove the need for that.

### 3. Figure 1 keeps Arial and Tahoma

Same cause: 28% of Figure 1's glyphs are Arial/Tahoma rather than TeX Gyre
Heros. Arial is Helvetica-metric so the difference is small, but it is the one
documented exception in `check_figure_type.py`'s font allowlist. No SVG
toolchain (Inkscape, rsvg, cairosvg) is installed here.

### 4. Orphan cross-references

`fig:arch_before` (S12) is still never cited in the running text; S13 now is,
from the Figure 1 and Figure 4 captions. Line 282's `\Cref{fig:s_matched}`
refers to a label defined nowhere — currently harmless, as it sits inside a
commented block.

### 5. Panel p-values now print `P < 0.0001` rather than `P < 10^{-k}`

mathtext sets a superscript at 0.7x the surrounding size, so `P < 10^{-6}` at
5.5 pt printed its exponent at 3.85 pt, and TeX Gyre Heros has no Unicode
superscript glyphs to fall back on (U+2070, U+2074-2079 and U+207B are absent).
Exact exponents remain in the captions and Results text, where LaTeX sets them
at 7 pt. Reverting would mean raising those annotations to 8 pt.

---

## Checked and correct

- Figure 2a: "24 egocentric sensory values — six field-of-view cells tuned to
  three landmark object types, plus wall cells" matches the 4x6 observation
  strip in `panel_b.py` (3 object types + walls, 6 FoV cells).
- Figure 3 a-d, Figure 5 a-d, Figure 6 a-c, Figure 7 a-d: every panel letter
  matches the artwork after the split and re-lettering.
- Methods "32 x 32 grid, occupancy-normalised, Gaussian sigma = 1.5 bins"
  matches `lib/ratemaps.py:smooth_and_norm(rm, sigma=1.5)` and `M = 32`.
- Methods field-centre and field-peak definitions match `field_peak` (argmax of
  the smoothed map) and the centre-of-mass computation used throughout.
- Figure 5 (relocation): reward A = (0.3, 0.3), B = (0.7, 0.7), as stated.

---

## Redundancy candidates — not removed, listed for your call

You gave latitude to drop panels that duplicate something shown elsewhere. Two
overlaps exist. Neither is an exact duplicate, so I left both in place:

| Panel | Overlaps with | Difference |
|---|---|---|
| `figS_dissociation` a — entropy-threshold sensitivity | `figS_relocation` e — sorted entropy strip | The sweep shows the *statistic* as a function of the gate; the strip shows *which seeds* the chosen gate excludes. Complementary, not duplicate. |
| `figS_dissociation` b — Skaggs-threshold sensitivity | `figS_pipeline` f — Skaggs-threshold sweep | `figS_pipeline` sweeps place-cell count and behavioural shift on the no-head cohort; `figS_dissociation` sweeps the frozen-replay contrast across both cohorts. Same knob, different quantity. |

Dropping either would save roughly one row of a supplementary figure. Say the
word and I will remove it and fold the sentence into the surviving caption.
