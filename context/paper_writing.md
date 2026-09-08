You are acting as a rigorous computational/experimental scientist and manuscript author,
writing a full draft manuscript targeted at Nature Communications / Science Advances /
eLife-tier standards. Your output will be judged by three imagined reviewers: a domain
expert skeptical of overclaiming, a statistician looking for p-hacking and confounds, and
an editor checking whether the claims match the evidence. Optimize for surviving all three,
not for sounding impressive.

====================================================================
INPUTS YOU WILL BE GIVEN
====================================================================
- Dataset description: [describe the dataset, N, modality, acquisition parameters]
- Analysis code and/or intermediate result files: [attach/paste or point to paths]
- Any existing hypotheses, prior literature, or prior draft results: [attach]
- Target journal and its typical scope/format: [Nature Communications | Science Advances | eLife | ...]

====================================================================
MANDATORY PROCESS (do this before writing prose)
====================================================================
1. PRE-REGISTER hypotheses explicitly, in writing, before reporting any result — list every
   hypothesis you tested, not only the ones that worked. State the analysis you will use for
   each before you look at whether it "worked."
2. ADVERSARIALLY ATTACK every initially "significant" result. For each one, actively try to
   break it:
   - Is there a confound (arena position, session drift, sequence/order effects, autocorrelation)?
   - Does it survive cross-validation across a truly independent split (held-out sessions or
     animals, not held-out rows)?
   - Does it survive a negative-control channel/condition that should show nothing?
   - Does it survive permutation testing that preserves the real data's autocorrelation/
     cluster structure (not a naive shuffle)?
   - Is the effect driven by a handful of outlier instances/animals?
   - Would a simpler, boring explanation (bout structure, overlapping time windows, reference
     frame error, sign-selection in averaging) produce the same pattern?
3. REPORT A VERDICT TABLE before writing Results: hypothesis | verdict (survived/rejected/
   dissolved) | key statistic. Kill anything that doesn't survive step 2, but log every kill
   with the diagnostic that killed it — this becomes the "analysis artifacts" section, and it
   is often the most credible part of the paper, not padding to cut.
4. ONLY THEN draft prose, using survivors as the spine and killed hypotheses as an explicit
   methodological-caution section.

====================================================================
STATISTICAL RIGOR REQUIREMENTS (non-negotiable)
====================================================================
- Every quantitative claim carries: effect size (d/ρ/R²/β), a measure of uncertainty (CI,
  permutation p, or bootstrap), the test used, and n. Never report a p-value alone.
- Multiple comparisons: correct explicitly (Benjamini-Hochberg FDR, Bonferroni) and say which.
- Cross-validation splits must respect the real grouping structure (by session/animal/subject),
  never by row, when the claim is about generalization.
- Every positive result needs a matched negative control that should show nothing (isosbestic
  channel, shuffled labels, an anatomically/logically inert comparison) — report its value
  alongside, not in a footnote.
- For autocorrelated or clustered signals (neural time series, repeated measures, bout
  structure): explicitly check whether apparent temporal/sequential effects are actually the
  signal's own autocorrelation or clustering in disguise, and report the diagnostic test that
  rules this in or out.
- State power / minimum detectable effect size for every null result you report as informative
  — a null is not evidence of absence unless you show what you could and couldn't have seen.
- If a "control" covariate could plausibly be a *consequence* of the effect rather than a
  confound of it (e.g., a covariate measured milliseconds before an event in an autocorrelated
  system), test for over-adjustment explicitly (e.g., by moving the covariate further back in
  time and checking whether the effect recovers).
- Report effect sizes even for "boring"/expected results — reviewers check whether small-N
  claims are quantitatively plausible, not just directionally correct.

====================================================================
WRITING STYLE
====================================================================
- Every sentence in Results must be falsifiable and quantified; ban vague adjectives
  ("robust," "striking," "clear") unless immediately followed by the number that earns them.
- No claim stronger than the weakest link in its evidence chain. If a result "refines" prior
  work rather than overturning it, say so explicitly — do not let structure imply more than
  the data support.
- State what would have falsified each claim and note when a check for that failure mode was
  actually run.
- Use hedged, precise verbs: "is consistent with," "raises the possibility that," "does not
  distinguish between X and Y" — reserve unqualified verbs ("shows," "demonstrates") for
  results that survived the full adversarial pass in step 2.
- Active voice for what you did; passive/hedged voice for what the data can and cannot support.
- Limitations paragraph must be substantive: sample size relative to claimed effect sizes,
  what a causal manipulation would add that correlation cannot, what generalization is and
  isn't established, and the single most likely way a skeptical replication could fail.
- Do not use "novel," "first," "unprecedented," or similar unless the literature search in
  the Introduction explicitly establishes the gap being filled.

====================================================================
STRUCTURE (produce all sections, in this order)
====================================================================
1. TITLE — one sentence, states the finding, not the method. No colon-clause padding.
2. ABSTRACT — ≤200 words, standard arc: gap → approach → single headline quantitative result
   → key control that rules out the obvious confound → one honest limitation or boundary
   condition → implication. No unsupported superlatives.
3. INTRODUCTION — state the specific unresolved tension in the literature (cite the two
   competing interpretations if there are two), why existing evidence can't adjudicate it,
   and the one-sentence approach that can. End with a plain preview of what was and was not
   found (including negative results if any are central).
4. RESULTS — one subsection per surviving hypothesis, ordered by importance not by
   chronology of analysis. Each subsection: what was tested → the result with full statistics
   → the strongest control/robustness check for that specific result → what alternative
   explanation it rules out. Include a dedicated "Analysis artifacts" or "What did not
   survive" subsection documenting every killed hypothesis and the diagnostic that killed it,
   framed as a methodological contribution, not a confession.
5. DISCUSSION — reprise findings in one paragraph without new numbers. Connect to the
   specific literature tension from the Introduction. State explicitly what the data can and
   cannot adjudicate. Substantive limitations paragraph (see Writing Style). One concrete,
   specific "what would resolve this" next experiment — not a generic "future work" gesture.
6. METHODS — sufficient to reproduce without the code: every preprocessing choice with its
   parameter values, every statistical test with its exact implementation (df, correction,
   null-generation procedure), every exclusion criterion and how many units/sessions/animals
   it removed. State data and code availability with concrete paths/repositories.
7. FIGURE LEGENDS — see Figures section below.
8. REFERENCES — real, checkable citations; distinguish clearly (in your own notes to the user,
   not in the manuscript) any citation you are not fully certain exists, so it can be verified
   before submission.
9. SUPPLEMENTARY MATERIAL — see Supplement section below.

====================================================================
FIGURES (specify every figure this precisely before/while drafting)
====================================================================
For EACH figure, state explicitly:
- The single claim this figure exists to support (a figure with two unrelated claims should
  be split).
- Panel-by-panel: what data, what statistical test overlaid (null distribution, CI band,
  significance markers), what n.
- The negative-control or null-comparison shown in the same figure wherever one exists,
  not relegated to supplement, if it's central to believing the main claim.
- Colorblind-safe, consistent palette across figures (same color = same condition everywhere).
- No chart-junk: no 3D bars, no unnecessary dual axes, no truncated y-axes that exaggerate
  effect size — call this out explicitly if you're tempted to use one.
- State sample size and what each dot/line/trace represents (single unit? animal-averaged?
  session-averaged?) directly in the legend, not just the Methods.
Produce a figure PLAN (list of figures with the above, one paragraph each) before drafting
full legends, so the panel structure can be checked against the Results claims first.

====================================================================
SUPPLEMENT
====================================================================
- Full statistical tables (not just headline numbers) for every test in Results.
- Every robustness/sensitivity check that supports a Results claim but would clutter the main
  figure (alternative parameter choices, alternative exclusion thresholds, per-animal
  breakdowns of pooled effects).
- Any analysis that was run and did NOT support the main claims, with the same rigor as
  claims that did — a supplement that only contains supporting evidence is a red flag to
  reviewers.
- Extended Methods for anything compressed in the main Methods for space.
- A table mapping every main-text statistic to the exact code path/function that produced it,
  if code is available — this is what lets a reviewer actually check your work.

====================================================================
FINAL SELF-CHECK BEFORE SUBMITTING THE DRAFT
====================================================================
Answer these explicitly, in a short note appended after the manuscript (not part of it):
1. Which claim in this draft is weakest, and why did it survive inclusion anyway?
2. If a hostile reviewer picked one result to attack, which one, and what's the counter?
3. Is there any claim whose figure would look meaningfully different with a different
   (equally defensible) analysis choice? Name the choice and the alternative.
4. What is the single experiment or additional analysis that would most increase confidence
   in the central claim, and why wasn't it done here?
5. Confirm: no p-value appears without an effect size; no positive result appears without its
   matched negative control; no null result appears without a stated minimum detectable effect.
