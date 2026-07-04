# Reflection

## Approach

One metered call per event (single-pass full coverage). Every event gets a real
measurement, so no verdict is a blind guess. On the 200-event private stream that
is 40×(1.0+1.5+1.0+2.0+2.0) = 300 credits against a 320 budget → zero cost
overage, which keeps the whole score in the TPR/FPR plane where detection quality
actually lives.

Each handler combines the published `mean ± 3σ` baseline bounds (hard bound,
catches large deviations) with a second, tighter "soft" bound placed just above
the clean envelope. I calibrated every soft margin against the actual clean-vs-
faulty separation on the practice **and** public streams jointly, keeping only
margins that recovered real subtle faults with **zero** added false positives on
either stream — a margin that generalizes, not one hand-fit to one run's numbers.

## Which fault types were hardest to catch, and why?

- **Subtle embedding drift / corpus staleness.** The sharpest edge. Clean
  `centroid_shift` runs right up to ~0.039 and the baseline bound is 0.0435, so a
  subtle drift at 0.04 sits *inside* the clean tail — almost no FP-safe room. A
  soft bound at 0.90·baseline for shift and 0.85·baseline for age reclaimed every
  subtle instance across both streams with no false positive, but the margin is
  thin: a private instance nudged even closer to clean is effectively unseparable
  by a single-field threshold without paying FP cost, and I chose not to pay it.
- **Missing-upstream lineage.** The event payload under-declares its own inputs
  (it lists one; the healthy graph resolves two), so trusting the declared
  `inputs` silently misses the fault. The robust signal is the structural norm
  (2 upstream sources), not the self-reported expectation.
- **Feature skew** was the *easiest* of the "subtle" family, because the tool
  returns a **normalized** statistic (`mean_shift_sigma`). Clean sits under 0.4σ
  and even subtle skew is >2σ, so one view-independent threshold at ~1σ is clean.

## What would you change about your cost/coverage tradeoff, with another pass?

I deliberately spend the full single-pass budget rather than skipping calls to
save credits: at a 320 budget the marginal credit is effectively free (no
overage), while a skipped call forces a blind no-alert that costs real TPR. If
the budget were tighter I would triage — skip the cheapest, highest-baseline-
margin checks first and reserve calls for the pillars where clean and faulty
overlap most.

With a genuine second call to spend, I would target the embedding pillar, where a
single static field can't separate subtle faults from clean: re-profile each
value against a rolling in-run baseline of recently-seen clean values held in
`ctx.state`, and tighten the soft bound adaptively instead of trusting the static
published one — trading a little cost for recall exactly where the static
threshold is weakest.
