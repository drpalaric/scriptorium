# Prompt v2/v3/v4 experiment: development results (dev-B only)

Protocol: [`prompt-v2.md`](prompt-v2.md), committed at `68fbfcb` before any of these runs. Code that produced the inputs: `6d1aa4a`. Model `claude-sonnet-5-5`, effort low, adaptive thinking, one run per condition, 2026-10-25. Run reports and paired comparisons: [`results/llm/dev/`](../../results/llm/dev/). **The frozen test split was not used by any of these runs.**

## Outcome under the registered rules

- **H1 is not supported.** v3 (8 retrieved labelled examples) versus v1 on dev-B: macro-F1 difference **0.000, 95% interval -0.034 to +0.033**; accuracy difference +0.007 (-0.022 to +0.036). The rule required the macro-F1 interval's lower bound to be above 0.
- Per the registered rule, **no v2-family prompt is run on the test split, and v1 remains the default.** v1's existing test result (accuracy 0.875, macro-F1 0.893) stands as the only test result.
- **v2's gain is attributed to inheritance, not understanding.** On enhancements, the parent-only diagnostic (the model never sees the enhancement's own text) scored 0.954; v2 scored 0.965, a gap of 0.011, inside the registered 0.05 threshold.
- v4 (exploratory) showed no detectable difference from v1, and was not adoptable regardless.

## Results on dev-B (407 items; 122 base controls)

| Condition | Status | Accuracy (95% CI) | Macro-F1 (95% CI) | Cost |
|---|---|---|---|---|
| v1 statement only | reference | 0.865 (0.811 to 0.915) | 0.888 (0.820 to 0.919) | $0.17 |
| v3 + 8 retrieved examples | **primary** | 0.872 (0.834 to 0.913) | 0.888 (0.826 to 0.913) | $0.79 |
| v2 + parent statement | secondary | 0.943 (0.903 to 0.973) | 0.934 (0.872 to 0.966) | $0.26 |
| v4 + examples with guidance | exploratory | 0.875 (0.838 to 0.915) | 0.898 (0.837 to 0.923) | $1.64 |

Paired differences against v1 (b minus a; intervals resample base controls):

| Comparison | Accuracy diff | Macro-F1 diff | Only b right / only a right |
|---|---|---|---|
| v3 vs v1 (**H1**) | +0.007 (-0.022, +0.036) | 0.000 (-0.034, +0.033) | 19 / 16 |
| v2 vs v1 | +0.079 (+0.035, +0.125) | +0.046 (+0.012, +0.096) | 35 / 3 |
| v4 vs v1 | +0.010 (-0.020, +0.043) | +0.010 (-0.028, +0.055) | 24 / 20 |

### Enhancements only (285 items): the inheritance diagnostic

| Input the model saw | Accuracy | Macro-F1 |
|---|---|---|
| v1: the enhancement's own statement | 0.856 | 0.862 |
| v2: enhancement plus parent | 0.965 | 0.956 |
| v2p: the **parent only** | 0.954 | 0.942 |
| v3: enhancement plus 8 retrieved examples | 0.856 | 0.863 |

v2p versus v2: accuracy -0.011 (-0.033, +0.006). An enhancement always shares its parent's family (0 exceptions in train and test), so classifying the parent and inheriting its label is nearly as good as having both texts, and much better than reading the enhancement alone.

## What this means

- **Showing the model how NIST filed similar requirements did not help**, at least with 8 examples drawn from a pool of 423 records and this prompt. The interval is about +/-0.03, so a small effect cannot be ruled out, but the 23 errors seen on the test split are not the kind that nearest-neighbour examples fix. Dev-B used a smaller pool than the final test would (dev-A only, not the full training split), so this likely understates the benefit slightly.
- **Guidance text in the examples did not help either**, and cost about 2x more per run.
- **The parent finding is practically useful but is not a model improvement.** For a catalog with a hierarchy (NIST enhancements, SCF sub-controls such as `GOV-03.1`), classifying the parent and inheriting its family beat reading the sub-control alone by about 10 points. For free-text requirements with no parent it does not apply.

## Departures and disclosures

- **Cost:** the pre-registration expected about $1 to $2 for dev runs. Actual: **$2.94** for dev-B plus about $0.12 for the dev-A pilot (12 items per condition, used only to check that every condition runs and returns valid answers; no prompt was chosen or changed from it). All runs were far below the $5 per-run cap.
- **Origin of the hypotheses:** the parent-context idea came from reading the 23 test-split errors of v1. That is disclosed here because the test split informed which hypotheses were tried, even though it played no part in any dev run and no prompt wording borrows from those errors (a test checks that no wording about the specific error cases appears in the prompt files).
- **No pipeline bug was found after dev-B results were seen**, so nothing was rerun.
- **v2p cache sharing:** enhancements of the same parent send identical input, so those requests share one cached answer. The bootstrap resamples whole base controls, so this correlation is accounted for.
- **dev-A was used only for the pilot.** No error reading or tuning took place on it.

## Limits that still apply

The model may have seen NIST SP 800-53 text. One model, one run per condition. Family prediction is a coarse stand-in for control mapping. A single dev split of 122 base controls gives wide intervals for small effects.
