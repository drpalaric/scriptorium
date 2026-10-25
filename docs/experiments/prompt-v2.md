# Pre-registration: prompt variants for NIST family classification

Written and committed **before any run of prompts v2, v3, v4 or v2p**. Anything done after this commit that departs from it is reported as a departure.

## Question

Can giving the model more context about how NIST files requirements improve control-family classification, and if so, is the gain real understanding or inherited structure? The reference point is prompt `mapping/v1` (statement text only), already run once on the test split (accuracy 0.875, macro-F1 0.893, `results/llm/`).

## Fixed settings (all conditions)

- Model `claude-sonnet-5-5`, effort `low`, thinking adaptive, `max_tokens` 1024. One run per condition. Answers are cached and replayable.
- Output schema and validity rules unchanged: anything but one valid family code is a failure and counts as wrong.
- No tuning of prompt wording, example count or any other setting after seeing dev-B results.

## Data and splits

- Development uses the **training split only**. The frozen test split is not read by any experiment run until the final step below.
- The training split's base controls are divided **50/50 within each family**, grouped by base control (an enhancement stays with its parent), seed `20261025`, into **dev-A** (pilot: pipeline checks and reading outputs; never used to choose between prompts) and **dev-B** (confirmatory: all decisions).
- Retrieval pool for examples: when evaluating dev-B the pool is **dev-A only**; when running the final test the pool is the full training split. Code refuses to run if any evaluated item's base control appears in its pool. Dev therefore likely understates the final benefit of examples (a smaller pool).

## Conditions

| Id | Prompt | Input per item | Status |
|---|---|---|---|
| v1 | `v1` | requirement statement only | reference |
| **v3** | `v3` | statement plus the **8** most similar pool records (statement text and their family), most similar last. Similarity: TF-IDF cosine (sublinear tf, English stop words), vectorizer fitted on the pool text only | **primary** |
| v2 | `v2` | statement plus, for enhancements, the parent control's statement | secondary (structural condition) |
| v2p | `v2p` (v1 wording) | for enhancements only: the **parent's statement in place of the item's own**; the model classifies it. Evaluated on enhancements only | diagnostic |
| v4 | `v4` | as v3, each example also showing the first 600 characters of its guidance text | exploratory |

Excluded from every input: titles, control IDs (replaced by `[control]` as in the data), related-control links, the `label`/`sort-id` properties. Guidance text has control IDs stripped the same way.

## Hypotheses and decision rules

- **H1 (the only confirmatory test):** v3 improves macro-F1 over v1 on dev-B. H1 is supported only if the **95% paired bootstrap interval for the macro-F1 difference (v3 minus v1), resampling base controls, has a lower bound above 0, and accuracy does not fall** (point estimate difference at least 0).
- If H1 is supported, v3 is run **once** on the test split and reported next to v1's existing result. If not, no v2-family prompt is run on the test split and v1 remains the default.
- **v2 and v2p** are reported with the same paired intervals but are **not adoptable**, because an enhancement always shares its parent's family (0 exceptions in train and test), so parent context supplies the label's source.
  - Interpretation rule: if v2p's accuracy on dev-B enhancements is within 0.05 of v2's, the v2 gain is attributed to inheritance and not to better understanding of the enhancement text.
- **v4** is exploratory: reported, never adopted, even if it scores highest.
- Multiplicity: only H1 is confirmatory, so no correction is applied. Secondary and exploratory results are labelled as such and not used to claim anything.

## Reporting commitments

- Dev results are reported whatever they show, including a null or negative result.
- If a pipeline bug is found after dev-B results are seen, it is fixed and the affected conditions rerun; both versions are reported.
- Per-run reports (model, prompt version and hash, date, tokens, cost) are kept. Replay from cache must reproduce every number.
- Known limits that apply regardless of outcome: the model may have seen NIST 800-53 text; one model, one run per condition; family prediction is a coarse stand-in for control mapping.

## Budget

Expected about $1 to $2 for all dev runs, about $0.07 for the single test run. Any single run is capped at $5 by the harness.
