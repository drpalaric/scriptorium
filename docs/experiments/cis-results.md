# CIS comparison: results

Protocol: [`cis.md`](cis.md), committed at `8651103` before any run, with Amendment 1 (corrected item counts) at `8dc6bd8`, also before any run. Model `claude-sonnet-5-5`, effort low, adaptive thinking, one run per condition, 2026-10-25. Aggregate summaries: [`results/llm/cis/`](../../results/llm/cis/). **Aggregates only:** this repository contains no CIS text, no safeguard IDs and no per-item predictions. The CIS Controls are © the Center for Internet Security and licensed [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/); see [cisecurity.org/controls](https://www.cisecurity.org/controls/). The frozen NIST test split was not used.

## Outcome under the registered rule

**H1 is supported.** On CIS safeguard text (never seen by either classifier), the LLM's lenient accuracy for predicting a NIST family exceeded the scikit-learn baseline's: **0.860 against 0.510, a difference of +0.350 (95% paired cluster-bootstrap interval +0.245 to +0.457)**, resampling the 18 CIS Controls. The rule needed the lower bound above 0.

## E2 (primary): CIS safeguard text to NIST family (143 safeguards, 18 clusters)

| | LLM | scikit-learn | LLM minus sklearn (95% CI) |
|---|---|---|---|
| Lenient accuracy (family is among the families of the safeguard's official NIST pairs) | 0.860 | 0.510 | **+0.350 (+0.245, +0.457)** |
| Strict accuracy (the 86 safeguards whose pairs all fall in one family) | 0.826 | 0.442 | +0.384 (+0.247, +0.536) |

Both right on 66, only the LLM on 57, only sklearn on 7, neither on 13. Reference points for the lenient metric: guessing a family at random averages 0.078; always predicting the most frequent family scores 0.224. All 143 LLM answers were valid.

For scale, on the NIST test split the same two systems differed by +0.353 accuracy. The gap on CIS-worded text is the same size, so there is no sign that either collapses under the change of wording, though the metrics differ (lenient here, exact there) and this is not a like-for-like comparison.

## E1 (secondary): classify safeguards into the 18 CIS Controls (153 safeguards)

| | Accuracy (95% CI) | Macro-F1 (95% CI) |
|---|---|---|
| LLM, zero-shot | 0.974 (0.948 to 0.993) | 0.975 (0.943 to 0.995) |
| scikit-learn, 5-fold cross-validation | 0.706 | 0.687 |
| LLM minus sklearn | +0.268 (+0.203, +0.340) | +0.288 (+0.227, +0.379) |

Both right on 107, only the LLM on 42, only sklearn on 1, neither on 3. This task is easy for the LLM partly because it is shown the 18 Control titles, which are descriptive labels, while sklearn learns from about 120 short texts per fold. The interval resamples safeguards as registered, but safeguards within one Control are not independent, so it is narrower than an interval clustered by Control would be.

## E3 (secondary): TF-IDF retrieval against the official pairs, no LLM

| Direction (queries, candidates) | hit@1 | hit@5 | hit@10 | MRR | random hit@1 / @5 / @10 |
|---|---|---|---|---|---|
| CIS safeguard to NIST control (143, 1,014) | 0.119 | 0.350 | 0.483 | 0.236 | 0.003 / 0.013 / 0.025 |
| NIST control to CIS safeguard (203, 153) | 0.232 | 0.517 | 0.626 | 0.365 | 0.012 / 0.058 / 0.112 |
| Equivalent pairs only, CIS to NIST (18 queries) | 0.111 | 0.278 | 0.500 | 0.202 | |
| Equivalent pairs only, NIST to CIS (20 queries) | 0.150 | 0.750 | 0.850 | 0.411 | |

Plain lexical similarity is far better than chance here but a weak crosswalk. For comparison, the same method against the SCF-to-NIST mapping (local results, not published) found the right control first about two times in three, because SCF borrows NIST's wording; CIS writes short action statements against NIST's long requirements. CIS also states that it deliberately under-maps, so a "miss" can be a correct pair that CIS did not record, and these are recall-style figures only. Whether embeddings or an LLM reranker would do better is untested.

## What this means

- **The LLM's advantage over a small supervised classifier replicates on differently worded text and a different taxonomy.** It does not look specific to NIST's phrasing.
- **It does not remove the memorisation concern.** The CIS-to-NIST mapping is public, and the model knows the NIST families, so some of the 0.860 may be recall of published pairs.
- **The lenient metric has a built-in generosity** (a prediction is right if it appears in any of the safeguard's families) and a built-in unfairness (CIS under-maps, so a sensible but unrecorded family counts as wrong). The strict figure on single-family safeguards is the cleaner number, and it points the same way.

## Departures and disclosures

- **Item counts changed from the first draft of the protocol.** A first read of the workbooks merged five safeguards (`.10` IDs stored as numbers collided with `.1`). This was caught by checking item counts before any scoring, fixed, and recorded as Amendment 1 before any run. E2 uses 143 items (86 single-family), not 140 (84).
- **One preprocessing step was added in the same amendment:** cross-references inside descriptions are replaced so a number cannot leak the label. It affected one description.
- **Cost:** about $0.14 in total for E1 and E2 (E3 makes no API call), against the registered $0.1 to $0.3.
- **CIS text was sent to the Anthropic API** behind the explicit `--allow-cis` flag, as approved. The run summaries record this.
- No pipeline bug was found after results were seen, and nothing was rerun.

## Limits that apply regardless

One model, one run per condition. 143 and 153 items give wide intervals, and 18 clusters give wider ones. Family prediction is a coarse stand-in for control mapping. The sklearn baseline is deliberately simple, and on E2 it is applied to text from a different source than it was trained on.
