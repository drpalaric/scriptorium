# Duplicate detection: scikit-learn against an LLM, results

Protocol: [`overlap.md`](overlap.md), committed at `ebdc7c6` before any code. Aggregates only; the real pair sets use CIS text, so pair-level data stays local. Summary: [`results/llm/overlap/`](../../results/llm/overlap/). Model `claude-sonnet-5-5`, one run, 2026-10-25.

## Outcome

**H1 is not supported.** On real, human-labelled "same concept" pairs, the LLM did not beat the character n-gram scikit-learn arm: AUC **0.946 against 0.930**, difference +0.016 (95% interval -0.074 to +0.130). At ranking pairs, the two are indistinguishable here.

## Results (AUC with 95% interval; F1 at the registered decision rule)

| Pair set | word | **char** | lsa | LLM |
|---|---|---|---|---|
| real-equivalent (21 positives, 84 pairs) | 0.933 | **0.930** (0.836-0.976) | 0.909 | 0.946 (0.873-1.000) |
| real-overlap (100 positives, 400 pairs) | 0.740 | **0.767** (0.712-0.820) | 0.756 | 0.787 (0.724-0.847) |
| paraphrase (200 positives, 400 pairs) | 0.786 | **0.846** (0.809-0.882) | 0.823 | 1.000 |
| F1, real-equivalent | 0.61 | **0.77** | 0.68 | 0.25 |
| F1, real-overlap | 0.55 | **0.52** | 0.55 | 0.72 |
| F1, paraphrase | 0.71 | **0.76** | 0.73 | 1.00 |

LLM minus char AUC: real-equivalent +0.016 (-0.074, +0.130); real-overlap +0.019 (-0.032, +0.059); paraphrase **+0.154 (+0.118, +0.191)**.

## What it shows

- **For ranking, character n-grams matched the LLM on real pairs** and beat plain words on all three sets. Scoring a whole set took 0.04 to 0.1 seconds; the LLM arm took 378 seconds and **$1.47** for 1,084 calls.
- **The LLM is a better reader than a ranker.** Under the registered "same" rule it was very strict on the equivalent set (precision 1.00, recall 0.14: it usually called CIS's "equivalent" pairs `overlap`). On partial-overlap pairs it was much more precise than scikit-learn (0.92 against 0.49).
- **The paraphrase result favours the LLM and is partly circular:** it also wrote the rewrites. Treat the perfect score as an upper bound, not as a measure of real-world performance.
- **Practical split:** scikit-learn to screen thousands of statements for candidates, the LLM (or a person) to read what it surfaces.
- **Thresholds depend on the job.** The best screening threshold for the character arm was about **0.13 to 0.14** for cross-framework and paraphrase pairs, where wording differs a lot (precision 0.74, 0.49 and 0.88 across the three sets). Near-verbatim duplicates inside one document score much higher (0.6 to 0.75 in a small trial), so use a higher threshold there.

## Departures and limits

- The pre-run cost estimate said $0.94 and the actual was $1.47 (the estimator undercounted judge outputs). The cap was not at risk.
- 21 positives give a wide interval on the primary set. CIS under-maps, so some "negatives" may be related. One model, one run.
- CIS text was sent to the Anthropic API behind `--allow-cis`, as approved.
