# Pre-registration: duplicate detection, scikit-learn against an LLM

Written and committed **before any code or run for it**. Scope is deliberately narrow: **do two statements say the same thing?** No coverage verdicts and no "does this policy satisfy this control" judging.

## Systems (fixed)

- **word:** TF-IDF on words (sublinear tf, English stop words), cosine similarity.
- **char:** TF-IDF on character 3-5-grams (`char_wb`, sublinear tf), cosine. **The registered scikit-learn arm.**
- **lsa:** the word matrix reduced to 150 dimensions by truncated SVD, cosine.
- All three are fitted on the text of all NIST statements plus the CIS descriptions (unsupervised, no labels or pairs).
- **LLM:** `claude-sonnet-5-5`, effort low, adaptive thinking, prompt `overlap/judge-v1` (committed with this file), one run, answers `same`, `overlap` or `different`; scored 2 / 1 / 0, an invalid answer as 0.

## Labelled pair sets (seed 20261025)

1. **real-equivalent (primary):** positives are the official CIS-to-NIST pairs that CIS marks *Equivalent* ("the exact same security concept"). Negatives, three per positive: the same safeguard against one NIST control from the same family that has no official pair with it, and two NIST controls from other families.
2. **real-overlap (secondary):** 100 positives sampled from the Subset/Superset pairs (partial overlap), negatives built the same way.
3. **paraphrase (secondary):** 200 NIST statements, each rewritten by the LLM (prompt `overlap/paraphrase-v1`, committed here) into a positive pair; one hard negative each (the rewrite against a different control of the same family). **Caveat:** the LLM both wrote the positives and judges them, which may favour the LLM arm.

## Metrics and rules

- **AUC** (separating positives from negatives) per arm and set; 95% intervals from a percentile bootstrap, 1,000 resamples, **resampling groups** (CIS Control for the real sets, base control for paraphrase).
- **Precision, recall, F1** at a decision rule: for scikit-learn, a similarity threshold chosen on one half of the groups and applied to the other half (two folds, both orders, averaged); for the LLM, `same` counts as positive on real-equivalent and paraphrase, `same` or `overlap` on real-overlap.
- **Cost and time:** seconds for scikit-learn (fit and score), dollars and wall time for the LLM.
- **H1 (the only confirmatory test):** on **real-equivalent**, the LLM's AUC exceeds the **char** arm's AUC, paired difference with a 95% interval whose lower bound is above 0. A null or negative result is reported as such. Everything else is descriptive.

## Handling

CIS text is local-only and goes to the API only with `--allow-cis`; the paraphrase set uses public NIST text only. Public docs get aggregates only. Departures are reported as departures.

## Limits

21 positives in the primary set give a wide interval. CIS under-maps, so a negative may in truth be related. The paraphrase set is partly circular. One model, one run.
