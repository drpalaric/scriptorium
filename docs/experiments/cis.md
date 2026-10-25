# Pre-registration: comparison against the CIS Controls

Written and committed **before any CIS experiment is run or any CIS-related evaluation code exists**. Departures after this commit are reported as departures.

## Inputs (user-supplied, local only)

- CIS Controls v8.1.2 workbook: 18 Controls and 153 Safeguards with descriptions.
- CIS Controls v8.1 to NIST SP 800-53 Rev 5 mapping workbook: 372 official (safeguard, NIST control) pairs covering 140 safeguards and 203 NIST controls, each with a relationship (Subset, Superset or Equivalent). CIS states it deliberately under-maps.
- Both workbooks state CC BY-NC-ND 4.0. Files and everything derived stay under `~/.scriptorium/`. CIS text is sent to the Anthropic API only with the explicit `--allow-cis` flag. Public documents report **aggregates only**: no safeguard text, no per-item predictions.

## Fixed settings

Model `claude-sonnet-5-5`, effort low, adaptive thinking, `max_tokens` 1024, **one run per condition**, no tuning of any prompt, model setting or classifier setting after seeing results. Input to every classifier is the safeguard's **description only** (no title, number or control). Output rules unchanged: anything but one valid label is a failure and counts as wrong. Bootstrap: 1,000 resamples, seed 20260424 (the project seed).

## E2 (primary): CIS safeguard text to NIST family

- **Items:** the 140 safeguards with at least one official NIST pair. For each, **A(safeguard)** = the set of NIST families of its mapped controls.
- **Systems:** the LLM with prompt `mapping/v1`, unchanged (sha256 `d631630734b7`); the scikit-learn baseline with its existing selected configuration, fitted on the NIST training split only (830 records). Neither has seen CIS text; the baseline's score includes the shift from NIST wording to CIS wording, which is part of what is tested.
- **Metrics:** *lenient accuracy* = prediction is in A(safeguard); *strict accuracy* on the safeguards with |A| = 1 (84 expected). Reference lines: a uniformly random family and the most frequent family in the A sets.
- **H1 (the only confirmatory test):** the LLM's lenient accuracy exceeds sklearn's. Supported only if the **95% paired cluster-bootstrap interval for (LLM minus sklearn) lenient accuracy has a lower bound above 0**, resampling whole CIS Controls (18 clusters, because safeguards in one Control are not independent).
- Strict accuracy and everything else are secondary and descriptive. A null or negative H1 is reported as such: it would mean the LLM's advantage on NIST text does not demonstrably transfer to CIS wording.

## E1 (secondary): classify safeguards into the 18 CIS Controls

- **Items:** all 153 safeguards; label = the parent Control number.
- **LLM:** new prompt `prompts/cis_control/v1.txt` (committed with this document). **sklearn:** TF-IDF (unigrams, sublinear tf) + linear SVM, C=10, balanced class weights, the same settings as the NIST baseline and **not tuned on CIS**, scored by stratified 5-fold cross-validation (seed 20261025), so every prediction is out-of-fold.
- **Reported:** accuracy and macro-F1 with bootstrap intervals (resampling safeguards) and the paired LLM-minus-sklearn difference. No decision rule; descriptive.

## E3 (secondary): crosswalk retrieval, both directions

- TF-IDF cosine similarity (sublinear tf, English stop words, vectorizer fitted on both texts), no training, no LLM, no API. Candidates: all 1,014 NIST statements (our OSCAL text) and the 153 CIS descriptions.
- **Directions:** CIS safeguard to NIST control (140 queries) and NIST control to CIS safeguard (203 queries). Truth = the official pairs. Also the Equivalent-only subset (21 pairs).
- **Reported:** hit@1, @5, @10 (any correct target in the top k), MRR, and the analytic random baseline. Descriptive. Because CIS under-maps, a missing pair is not evidence of a wrong answer, so these are recall-style figures only.

## Known limits that apply whatever the outcome

The CIS-to-NIST mapping is public and the model may have seen it, so E2 and E3 reduce but do not remove the memorisation concern. One model, one run per condition. 140 and 153 items give wide intervals, and 18 clusters give wider ones. The lenient E2 metric can reward a family that happens to appear in an under-mapped set. Family prediction is a coarse stand-in for control mapping.

## Budget

About $0.1 to $0.3 in total. Any single run is capped at $5 by the harness.

## Amendment 1 (committed before any CIS experiment was run)

Found while checking item counts, before any scoring: both CIS workbooks store safeguard IDs as numbers, so `3.10` was stored as `3.1` and collided with the real `3.1`. The first read merged five safeguards (3.10, 4.10, 8.10, 13.10, 16.10) into their `.1` neighbours, and it attributed their mapping pairs to the wrong safeguards. The loader now rebuilds each ID from its position within its Control and verifies it against the stored number, failing loudly if the order is unexpected; the rebuilt IDs agree with the text IDs in the ATT&CK file for all 152 safeguards it mentions (three titles differ only in wording).

Corrected facts, replacing figures quoted above: **E1 has 153 items (as registered); E2 has 143 items, of which 86 are single-family, not 140 and 84**; the 372 official pairs, the 18 clusters and everything else are unchanged. The registered rules, metrics, systems and settings are unchanged. No result had been computed at the time of this amendment.

One preprocessing detail not stated above: cross-references inside descriptions ("Safeguard 4.1", "Control 12") are replaced with `[safeguard]` / `[control]`, as NIST control IDs are, so that a number cannot leak the label. It affects one description.
