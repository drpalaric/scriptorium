# Scriptorium

Policy-as-code and control mapping for AI governance. Public data only; no employer material.

**Status: Stage 2 (scikit-learn baseline).** The only measured results so far are the scikit-learn baseline below. No LLM has been called, and no LLM results exist yet. Results are added only after real runs.

## Planned components

| Layer | Tooling | Status |
|---|---|---|
| Data: NIST SP 800-53 rev5 requirement text labelled by control family | OSCAL catalog, `scriptorium.ingest` | Stage 1 |
| Baseline classifier: requirement text to control family | scikit-learn (TF-IDF + linear model) | Stage 2, results below |
| Record layer: policy rules carrying control IDs | Rego on OPA, `opa test`, GitHub Actions | Not started |
| Document layer: lint for LLM-drafted policy text | Vale | Not started |
| Prompt-engineering eval harness (control mapping, policy drafting, prompt-injection robustness) | Anthropic API, Pydantic, offline replay | Not started |

### Design notes

- **Why Rego/OPA** for the record layer, rather than Cedar or Kyverno: _(to be written when the layer is built)_.
- **Vale checks style and structure, not factual correctness.** It can flag missing sections, inconsistent shall/must usage and vague wording. It cannot tell whether a drafted policy is true or adequate.

## Data

Source: NIST SP 800-53 Rev 5 OSCAL catalog from [usnistgov/oscal-content](https://github.com/usnistgov/oscal-content), pinned to a commit and file hash in `data/manifest.json`. The raw file is not committed; fetch it with:

```bash
uv run python -m scriptorium.ingest fetch
uv run python -m scriptorium.ingest build
```

Each active control or enhancement is one record: statement text with control IDs replaced by `[control]`, labelled with its family. Withdrawn controls are dropped. The train/test split is stratified by family and grouped by base control (`AC-2` and `AC-2(1)` stay together), frozen with a fixed seed.

## Baseline results (scikit-learn)

TF-IDF (word unigrams, sublinear tf) + linear SVM (C=10, balanced class weights). The configuration was chosen by grouped 5-fold cross-validation (by base control) on the training split only, from 32 candidates. The held-out test split was evaluated once. Full numbers: [`results/baseline_metrics.json`](results/baseline_metrics.json). Reproduce with `uv run python -m scriptorium.baseline`.

| | Value | 95% CI (bootstrap over base controls) |
|---|---|---|
| Test accuracy (184 records, 20 families) | 0.522 | 0.420 to 0.630 |
| Test macro-F1 | 0.503 | 0.363 to 0.560 |
| Majority-class accuracy | 0.060 | |
| Cross-validation macro-F1 (train, selected config) | 0.523 | |

How to read this:
- The model is far above chance (20 families) but wrong about half the time. Family is a coarse label, and controls in different families often share vocabulary; I have not analyzed which families are confused with which beyond spot checks.
- Several families score 0 on the test set (CA, PL, PS, RA). Per-family scores rest on very few test records: AT has 1, MA and PS have 2, MP and SR have 3, PL and PT have 4. Families with fewer than 5 test records are flagged `low_support` in the JSON and should not be read as reliable.
- Cross-validation and test scores agree closely, which suggests the model was not tuned to the test split.
- Family prediction is a stand-in for control mapping. It does not show the model can pick the exact control.

## Optional: Secure Controls Framework (SCF)

Scriptorium can optionally read a copy of the [Secure Controls Framework](https://securecontrolsframework.com) (SCF) that **you download yourself**, to classify requirements into SCF domains and to map between SCF and NIST. This is an optional input, not part of the core pipeline.

```bash
uv run python -m scriptorium.scf build --catalog /path/to/oscal-catalog.json   # writes to ~/.scriptorium/scf/
uv run python -m scriptorium.baseline --dataset scf                            # domain classifier, results stay local
uv run python -m scriptorium.crosswalk                                         # SCF <-> NIST retrieval, scored against the file's own NIST mappings
```

- `scriptorium.scf` reads the OSCAL form of the SCF catalog (statement text as input, domain as label, orphaned controls dropped). Everything it writes goes under `~/.scriptorium/` (or `$SCRIPTORIUM_HOME`), never into this repository.
- `scriptorium.crosswalk` ranks controls in one catalog by TF-IDF similarity to a control in the other, with no training, and scores the ranking against the NIST 800-53 Rev 5.2 mappings contained in your SCF file. A test shuffles those mappings and confirms the score falls to chance. **Results from SCF input are not published here.** Run it yourself to get numbers for your copy. Agreement with the SCF's own crosswalk is not the same as a mapping being correct.
- A test fails if any SCF-derived data is tracked in git.

- **Nothing from SCF is in this repository.** No SCF control text, mappings, derived datasets, or text generated from SCF content is committed, cached for replay, or published on the project site. Outputs from SCF input are written to `data/scf/`, which is gitignored.
- **The repository contains code only.** The loader takes a path to a file you obtained from SCF. Tests use synthetic fixtures with made-up IDs.
- **SCF's terms apply to you.** The SCF is published by the Secure Controls Framework Council under a license based on CC BY-ND 4.0, with additional terms on its [terms page](https://securecontrolsframework.com/terms-conditions). Read them before using SCF content, especially the terms on derivative works and on using AI with SCF content. This project does not grant any rights to SCF content, and its MIT license covers the code only.
- **Attribution.** Secure Controls Framework® (SCF) is a registered mark and copyright of the Secure Controls Framework Council, LLC. This project is not affiliated with or endorsed by the SCF Council.

The data this project publishes and reports on comes from NIST SP 800-53 Rev 5 (OSCAL), a U.S. government work.

## Development

```bash
uv sync
uv run pytest
```

## License

MIT. NIST SP 800-53 content is a U.S. government work (public domain in the U.S.).
