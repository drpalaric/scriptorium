# Scriptorium

Policy-as-code and control mapping for AI governance. Public data only; no employer material.

**Status: Stage 1 (scaffold and data ingestion).** No models have been trained and no LLM has been called. This README reports no accuracy numbers because none exist yet. Results will be added only after real runs.

## Planned components

| Layer | Tooling | Status |
|---|---|---|
| Data: NIST SP 800-53 rev5 requirement text labelled by control family | OSCAL catalog, `scriptorium.ingest` | Stage 1 |
| Baseline classifier: requirement text to control family | scikit-learn (TF-IDF + linear model) | Not started |
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

## Optional: Secure Controls Framework (SCF)

Scriptorium can optionally read a copy of the [Secure Controls Framework](https://securecontrolsframework.com) (SCF) that **you download yourself**, for example to explore crosswalks to other frameworks. This is an example input, not part of the core pipeline, and it is not implemented yet.

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
