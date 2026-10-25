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
| Prompt-engineering eval harness (control mapping, policy drafting, prompt-injection robustness) | Anthropic API, Pydantic, offline replay | Control-mapping task built and run once (see results below). Other tasks not started |

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

## LLM evaluation: control mapping

`scriptorium.evals` asks a Claude model to put each held-out NIST requirement into a control family and scores it on the **same 184 items** as the scikit-learn baseline, with a paired comparison. Every number comes from a real run; the full report is in [`results/llm/`](results/llm/).

```bash
uv sync --extra llm
export ANTHROPIC_API_KEY=...      # in your own shell; it is never written to the repo, logs, cache or reports
export ANTHROPIC_WORKSPACE_ID=...   # only if your key is not scoped to a workspace (the API then returns a 400 asking for it)
uv run python -m scriptorium.evals plan mapping --model claude-sonnet-5-5               # no API call; shows a rough projection
uv run python -m scriptorium.evals run  mapping --model claude-sonnet-5-5 --limit 20 --yes   # a small trial first
uv run python -m scriptorium.evals run  mapping --model claude-sonnet-5-5 --yes          # full run, capped at $5 by default
uv run python -m scriptorium.evals run  mapping --model claude-sonnet-5-5 --replay       # reproduce from the cache, no API calls
```

- **Prompts are versioned files** in `prompts/mapping/`. A report records the prompt version and its SHA-256, the model, parameters, date, token counts and cost.
- **Cache:** answers are stored in `~/.scriptorium/evals/cache.db`, keyed by model, prompt version and hash, parameters and input. Re-running is free, and `--replay` never calls the API (a miss is reported, never filled in).
- **Output must match the schema** (one of the 20 family codes, nothing else). Invalid JSON, a wrong value, an extra field, a refusal or a truncated reply counts as a wrong answer and is never repaired.
- **Spend controls:** `run` refuses to call the API without `--yes`, refuses if the rough projection exceeds `--budget-usd` (default 5), and stops making new calls when recorded spend reaches the cap. At most `--workers` calls already in flight can finish after that, so the cap can be overshot by a few calls. Projections are offline estimates, not quotes, and prices are copied from a dated reference (`budget.py`) that can go stale. Unpriced models are refused.
- **What is sent:** NIST requirement statements (public domain) and the family list. SCF text is never sent.
- **Model settings:** thinking is left adaptive at `low` effort by default (`--thinking off` sends the Sonnet 5.5 thinking-off setting). Sampling parameters, forced tool choice and assistant prefill are not sent, because these models reject them. Model output is not deterministic, so a fresh run can differ from a cached one.
- **Tested** with fake clients and a loopback server standing in for the API (request shape on the wire, errors, cache, budget, replay, scoring), and run once for real as described below.

### First real run (one run, one model, one prompt)

`claude-sonnet-5-5`, prompt `mapping/v1` (sha256 `d631630734b7`), adaptive thinking at low effort, run 2026-10-25. All 184 answers were valid. Recorded cost for all answers: about $0.085. Replaying from the cache reproduced it exactly.

| | LLM | sklearn baseline | LLM minus sklearn (paired, 95% CI) |
|---|---|---|---|
| Accuracy | 0.875 (0.807 to 0.937) | 0.522 | +0.353 (+0.237 to +0.479) |
| Macro-F1 | 0.893 (0.812 to 0.938) | 0.503 | +0.391 (+0.292 to +0.484) |

On the same items, both were right on 89, only the LLM on 72, only sklearn on 7, and neither on 16. Intervals resample whole base controls. Weakest families by F1 included PL (4 test items), SI and SC.

**Read this before quoting it:**
- **The model may have seen this text.** NIST SP 800-53 is public and widely reproduced, so a large model may partly recall which family each control belongs to. This run cannot separate recall from understanding. The sklearn baseline learned from 830 records only. The comparison is "zero-shot LLM against a small supervised model", not a test of generalisation to unseen frameworks.
- **The prompt names all 20 families with their titles**, which is a strong hint on its own.
- **One run, one prompt, one model.** Model output varies between runs and prompt wording matters. No prompt tuning was done, and no other model was tried.
- **Family prediction is a coarse stand-in for control mapping** (see above).
- **The cost projection was too high** (about $0.63 projected for this run, $0.069 actual) because the system prompt was served from the prompt cache.

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

## MCP server

`scriptorium.mcp_server` exposes the models to an MCP client over stdio. It serves NIST, plus SCF if you built it locally.

```bash
uv sync --extra mcp
claude mcp add scriptorium -- uv run --directory /path/to/scriptorium python -m scriptorium.mcp_server
```

| Tool | What it does |
|---|---|
| `list_catalogs` | Available catalogs, labels, model version, confidence calibration |
| `classify_requirement(text, catalog, top_k)` | Top families/domains with scores and a `low_confidence` flag |
| `map_requirement(text, to_catalog, top_k)` | Most similar controls in another catalog (TF-IDF cosine) |
| `map_control(control_id, from_catalog, to_catalog, top_k)` | Same, from an existing control, plus the published mapping if your SCF file has one |
| `submit_feedback(text, label, catalog)` | Queue a correction as **pending**. Returns a receipt with an ID. Does not change the model. |
| `list_feedback(catalog, status, limit)` | This server's submissions and their status, newest first |
| `withdraw_feedback(feedback_id, catalog, reason)` | Withdraw one of this server's own pending corrections |

Security properties, each covered by a test:
- **No client can change a model.** Read tools never write. The feedback tools can only queue pending corrections and withdraw the server's own pending ones. Approving, applying, rolling back and resetting are CLI-only admin operations (see below). Submissions are rate-limited (20 per minute).
- **No file paths or URLs as input.** Catalogs come from a fixed allowlist, control IDs and text are validated and length-capped, and error messages don't echo input.
- **Text is data.** Input is never executed or interpolated into prompts, and no LLM is called. Submitted text is stored verbatim for a human to review.
- **Errors are useful but safe.** Validation errors tell the client what was wrong without echoing its input. Unexpected failures reach the client only as a generic error; details stay in the server log.
- **Calibration, not confidence theatre.** `low_confidence` uses a threshold derived from out-of-fold predictions on the training split only.
- If SCF is built locally, tool results can contain SCF control text and are sent to whichever client you connect. Connect only clients you trust.
- Every response says its results are suggestions that need human review.

## Corrections (feedback) with approval and rollback

Users can submit corrections ("that's not AC, it's IA"). A correction changes the model only after an admin approves it and applies it, and only if the frozen held-out score does not drop. Everything is local, under the local database (`~/.scriptorium/scriptorium.db`).

```bash
uv run python -m scriptorium.feedback --catalog nist submit --text "..." --label IA   # returns a receipt with an ID
uv run python -m scriptorium.feedback --catalog nist list --status pending
uv run python -m scriptorium.feedback --catalog nist approve fb_123456789abc --reason "..."
uv run python -m scriptorium.feedback --catalog nist apply                              # gate + report
uv run python -m scriptorium.feedback --catalog nist rollback fb_123456789abc --reason "..."
uv run python -m scriptorium.feedback --catalog nist reset                              # roll back all applied
```

How it works:
- **Append-only log.** Status (`pending`, `approved`, `applied`, `withdrawn`, `rejected`, `rolled_back`) is derived by replaying events, so every change has a history. The database is private to your user account (`0600`). A corrupt event is refused rather than partly used, and the hash chain detects edits made outside this code.
- **Pending and approved entries never change a model.** `apply` refits on the training split plus applied corrections, each with a training weight (default 5, a starting guess). It compares held-out macro-F1 before and after and applies nothing if the drop exceeds the tolerance (default 0.02). `--force` overrides a failed gate only with a reason, and the override is recorded.
- **The test split never leaks into training.** An approved correction whose text matches a held-out record is rejected.
- **Rollback is exact.** Rolling back restores the previous model version, and every response reports `model_version`.
- **Exact matches are surfaced.** If a query matches an applied correction's text, the response includes `applied_correction` next to the model's own prediction.

**What the gate does not catch.** It blocks broad poisoning that hurts held-out accuracy (tested). A targeted attack, such as many corrections that tie a trigger word to a wrong label, leaves held-out records alone and passes the gate. A test documents this. The control for that is human review, which is why approval is required. `apply` also reports warnings for corrections that contradict a confident prediction and for bulk relabelling of one label as another.

## Local API, tokens and the audit database

Feedback, tokens and the audit trail live in one SQLite file, `~/.scriptorium/scriptorium.db` (mode 0600, not encrypted: use full-disk encryption). Every change is a row in an insert-only, SHA-256 hash-chained `events` table.

```bash
uv run python -m scriptorium.db verify          # recompute the chain; checks anchors too
uv run python -m scriptorium.db anchor          # record the head hash in ~/.scriptorium/anchors.jsonl
uv run python -m scriptorium.db backup --to ~/scriptorium-backup.db
uv run python -m scriptorium.db import-legacy   # copy older JSONL feedback logs into the database
uv run python -m scriptorium.auth create --role admin --name phil    # prints the token once
uv run python -m scriptorium.auth create --role user --name alice
uv run python -m scriptorium.auth list
uv run python -m scriptorium.auth revoke tok_1a2b3c4d
uv sync --extra web && uv run python -m scriptorium.api              # http://127.0.0.1:8765
```

- The chain makes edits, gaps and reordering detectable. It is **tamper-evident, not tamper-proof**; keep copies of the anchors file somewhere you can't quietly change (a git commit, an email).
- Tokens are created and revoked only here. The API has no endpoint that mints them. Only SHA-256 hashes are stored.
- The API serves loopback only and refuses other hosts. `user` tokens can classify, map, submit corrections and manage their own pending ones. `admin` tokens can also approve, reject, preview and apply, roll back, reset and view metrics and integrity.
- Use it with `curl -H "Authorization: Bearer <token>" http://localhost:8765/api/me`. Endpoints are listed in `src/scriptorium/api.py`.
- Threat model, per-risk controls with the test that covers each, and known limits: [docs/security.md](docs/security.md).

## Development

```bash
uv sync
uv run pytest
```

## License

MIT. NIST SP 800-53 content is a U.S. government work (public domain in the U.S.).
