# Security notes

What the local API, MCP server and feedback store defend against, what test covers each claim, and what they do not defend against. Written against OWASP guidance, mapped **by risk name** because I could not confirm the numbering of the 2025 web Top 10 (I confirmed only that A10:2025 is *Mishandling of Exceptional Conditions*). The OWASP MCP Top 10 is at v0.1 and I confirmed only MCP1 (token mismanagement and secret exposure) and MCP10 (context injection and over-sharing). Re-check these lists before relying on this mapping.

Threat model: a single user's machine, serving `localhost` only. Attackers considered: a web page in the user's browser, another local process, a client or document that feeds text to the tools, a submitter trying to poison the model, and someone who obtains a token or a copy of the database file.

## Web application risks

| Risk | Control | Test |
|---|---|---|
| Broken access control | Two roles; admin endpoints return 403 to users; users see and withdraw only their own entries (a foreign id looks like "not found"). No endpoint mints or lists tokens. | `test_permission_matrix` (every endpoint x five identities), `test_users_see_and_withdraw_only_their_own_entries`, `test_health_is_open_and_every_route_is_accounted_for` |
| Cryptographic failures | Tokens are 256-bit random and stored only as SHA-256 hashes. **No TLS** (loopback only); the server refuses to bind elsewhere. Database is not encrypted at rest. | `test_create_authenticate_and_plaintext_is_never_stored`, `test_server_refuses_to_bind_beyond_loopback` |
| Injection | Every SQL statement uses bound parameters. Request bodies are strict schemas (unknown fields rejected, lengths capped). Text is stored and returned as data and never executed or interpolated into prompts. | `test_sql_injection_in_name_is_stored_as_data_or_rejected`, `test_injection_style_text_is_stored_as_data`, `test_validation_errors_never_echo_input_and_unknown_fields_are_rejected` |
| Insecure design | Corrections need human approval, a held-out gate and are reversible. History is append-only. | `test_poisoning_through_the_api_is_stopped_by_review_and_the_gate`, `test_rollback_restores_original_model_exactly` |
| Security misconfiguration | Loopback bind; Host-header allowlist (DNS rebinding); no CORS; docs and OpenAPI disabled; security headers on every response including errors; database files mode 0600/0700. | `test_dns_rebinding_host_header_is_rejected`, `test_no_cors_headers_even_for_a_preflight`, `test_docs_and_openapi_are_disabled`, `test_security_headers_on_success_and_every_kind_of_error`, `test_database_files_are_private` |
| Vulnerable and outdated components | Dependencies pinned in `uv.lock` and installed with `--locked`; few runtime dependencies. CI runs `uv audit` on every push and weekly (the command is marked experimental by uv; the last local run found no known vulnerabilities in 56 locked packages). GitHub Actions are pinned to commit SHAs, the workflow has read-only permissions and uses no secrets. | `.github/workflows/ci.yml` |
| Authentication failures | Bearer tokens only, no cookies. Revocation applies on the next request. Ten failed attempts lock the client out for 60 s. Tokens never appear in logs. | `test_revocation_applies_to_the_next_request`, `test_failed_logins_trigger_a_lockout_that_expires`, `test_failed_auth_is_logged_without_the_token`, `test_malformed_authorization_headers_are_401` |
| Software and data integrity failures | The event log is a SHA-256 hash chain with insert-only triggers; `verify` detects edits, gaps and reordering; anchors detect removal of recent rows. Source data is pinned by commit and hash in the manifest. | `test_verify_detects_out_of_band_edits_to_history`, `test_verify_detects_a_gap_in_the_middle`, `test_anchors_detect_removed_rows_that_the_chain_alone_cannot` |
| Logging and alerting failures | Every state change is an audited event with its actor (`user:alice`, `admin:phil`). Failed authentication is logged with client and path. **There is no alerting.** | `test_full_admin_flow_matches_the_cli_and_every_actor_is_recorded` |
| Server-side request forgery | No endpoint fetches a URL or file named by the caller. Catalog names are allowlisted; no tool takes a path. | `test_mcp_tool_surface_annotations_and_no_path_like_parameters`, `test_catalog_is_allowlisted_and_input_is_not_echoed` |
| Mishandling of exceptional conditions | Errors fail closed: a corrupt or tampered log is refused, not partly used. Unexpected errors return a generic 500 with details only in the server log. A request body over 16 KB gets 413 whether or not it declares a length. | `test_tampered_event_content_is_refused_by_replay`, `test_security_headers_on_success_and_every_kind_of_error`, `test_body_size_limit_with_and_without_content_length` |

## LLM application risks (2025 list)

The only code that calls an LLM is the opt-in evaluation harness (`scriptorium.evals`); the API, MCP server and feedback store call none. Several risks therefore apply only to the harness, and the planned reviewer will be built against the rest.

| Risk | Status |
|---|---|
| LLM01 Prompt injection | The harness quotes requirement text inside `<requirement>` tags with any look-alike tag neutralised, tells the model the text is untrusted data, and gives it no tools. The reply must match a strict schema or it counts as a failure. A prompt-injection robustness task is planned but not built, so resistance is **untested**. **Planned reviewer:** the same treatment, and no write access. |
| LLM02 Sensitive information disclosure | SCF content and SCF-derived models stay local, and the harness never sends SCF text. It sends only NIST statements (public domain) and the family list. The API key is read from the environment and never stored, logged, cached or reported (a test plants a fake key and searches every output file for it). API error text is redacted and length-capped before display. |
| LLM03 Supply chain | Pinned lockfile; no third-party models downloaded. |
| LLM04 Data and model poisoning | Approval step, held-out gate with tolerance, anomaly warnings, per-submitter attribution, rollback. **Known limit below.** |
| LLM05 Improper output handling | API responses are JSON built from validated values; no model output is executed. The UI must escape text it displays (stage 4). |
| LLM06 Excessive agency | MCP clients can only queue pending corrections and withdraw their own; approve, apply, rollback and reset are not exposed to MCP at all. Test asserts no such tool exists. |
| LLM07 System prompt leakage | Not applicable: no system prompt. |
| LLM08 Vector and embedding weaknesses | Not applicable: no vector store. |
| LLM09 Misinformation | Every classification and mapping response carries a disclaimer and a low-confidence flag calibrated on out-of-fold predictions. Corrections require human review. |
| LLM10 Unbounded consumption | Text and body size caps, `top_k` caps, per-token submission limit (20 per minute), pending-queue cap, lockout. For paid calls: no API call without `--yes`, a projected-cost refusal, a hard spend cap that stops new calls, unpriced models refused, `max_tokens` bounded, and a cache so reruns are free. |

## OWASP MCP Top 10 (v0.1), the two items I verified

- **Token mismanagement and secret exposure:** the MCP server holds no secrets and calls no external services. API tokens are hashed at rest and never logged.
- **Context injection and over-sharing:** tools return only what the caller asks for. The feedback tools show a caller only its own submissions. If SCF is built locally, query results can include SCF control text, which goes to whichever client you connect, so connect only clients you trust.

## Known limits (do not rely on these)

1. **Trigger-word poisoning passes the gate.** An approved batch that ties an unusual phrase to a wrong label does not touch the held-out records, so the score does not drop. Human review is the control; warnings point the reviewer at it. A test documents this behavior.
2. **Review is only as good as the reviewer.** The model trusts the admin.
3. **No TLS and no encryption at rest.** Use full-disk encryption. Do not expose this beyond localhost without adding TLS and a real login.
4. **The hash chain is tamper-evident, not tamper-proof.** Without anchors, someone with write access to the file can rebuild it, and removing the newest rows is invisible. Run `python -m scriptorium.db anchor` regularly and keep copies of `anchors.jsonl` somewhere the database owner cannot quietly change.
5. **The lockout is per client address.** On localhost every client looks the same, so repeated failures from one process also lock out the UI for 60 s. Tokens have 256 bits of entropy, so brute force is not the realistic threat; the lockout mostly limits noise.
6. **Tokens are not scoped to catalogs** and do not expire. Revoke tokens you no longer need.
7. **Any process running as your user can read the database and tokens file.** The hashes protect copies of the file, not a local attacker with your account.
8. **The evaluation cache is not integrity-protected.** Someone who can write `~/.scriptorium/evals/cache.db` could change cached answers, and `--replay` would then reproduce the altered results. Treat a replayed number as only as trustworthy as that file.
9. **The spend cap can be overshot** by the calls already in flight when it is reached (at most `--workers`), and cost figures come from recorded token counts at listed prices, not from the invoice.
10. **Prompt-injection resistance of the models is unmeasured** until the robustness task exists and has been run.
