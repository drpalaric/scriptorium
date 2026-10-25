"""`python -m scriptorium.evals cis <e1|e2|e3>`: the pre-registered CIS experiments (docs/experiments/cis.md).

E3 needs no API. E1 and E2 send CIS safeguard descriptions to the Anthropic API, so a live run needs BOTH
`--allow-cis` (you accept that CIS text leaves this machine) and `--yes` (you accept the spend). `--replay` reads
only the local cache. The summary file holds aggregates only; the items file (ids and predictions) stays local.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scriptorium import baseline, ingest, paths
from scriptorium.crosswalks import MappingStore
from scriptorium.evals import cis, mapping
from scriptorium.evals.budget import Budget, estimate_tokens_offline, price_for
from scriptorium.evals.llm import AnthropicClient, CachedClient, EFFORTS, LazyClient, LLMError, LLMRequest, ResponseCache, THINKING_MODES, cache_key
from scriptorium.evals.prompts import load_prompt

PREREG = "docs/experiments/cis.md (commit 8651103)"
EST_OUTPUT_TOKENS = 300


def add_parser(sub) -> None:
    p = sub.add_parser("cis", help="the pre-registered CIS comparison experiments")
    p.add_argument("experiment", choices=["e1", "e2", "e3"])
    p.add_argument("--plan", action="store_true", help="show the plan and projected cost; make no call")
    p.add_argument("--model", default="claude-sonnet-5-5")
    p.add_argument("--effort", choices=EFFORTS, default="low")
    p.add_argument("--thinking", choices=THINKING_MODES, default="adaptive")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--budget-usd", type=float, default=5.0)
    p.add_argument("--yes", action="store_true", help="allow this run to spend up to the budget")
    p.add_argument("--allow-cis", action="store_true", help="allow CIS safeguard text to be sent to the Anthropic API")
    p.add_argument("--replay", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--workspace-id", default=None)
    p.add_argument("--cache", type=Path, default=None)
    p.add_argument("--store", type=Path, default=None, help="crosswalk store directory (default ~/.scriptorium/crosswalks)")
    p.add_argument("--out", type=Path, default=None, help="default ~/.scriptorium/evals/cis")


def _write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")


def _usage(outcomes: list[mapping.Outcome]) -> dict[str, Any]:
    return {"input_tokens": sum(o.input_tokens for o in outcomes), "output_tokens": sum(o.output_tokens for o in outcomes),
            "cache_read_tokens": sum(o.cache_read_tokens for o in outcomes), "cache_write_tokens": sum(o.cache_write_tokens for o in outcomes),
            "recorded_cost_usd_all_answers": round(sum(o.cost for o in outcomes), 6)}


def run(a, default_cache: Path | None = None) -> int:
    from scriptorium.crosswalks import store_dir

    out_dir = a.out or paths.scriptorium_home() / "evals" / "cis"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    try:
        store = MappingStore.load(a.store or store_dir())
        if a.experiment == "e3":
            res = cis.run_e3(store)
            summary = {"experiment": "e3", "preregistration": PREREG, "created_utc": stamp, "method": "TF-IDF cosine similarity, no LLM, no API", "results": res}
            path = out_dir / f"cis-e3-{stamp}-summary.json"
            _write(path, summary)
            print(json.dumps(summary, indent=1))
            print(f"wrote {path}")
            return 0

        if a.experiment == "e2":
            d = cis.build_e2(store)
            prompt = load_prompt("mapping", "v1")
            nist = cis._nist_rows()
            families = mapping.families_from(nist)
            system, codes = mapping.system_prompt(prompt, families), sorted(families)
            task = "cis_e2"
        else:
            d = cis.build_e1(store)
            prompt = load_prompt("cis_control", "v1")
            families = d["families"]
            system, codes = cis.e1_system_prompt(prompt, families), list(families)
            task = "cis_e1"
        items = d["items"]
        schema = mapping.json_schema(codes)
        reqs = [LLMRequest(a.model, system, m, schema, a.max_tokens, a.effort, a.thinking) for m in cis.user_messages(items)]
        cache = ResponseCache(a.cache or (paths.scriptorium_home() / "evals" / "cache.db"))
        p = price_for(a.model)
        cached = sum(1 for q in reqs if cache.get(cache_key(q, task, prompt.version, prompt.sha256)) is not None)
        todo = len(reqs) - cached
        per_in = int(sum(estimate_tokens_offline(q.system + q.user) for q in reqs) / len(reqs)) + 120
        projected = todo * (per_in * p.input + EST_OUTPUT_TOKENS * p.output) / 1_000_000
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    print(f"cis {a.experiment} | model {a.model} | prompt {prompt.task}/{prompt.version} | items {len(items)}, cached {cached}, to call {todo}")
    print(f"rough projection (offline estimate, not a quote): ${projected:.4f} | cap ${a.budget_usd:.2f}")
    if a.plan:
        print("No API call was made.")
        return 0
    if a.replay:
        client, budget = CachedClient(None, cache, task, prompt.version, prompt.sha256, replay=True), None
    else:
        if todo and not a.allow_cis:
            print("Not running: this would send CIS safeguard text to the Anthropic API. Re-run with --allow-cis (and --yes) if you accept that.", file=sys.stderr)
            return 3
        if todo and not a.yes:
            print("Not running: this would spend money. Re-run with --yes to allow spending up to the cap.", file=sys.stderr)
            return 3
        if projected > a.budget_usd:
            print(f"Not running: projected ${projected:.2f} exceeds the cap ${a.budget_usd:.2f}.", file=sys.stderr)
            return 3
        budget = Budget(a.budget_usd)
        client = CachedClient(LazyClient(lambda: AnthropicClient(workspace_id=a.workspace_id)), cache, task, prompt.version, prompt.sha256, budget=budget)
    try:
        outcomes = mapping.run_items(client, items, reqs, codes, a.workers)
    except LLMError as e:
        print(f"error: {e}", file=sys.stderr)
        return 4

    base = {"experiment": a.experiment, "preregistration": PREREG, "created_utc": stamp, "model": a.model,
            "prompt": {"task": prompt.task, "version": prompt.version, "sha256": prompt.sha256},
            "request_parameters": {"effort": a.effort, "thinking": a.thinking, "max_tokens": a.max_tokens},
            "mode": "replay (cache only)" if a.replay else "live", "allow_cis": bool(a.allow_cis),
            "items": {"total": len(items), "scored": sum(o.status != "not_run" for o in outcomes)},
            "usage": {**_usage(outcomes), "spent_this_run_usd": round(budget.spent, 6) if budget else 0.0, "live_calls": client.calls, "answers_from_cache": client.hits},
            "input_preprocessing": {"cross_references_replaced": d["references_stripped"], "sent_to_model": "description text only, quoted as data"}}
    if a.experiment == "e2":
        sk = mapping.sklearn_predictions(items)
        clusters = {i["id"]: i["base_id"] for i in items}
        res = cis.score_e2(outcomes, sk, d["acceptable"], clusters)
        res["reference"] = {"random_family_lenient": d["random_lenient"], "most_frequent_family_lenient": d["most_frequent_lenient"], "single_family_items": d["n_single_family"]}
    else:
        sk = cis.sklearn_cv_predictions(items)
        labels = list(families)
        res = {"llm": mapping.score(outcomes, labels), "comparison_with_sklearn_cv": mapping.compare_with_sklearn(outcomes, sk, labels),
               "sklearn_cv": {"folds": 5, "seed": cis.CV_SEED, "config": cis.CIS_CV_CONFIG}}
        res["llm"].pop("per_family", None)  # per-Control detail stays out of the publishable summary
    summary = {**base, "results": res}
    spath = out_dir / f"cis-{a.experiment}-{a.model}-{stamp}-summary.json"
    ipath = out_dir / f"cis-{a.experiment}-{a.model}-{stamp}-items.json"
    _write(spath, summary)
    _write(ipath, {"note": "LOCAL ONLY: per-item predictions with CIS ids; do not publish", "items": [{"id": o.id, "true": o.true, "pred": o.pred, "status": o.status} for o in outcomes]})
    print(json.dumps(summary, indent=1))
    print(f"wrote {spath}\nwrote {ipath} (local only)")
    return 0 if base["items"]["scored"] == len(items) else 1
