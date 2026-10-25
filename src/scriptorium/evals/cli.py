"""Command line for the evaluation harness.

    python -m scriptorium.evals plan mapping --model claude-sonnet-5-5 --prompt v1
    python -m scriptorium.evals run  mapping --model claude-sonnet-5-5 --prompt v1 --yes
    python -m scriptorium.evals run  mapping --model claude-sonnet-5-5 --prompt v1 --replay

`plan` makes no API call. `run` spends money only with `--yes`, never above `--budget-usd`, and
answers from the local cache whenever it can. `--replay` makes no API call at all.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scriptorium import baseline, ingest, paths
from scriptorium.evals import mapping
from scriptorium.evals.budget import Budget, estimate_tokens_offline, price_for
from scriptorium.evals.llm import AnthropicClient, CachedClient, EFFORTS, LazyClient, LLMError, ResponseCache, THINKING_MODES, cache_key
from scriptorium.evals.prompts import load_prompt
from scriptorium.evals.report import build_report, render_markdown, write_report

DEFAULT_MODEL = "claude-sonnet-5-5"
DEFAULT_BUDGET = 5.0
EST_OUTPUT_TOKENS = 300  # per call, a rough allowance for the answer plus any thinking at low effort
SCHEMA_OVERHEAD_TOKENS = 120


def default_cache_path() -> Path:
    return paths.scriptorium_home() / "evals" / "cache.db"


def default_reports_dir() -> Path:
    return paths.scriptorium_home() / "evals" / "reports"


def _args(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "run"):
        p = sub.add_parser(name)
        p.add_argument("task", choices=["mapping"])
        p.add_argument("--model", default=DEFAULT_MODEL)
        p.add_argument("--prompt", default="v1", help="prompt version under prompts/<task>/")
        p.add_argument("--effort", choices=EFFORTS, default="low")
        p.add_argument("--thinking", choices=THINKING_MODES, default="adaptive", help="'off' sends thinking type between_tools")
        p.add_argument("--max-tokens", type=int, default=1024)
        p.add_argument("--limit", type=int, default=None, help="use only the first N test items (a cheap trial)")
        p.add_argument("--budget-usd", type=float, default=DEFAULT_BUDGET)
        p.add_argument("--cache", type=Path, default=None)
    r = sub.choices["run"]
    r.add_argument("--yes", action="store_true", help="confirm that this run may spend up to the budget")
    r.add_argument("--replay", action="store_true", help="answer only from the cache; make no API calls")
    r.add_argument("--refresh", action="store_true", help="ignore cached answers and call the API again")
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--workspace-id", default=None, help="needed when the API key is not scoped to a workspace (or set ANTHROPIC_WORKSPACE_ID)")
    r.add_argument("--out", type=Path, default=None, help="report directory (default: ~/.scriptorium/evals/reports)")
    return ap.parse_args(argv)


def _prepare(a):
    prompt = load_prompt(a.task, a.prompt)
    test = baseline.load_jsonl(ingest.PROCESSED_DIR / "test.jsonl")
    train = baseline.load_jsonl(ingest.PROCESSED_DIR / "train.jsonl")
    items = test[: a.limit] if a.limit else test
    families = mapping.families_from(train + test)
    reqs = mapping.build_requests(items, prompt, families, model=a.model, effort=a.effort, thinking=a.thinking, max_tokens=a.max_tokens)
    return prompt, items, families, reqs


def _plan(a, prompt, reqs, cache: ResponseCache) -> dict:
    price_for(a.model)  # refuses unpriced models
    cached = sum(1 for q in reqs if cache.get(cache_key(q, a.task, prompt.version, prompt.sha256)) is not None)
    todo = len(reqs) - cached
    per_in = estimate_tokens_offline(reqs[0].system + reqs[0].user) + SCHEMA_OVERHEAD_TOKENS
    p = price_for(a.model)
    projected = todo * (per_in * p.input + EST_OUTPUT_TOKENS * p.output) / 1_000_000
    return {"items": len(reqs), "cached": cached, "to_call": todo, "est_input_tokens_per_call": per_in,
            "est_output_tokens_per_call": EST_OUTPUT_TOKENS, "projected_usd": round(projected, 4)}


def main(argv: list[str] | None = None) -> int:
    a = _args(argv)
    try:
        prompt, items, families, reqs = _prepare(a)
        cache = ResponseCache(a.cache or default_cache_path())
        plan = _plan(a, prompt, reqs, cache)
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"{a.task} | model {a.model} | prompt {prompt.task}/{prompt.version} | effort {a.effort} | thinking {a.thinking}")
    print(f"items {plan['items']}, already cached {plan['cached']}, to call {plan['to_call']}")
    print(f"rough projection (offline estimate, not a quote): ${plan['projected_usd']:.4f}  | cap ${a.budget_usd:.2f}")
    if a.cmd == "plan":
        print("No API call was made. Run again with `run ... --yes` to spend, or `--replay` to use only the cache.")
        return 0

    if a.replay:
        client = CachedClient(None, cache, a.task, prompt.version, prompt.sha256, replay=True)
        budget = None
    else:
        if plan["to_call"] and not a.yes:
            print("Not running: this would call the API. Re-run with --yes to allow spending up to the cap.", file=sys.stderr)
            return 3
        if plan["projected_usd"] > a.budget_usd:
            print(f"Not running: projected ${plan['projected_usd']:.2f} exceeds the cap ${a.budget_usd:.2f}. Raise --budget-usd or use --limit.", file=sys.stderr)
            return 3
        budget = Budget(a.budget_usd)
        inner = LazyClient(lambda: AnthropicClient(workspace_id=a.workspace_id))  # credentials are needed only if a call is actually made
        client = CachedClient(inner, cache, a.task, prompt.version, prompt.sha256, budget=budget, refresh=a.refresh)

    try:
        outcomes = mapping.run_items(client, items, reqs, sorted(families), a.workers)
    except LLMError as e:
        print(f"error: {e}", file=sys.stderr)
        return 4
    except ValueError as e:  # for example a malformed workspace id
        print(f"error: {e}", file=sys.stderr)
        return 2
    sk = mapping.sklearn_predictions(items)
    report = build_report(
        model=a.model, prompt=prompt, sample=reqs[0], outcomes=outcomes, labels=sorted(families), sk_pred=sk,
        live_calls=client.calls, cache_hits=client.hits, spent_this_run=budget.spent if budget else 0.0,
        budget_usd=None if a.replay else a.budget_usd, replay=a.replay, limit=a.limit, n_boot=baseline.N_BOOTSTRAP,
    )
    js, md = write_report(report, a.out or default_reports_dir())
    print(render_markdown(report))
    print(f"wrote {js}\nwrote {md}")
    return 0 if report["items"]["complete"] else 1
