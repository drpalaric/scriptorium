"""Command line for the evaluation harness.

    python -m scriptorium.evals plan    mapping --prompt v1 --split dev-b
    python -m scriptorium.evals run     mapping --prompt v3 --split dev-b --yes
    python -m scriptorium.evals run     mapping --prompt v1 --replay
    python -m scriptorium.evals compare mapping --split dev-b --a v1 --b v3

`plan` makes no API call. `run` spends money only with `--yes`, never above `--budget-usd`, and answers from
the local cache whenever it can. `--replay` makes no API call at all. Prompts other than v1 may not be run on
the frozen test split without `--final` (see docs/experiments/prompt-v2.md).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scriptorium import baseline, ingest, paths
from scriptorium.evals import cis_cli, context, injection, mapping, splits
from scriptorium.evals.budget import Budget, EST_OUTPUT_BY_EFFORT, price_for, projected_cost
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


def _common(p):
    p.add_argument("task", choices=["mapping"])
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--effort", choices=EFFORTS, default="low")
    p.add_argument("--thinking", choices=THINKING_MODES, default="adaptive", help="'off' sends thinking type between_tools")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--split", choices=splits.SPLITS, default="test")
    p.add_argument("--only-enhancements", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="use only the first N items (a cheap trial)")
    p.add_argument("--k", type=int, default=context.EXAMPLES_K, help="examples per item for v3/v4 (registered value: 8)")
    p.add_argument("--cache", type=Path, default=None)


def _args(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "run"):
        p = sub.add_parser(name)
        _common(p)
        p.add_argument("--prompt", default="v1", help="prompt version under prompts/<task>/")
        p.add_argument("--budget-usd", type=float, default=DEFAULT_BUDGET)
    r = sub.choices["run"]
    r.add_argument("--yes", action="store_true", help="confirm that this run may spend up to the budget")
    r.add_argument("--replay", action="store_true", help="answer only from the cache; make no API calls")
    r.add_argument("--refresh", action="store_true", help="ignore cached answers and call the API again")
    r.add_argument("--final", action="store_true", help="allow a non-v1 prompt on the frozen test split (the single registered run)")
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--workspace-id", default=None, help="needed when the API key is not scoped to a workspace (or set ANTHROPIC_WORKSPACE_ID)")
    r.add_argument("--out", type=Path, default=None, help="report directory (default: ~/.scriptorium/evals/reports)")
    cis_cli.add_parser(sub)
    injection.add_parser(sub)
    c = sub.add_parser("compare", help="paired comparison of two cached runs (no API call)")
    _common(c)
    c.add_argument("--a", required=True, help="reference prompt version")
    c.add_argument("--b", required=True, help="candidate prompt version (b minus a is reported)")
    c.add_argument("--out", type=Path, default=None)
    return ap.parse_args(argv)


def _prepare(a, prompt_version: str):
    """Everything needed to run one condition on one split. Context is built from the split's own pool only."""
    condition = context.CONDITIONS.get(prompt_version)
    if condition is None:
        raise ValueError(f"prompt version {prompt_version!r} has no registered condition; known: {sorted(context.CONDITIONS)}")
    prompt = load_prompt(a.task, prompt_version)
    all_rows = baseline.load_jsonl(ingest.PROCESSED_DIR / "train.jsonl") + baseline.load_jsonl(ingest.PROCESSED_DIR / "test.jsonl")
    families = mapping.families_from(all_rows)
    split_rows = splits.load_split(a.split)
    only_enh = a.only_enhancements or condition == "parent_only"
    items = [r for r in split_rows if r["is_enhancement"]] if only_enh else split_rows
    items = items[: a.limit] if a.limit else items
    pool = splits.pool_for(a.split) if condition.startswith("examples") else None
    guidance = context.guidance_by_id() if condition == "examples_guidance" else None
    msgs = context.build_user_messages(condition, items, lookup=split_rows, pool=pool, k=a.k, guidance=guidance)
    reqs = mapping.build_requests(items, prompt, families, model=a.model, effort=a.effort, thinking=a.thinking, max_tokens=a.max_tokens, user_messages=msgs)
    return prompt, condition, items, families, reqs, only_enh, context.fingerprint(msgs)


def _plan(model, task, prompt, reqs, cache: ResponseCache) -> dict:
    price_for(model)  # refuses unpriced models
    cached = sum(1 for q in reqs if cache.get(cache_key(q, task, prompt.version, prompt.sha256)) is not None)
    todo = len(reqs) - cached
    projected, per_in = projected_cost(model, reqs, todo, schema_overhead_tokens=SCHEMA_OVERHEAD_TOKENS)
    return {"items": len(reqs), "cached": cached, "to_call": todo, "est_input_tokens_per_call": per_in,
            "est_output_tokens_per_call": EST_OUTPUT_BY_EFFORT.get(reqs[0].effort, 300), "projected_usd": round(projected, 4)}


def _cmd_compare(a) -> int:
    cache = ResponseCache(a.cache or default_cache_path())
    runs = {}
    for v in (a.a, a.b):
        prompt, condition, items, families, reqs, _, _ = _prepare(a, v)
        client = CachedClient(None, cache, a.task, prompt.version, prompt.sha256, replay=True)
        runs[v] = (mapping.run_items(client, items, reqs, sorted(families)), families)
    labels = sorted(runs[a.a][1])
    only = {o.id for o in runs[a.a][0]} & {o.id for o in runs[a.b][0]}
    if a.only_enhancements:
        enh = {r["id"] for r in splits.load_split(a.split) if r["is_enhancement"]}
        only &= enh
    cmp = mapping.compare_runs(runs[a.a][0], runs[a.b][0], labels, only_ids=only)
    out = {"task": a.task, "split": a.split, "only_enhancements": a.only_enhancements, "a": a.a, "b": a.b, "model": a.model,
           "note": "b minus a over items both runs answered; 95% intervals resample base controls", **cmp,
           "h1_rule_met": mapping.h1_supported(cmp)}
    d = a.out or default_reports_dir()
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = d / f"compare-{a.split}-{a.a}-vs-{a.b}{'-enh' if a.only_enhancements else ''}.json"
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))
    print(f"wrote {path}")
    return 0 if cmp.get("n_paired") else 1


def main(argv: list[str] | None = None) -> int:
    a = _args(argv)
    if a.cmd == "cis":
        return cis_cli.run(a)
    if a.cmd == "injection":
        return injection.run(a)
    try:
        if a.cmd == "compare":
            return _cmd_compare(a)
        if a.split == "test" and a.prompt != "v1" and a.cmd == "run" and not a.final:
            print("error: prompts other than v1 may not run on the frozen test split without --final (docs/experiments/prompt-v2.md)", file=sys.stderr)
            return 2
        prompt, condition, items, families, reqs, only_enh, fingerprint = _prepare(a, a.prompt)
        cache = ResponseCache(a.cache or default_cache_path())
        plan = _plan(a.model, a.task, prompt, reqs, cache)
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"{a.task} | model {a.model} | prompt {prompt.task}/{prompt.version} ({condition}) | split {a.split}{' (enhancements)' if only_enh else ''} | effort {a.effort} | thinking {a.thinking}")
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
    sk = mapping.sklearn_predictions(items) if a.split == "test" and not only_enh else None
    report = build_report(
        model=a.model, prompt=prompt, sample=reqs[0], outcomes=outcomes, labels=sorted(families), sk_pred=sk,
        live_calls=client.calls, cache_hits=client.hits, spent_this_run=budget.spent if budget else 0.0,
        budget_usd=None if a.replay else a.budget_usd, replay=a.replay, limit=a.limit, n_boot=baseline.N_BOOTSTRAP,
        split=a.split, condition=condition, only_enhancements=only_enh, k=a.k, input_fingerprint=fingerprint,
    )
    js, md = write_report(report, a.out or default_reports_dir())
    print(render_markdown(report))
    print(f"wrote {js}\nwrote {md}")
    return 0 if report["items"]["complete"] else 1
