"""Build and write the evaluation report: what ran, with what, what it cost, and what it scored."""

from __future__ import annotations

import json
import os
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scriptorium import ingest
from scriptorium.evals import mapping
from scriptorium.evals.budget import PRICES_AS_OF
from scriptorium.evals.llm import LLMRequest
from scriptorium.evals.prompts import Prompt


def _versions() -> dict[str, str]:
    import numpy
    import sklearn

    out = {"python": platform.python_version(), "scikit_learn": sklearn.__version__, "numpy": numpy.__version__}
    try:
        import anthropic

        out["anthropic_sdk"] = anthropic.__version__
    except ImportError:  # replay without the SDK installed
        pass
    return out


def build_report(*, model: str, prompt: Prompt, sample: LLMRequest, outcomes: list[mapping.Outcome], labels: list[str],
                 sk_pred: dict[str, str] | None, live_calls: int, cache_hits: int, spent_this_run: float, budget_usd: float | None,
                 replay: bool, limit: int | None, n_boot: int, split: str = "test", condition: str = "none",
                 only_enhancements: bool = False, k: int | None = None, input_fingerprint: str | None = None) -> dict[str, Any]:
    manifest = json.loads(ingest.MANIFEST_PATH.read_text(encoding="utf-8"))
    scored = mapping.score(outcomes, labels, n_boot)
    status = Counter(o.status for o in outcomes)
    n_not_run = status.get("not_run", 0)
    report = {
        "task": mapping.TASK,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "replay (cache only, no API calls)" if replay else ("live" if live_calls else "live (all answers from cache)"),
        "model": model,
        "prompt": {"task": prompt.task, "version": prompt.version, "sha256": prompt.sha256},
        "request_parameters": {"effort": sample.effort, "thinking": sample.thinking, "max_tokens": sample.max_tokens},
        "data": {"split": split, "limit": limit, "only_enhancements": only_enhancements, "source_commit": manifest["source"]["commit"],
                 "train_sha256": manifest["outputs"]["train.jsonl"], "test_sha256": manifest["outputs"]["test.jsonl"]},
        "condition": {"name": condition, "examples_k": k if condition.startswith("examples") else None, "input_sha256_16": input_fingerprint},
        "items": {"total": len(outcomes), "scored": len(outcomes) - n_not_run, "not_run": n_not_run, "complete": n_not_run == 0},
        "status_counts": dict(sorted(status.items())),
        "metrics": scored,
        "comparison_with_sklearn": mapping.compare_with_sklearn(outcomes, sk_pred, labels, n_boot) if sk_pred else None,
        "usage": {
            "input_tokens": sum(o.input_tokens for o in outcomes),
            "output_tokens": sum(o.output_tokens for o in outcomes),
            "cache_read_tokens": sum(o.cache_read_tokens for o in outcomes),
            "cache_write_tokens": sum(o.cache_write_tokens for o in outcomes),
            "recorded_cost_usd_all_answers": round(sum(o.cost for o in outcomes), 6),
            "spent_this_run_usd": round(spent_this_run, 6),
            "live_calls": live_calls,
            "answers_from_cache": cache_hits,
            "budget_usd": budget_usd,
            "prices_as_of": PRICES_AS_OF,
        },
        "predictions": [{"id": o.id, "true": o.true, "pred": o.pred, "status": o.status} for o in outcomes],
        "versions": _versions(),
        "caveats": [
            "Costs are computed from recorded token counts at listed prices and may differ from the invoice.",
            "Model output is not deterministic; answers served from the cache are reproducible, new calls may differ.",
        ],
    }
    if n_not_run:
        report["caveats"].append(f"INCOMPLETE: {n_not_run} item(s) were not run, so the metrics cover only the scored items.")
    return report


def render_markdown(r: dict[str, Any]) -> str:
    m, u, it = r["metrics"], r["usage"], r["items"]
    lines = [
        f"# Evaluation report: {r['task']}", "",
        f"- Model: `{r['model']}`", f"- Prompt: `{r['prompt']['task']}/{r['prompt']['version']}` (sha256 `{r['prompt']['sha256'][:12]}`)",
        f"- Parameters: effort `{r['request_parameters']['effort']}`, thinking `{r['request_parameters']['thinking']}`, max_tokens {r['request_parameters']['max_tokens']}",
        f"- Split: `{r['data']['split']}`" + (" (enhancements only)" if r['data'].get('only_enhancements') else "") + f", condition `{r['condition']['name']}`",
        f"- Date (UTC): {r['created_utc']}", f"- Mode: {r['mode']}",
        f"- Items: {it['scored']} scored of {it['total']}" + ("" if it["complete"] else " (**incomplete**)"),
        f"- Statuses: {r['status_counts']}",
        f"- Tokens: {u['input_tokens']} input, {u['output_tokens']} output, {u['cache_read_tokens']} cache-read, {u['cache_write_tokens']} cache-write",
        f"- Cost this run: ${u['spent_this_run_usd']:.4f} ({u['live_calls']} live calls, {u['answers_from_cache']} from cache); prices as of {u['prices_as_of']}", "",
    ]
    if m.get("n_scored"):
        ci = m["ci95"]
        lines += [
            "## Held-out scores", "", "| Metric | Value | 95% CI (bootstrap over base controls) |", "|---|---|---|",
            f"| Accuracy | {m['accuracy']:.3f} | {ci['accuracy'][0]:.3f} to {ci['accuracy'][1]:.3f} |",
            f"| Macro-F1 | {m['macro_f1']:.3f} | {ci['macro_f1'][0]:.3f} to {ci['macro_f1'][1]:.3f} |", "",
        ]
    c = r.get("comparison_with_sklearn")
    if c and c.get("n_paired"):
        d, dci = c["llm_minus_sklearn"], c["llm_minus_sklearn_ci95"]
        lines += [
            "## Against the scikit-learn baseline (same items)", "",
            f"- sklearn accuracy {c['sklearn']['accuracy']:.3f}, macro-F1 {c['sklearn']['macro_f1']:.3f}",
            f"- LLM minus sklearn, accuracy: {d['accuracy_diff']:+.3f} (95% CI {dci['accuracy_diff'][0]:+.3f} to {dci['accuracy_diff'][1]:+.3f})",
            f"- LLM minus sklearn, macro-F1: {d['macro_f1_diff']:+.3f} (95% CI {dci['macro_f1_diff'][0]:+.3f} to {dci['macro_f1_diff'][1]:+.3f})",
            f"- Both right {c['both_right']}, only LLM {c['only_llm_right']}, only sklearn {c['only_sklearn_right']}, neither {c['neither_right']}", "",
        ]
    lines += ["## Caveats", ""] + [f"- {x}" for x in r["caveats"]]
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    stamp = report["created_utc"].replace("-", "").replace(":", "").replace("+0000", "Z").replace("+00:00", "Z")
    stem = f"{report['task']}-{report['model']}-{report['prompt']['version']}-{stamp}"
    js, md = out_dir / f"{stem}.json", out_dir / f"{stem}.md"
    for path, body in ((js, json.dumps(report, indent=1) + "\n"), (md, render_markdown(report))):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(body)
    return js, md
