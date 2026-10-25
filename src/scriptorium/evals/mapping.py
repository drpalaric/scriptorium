"""Task 1: classify NIST requirement text into a control family with an LLM.

Uses the same frozen held-out split as the scikit-learn baseline (`data/processed/test.jsonl`),
so the two are scored on identical items and can be compared item by item. Output must match
the schema exactly; anything else is a failure and is never repaired.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError
from sklearn.metrics import accuracy_score, f1_score

from scriptorium import baseline, ingest
from scriptorium.evals.budget import BudgetExceeded
from scriptorium.evals.llm import CacheMiss, CachedClient, FatalLLMError, LLMError, LLMRequest, LLMResponse
from scriptorium.evals.prompts import Prompt

TASK = "mapping"
INVALID = "INVALID"
_TAG_RE = re.compile(r"<(/?)requirement", re.IGNORECASE)


def families_from(rows: list[dict[str, Any]]) -> dict[str, str]:
    return {r["family"]: r["family_title"] for r in sorted(rows, key=lambda r: r["family"])}


def answer_model(codes: list[str]) -> type[BaseModel]:
    class MappingAnswer(BaseModel):
        model_config = ConfigDict(extra="forbid")
        family: Literal[tuple(codes)]  # type: ignore[valid-type]

    return MappingAnswer


def json_schema(codes: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"family": {"type": "string", "enum": sorted(codes)}},
        "required": ["family"],
        "additionalProperties": False,
    }


def system_prompt(prompt: Prompt, families: dict[str, str]) -> str:
    return prompt.render(family_list="\n".join(f"{c}: {t}" for c, t in sorted(families.items())))


def user_message(text: str) -> str:
    """Requirement text, quoted as data. Anything that looks like our own tag is neutralised."""
    return f"<requirement>\n{_TAG_RE.sub(lambda m: '&lt;' + m.group(1) + 'requirement', text)}\n</requirement>"


def parse_answer(resp: LLMResponse, model_cls: type[BaseModel]) -> tuple[str | None, str]:
    """Return (family, status). Anything that is not a clean, schema-valid answer is a failure."""
    if resp.stop_reason == "refusal":
        return None, "refusal"
    if resp.stop_reason == "max_tokens":
        return None, "max_tokens"
    if not resp.text or not resp.text.strip():
        return None, "invalid_json"
    try:
        data = json.loads(resp.text)
    except (ValueError, RecursionError):
        return None, "invalid_json"
    try:
        return model_cls.model_validate(data).family, "ok"  # type: ignore[attr-defined]
    except ValidationError:
        return None, "invalid_schema"


@dataclass
class Outcome:
    id: str
    base_id: str
    true: str
    pred: str | None
    status: str  # ok | invalid_json | invalid_schema | refusal | max_tokens | error | not_run
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost: float = 0.0
    from_cache: bool = False
    detail: str | None = None


def build_requests(items: list[dict[str, Any]], prompt: Prompt, families: dict[str, str], *, model: str, effort: str,
                   thinking: str, max_tokens: int) -> list[LLMRequest]:
    system = system_prompt(prompt, families)
    schema = json_schema(list(families))
    return [LLMRequest(model, system, user_message(r["text"]), schema, max_tokens, effort, thinking) for r in items]


def run_items(client: CachedClient, items: list[dict[str, Any]], requests: list[LLMRequest], codes: list[str], workers: int = 4) -> list[Outcome]:
    model_cls = answer_model(codes)
    stop = {"fatal": None}

    def one(r: dict[str, Any], req: LLMRequest) -> Outcome:
        base = Outcome(r["id"], r["base_id"], r["family"], None, "not_run")
        if stop["fatal"] is not None:
            return base
        try:
            resp = client.complete(req)
        except BudgetExceeded as e:
            stop["fatal"] = stop["fatal"] or e
            base.detail = str(e)
            return base
        except CacheMiss as e:
            base.detail = str(e)
            return base
        except FatalLLMError as e:
            stop["fatal"] = stop["fatal"] or e
            base.detail = str(e)
            return base
        except LLMError as e:
            base.status, base.detail = "error", str(e)
            return base
        pred, status = parse_answer(resp, model_cls)
        base.pred, base.status = pred, status
        base.input_tokens, base.output_tokens = resp.input_tokens, resp.output_tokens
        base.cache_read_tokens, base.cache_write_tokens = resp.cache_read_tokens, resp.cache_write_tokens
        base.cost, base.from_cache = resp.cost(), resp.from_cache
        return base

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        outcomes = list(pool.map(one, items, requests))
    if isinstance(stop["fatal"], FatalLLMError):
        raise stop["fatal"]
    return outcomes


# ---- scoring -----------------------------------------------------------------------------------


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, float]:
    """Accuracy, and macro-F1 over the families that actually occur in `y_true` (so a --limit subset
    or a bootstrap resample is not penalised for families it does not contain). An INVALID prediction
    is never a label of its own: it is simply wrong."""
    present = [f for f in labels if (y_true == f).any()]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)),
    }


def _boot(bases: list[str], fn, n: int, seed: int) -> dict[str, list[float]]:
    """95% percentile intervals of `fn(idx)->dict`, resampling whole base controls."""
    by_base: dict[str, list[int]] = {}
    for i, b in enumerate(bases):
        by_base.setdefault(b, []).append(i)
    groups = [np.array(v) for v in by_base.values()]
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {}
    for _ in range(n):
        idx = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        for k, v in fn(idx).items():
            draws.setdefault(k, []).append(v)
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in draws.items()}


def score(outcomes: list[Outcome], labels: list[str], n_boot: int = baseline.N_BOOTSTRAP, seed: int = baseline.SEED) -> dict[str, Any]:
    done = [o for o in outcomes if o.status != "not_run"]
    if not done:
        return {"n_scored": 0}
    y = np.array([o.true for o in done])
    p = np.array([o.pred if o.pred is not None else INVALID for o in done])
    m = _metrics(y, p, labels)
    per_family = {}
    for f in labels:
        tp = int(((y == f) & (p == f)).sum())
        support, predicted = int((y == f).sum()), int((p == f).sum())
        prec = tp / predicted if predicted else 0.0
        rec = tp / support if support else 0.0
        per_family[f] = {"precision": prec, "recall": rec, "f1": (2 * prec * rec / (prec + rec)) if prec + rec else 0.0,
                         "support": support, "low_support": support < baseline.LOW_SUPPORT}
    ci = _boot([o.base_id for o in done], lambda idx: _metrics(y[idx], p[idx], labels), n_boot, seed)
    return {"n_scored": len(done), **m, "ci95": ci, "per_family": per_family}


def sklearn_predictions(items: list[dict[str, Any]]) -> dict[str, str]:
    """Predictions of the baseline model (config chosen on train only) for the same items."""
    train = baseline.load_jsonl(ingest.PROCESSED_DIR / "train.jsonl")
    cfg = json.loads((ingest.ROOT / "results" / "baseline_metrics.json").read_text(encoding="utf-8"))["selected_config"]
    pipe = baseline.build_pipeline(cfg).fit([r["text"] for r in train], [r["family"] for r in train])
    pred = pipe.predict([r["text"] for r in items])
    return {r["id"]: str(p) for r, p in zip(items, pred)}


def compare_with_sklearn(outcomes: list[Outcome], sk_pred: dict[str, str], labels: list[str], n_boot: int = baseline.N_BOOTSTRAP,
                         seed: int = baseline.SEED) -> dict[str, Any]:
    done = [o for o in outcomes if o.status != "not_run" and o.id in sk_pred]
    if not done:
        return {"n_paired": 0}
    y = np.array([o.true for o in done])
    llm = np.array([o.pred if o.pred is not None else INVALID for o in done])
    sk = np.array([sk_pred[o.id] for o in done])
    l_ok, s_ok = llm == y, sk == y
    diff = lambda idx: {
        "accuracy_diff": float(l_ok[idx].mean() - s_ok[idx].mean()),
        "macro_f1_diff": _metrics(y[idx], llm[idx], labels)["macro_f1"] - _metrics(y[idx], sk[idx], labels)["macro_f1"],
    }
    point = diff(np.arange(len(done)))
    return {
        "n_paired": len(done),
        "sklearn": _metrics(y, sk, labels),
        "llm_minus_sklearn": point,
        "llm_minus_sklearn_ci95": _boot([o.base_id for o in done], diff, n_boot, seed),
        "both_right": int((l_ok & s_ok).sum()),
        "only_llm_right": int((l_ok & ~s_ok).sum()),
        "only_sklearn_right": int((~l_ok & s_ok).sum()),
        "neither_right": int((~l_ok & ~s_ok).sum()),
    }
