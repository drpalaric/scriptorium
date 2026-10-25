"""Duplicate detection: scikit-learn similarity against an LLM judge (pre-registered in docs/experiments/overlap.md).

    python -m scriptorium.evals overlap --plan
    python -m scriptorium.evals overlap --yes --allow-cis        # sends CIS text and public NIST text to the API
    python -m scriptorium.evals overlap --skip-llm               # scikit-learn arms only, no API

Real pair sets use CIS text (local only). Aggregate summaries are the only publishable output; pair-level scores stay local.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from scriptorium import baseline, ingest, paths
from scriptorium.crosswalks import MappingStore, store_dir
from scriptorium.evals import cis, mapping
from scriptorium.evals.budget import Budget, price_for, projected_cost
from scriptorium.evals.llm import AnthropicClient, CachedClient, FatalLLMError, LazyClient, LLMError, LLMRequest, LLMResponse, ResponseCache, cache_key
from scriptorium.evals.prompts import load_prompt
from scriptorium.overlap import Scorer, nist_corpus

SEED = 20261025
PREREG = "docs/experiments/overlap.md (commit ebdc7c6)"
RELATIONS = ("different", "overlap", "same")  # scored 0, 1, 2
N_OVERLAP_POS = 100
N_PARAPHRASE = 200
SK_ARMS = ("word", "char", "lsa")
_TAG_RE = re.compile(r"<(/?)([ab])\b", re.IGNORECASE)
JUDGE_SCHEMA = {"type": "object", "properties": {"relation": {"type": "string", "enum": list(RELATIONS)}}, "required": ["relation"], "additionalProperties": False}
PARA_SCHEMA = {"type": "object", "properties": {"rewrite": {"type": "string"}}, "required": ["rewrite"], "additionalProperties": False}


def _neut(text: str) -> str:
    return _TAG_RE.sub(lambda m: f"&lt;{m.group(1)}{m.group(2)}", text)


def pair_message(a: str, b: str) -> str:
    return f"<a>\n{_neut(a)}\n</a>\n<b>\n{_neut(b)}\n</b>"


# ---- pair sets ---------------------------------------------------------------------------------------------


def _negatives(rng: random.Random, sg: str, nid: str, fam_of: dict[str, str], ids_by_fam: dict[str, list[str]], related: dict[str, set[str]]) -> list[str]:
    bad = related.get(sg, set()) | {nid}
    same = [i for i in ids_by_fam[fam_of[nid]] if i not in bad]
    other = [i for f, v in ids_by_fam.items() if f != fam_of[nid] for i in v if i not in bad]
    return [rng.choice(same)] + [rng.choice(other) for _ in range(2)]


def build_real_sets(store: MappingStore, nist_rows: list[dict[str, Any]] | None = None, n_overlap: int = N_OVERLAP_POS, seed: int = SEED) -> dict[str, list[dict[str, Any]]]:
    safeguards, _ = cis.load_cis(store)
    nist = nist_rows if nist_rows is not None else cis._nist_rows()
    text_of, fam_of = {r["id"]: r["text"] for r in nist}, {r["id"]: r["family"] for r in nist}
    ids_by_fam: dict[str, list[str]] = {}
    for r in nist:
        ids_by_fam.setdefault(r["family"], []).append(r["id"])
    pairs = [m for m in cis._cis_pairs(store) if m["to_id"] in text_of and m["from_id"] in safeguards]
    related: dict[str, set[str]] = {}
    for m in pairs:
        related.setdefault(m["from_id"], set()).add(m["to_id"])
    rng = random.Random(f"{seed}:real")

    def build(positives: list[dict]) -> list[dict[str, Any]]:
        out = []
        for m in sorted(positives, key=lambda m: (m["from_id"], m["to_id"])):
            sg, nid = m["from_id"], m["to_id"]
            a = cis.strip_cis_refs(safeguards[sg]["text"])[0]
            group = safeguards[sg]["meta"]["control"]
            out.append({"a": a, "b": text_of[nid], "label": 1, "group": group, "ids": (sg, nid), "kind": m["relationship"]})
            for neg in _negatives(rng, sg, nid, fam_of, ids_by_fam, related):
                out.append({"a": a, "b": text_of[neg], "label": 0, "group": group, "ids": (sg, neg), "kind": "negative"})
        return out

    equiv = [m for m in pairs if m["relationship"] == "equivalent"]
    partial = sorted((m for m in pairs if m["relationship"] in ("subset", "superset")), key=lambda m: (m["from_id"], m["to_id"]))
    sample = random.Random(f"{seed}:overlap").sample(partial, min(n_overlap, len(partial)))
    return {"real_equivalent": build(equiv), "real_overlap": build(sample)}


def paraphrase_requests(nist_rows: list[dict[str, Any]], model: str, n: int = N_PARAPHRASE, seed: int = SEED) -> tuple[list[dict[str, Any]], list[LLMRequest]]:
    pool = sorted((r for r in nist_rows if len(r["text"]) >= 60), key=lambda r: r["id"])
    items = sorted(random.Random(f"{seed}:para").sample(pool, min(n, len(pool))), key=lambda r: r["id"])
    prompt = load_prompt("overlap", "paraphrase-v1")
    from scriptorium.evals.context import _block

    reqs = [LLMRequest(model, prompt.template, _block("requirement", r["text"]), PARA_SCHEMA, 1024, "low", "adaptive") for r in items]
    return items, reqs


def build_paraphrase_pairs(items: list[dict[str, Any]], rewrites: list[str | None], nist_rows: list[dict[str, Any]], seed: int = SEED) -> list[dict[str, Any]]:
    by_fam: dict[str, list[dict]] = {}
    for r in nist_rows:
        by_fam.setdefault(r["family"], []).append(r)
    rng = random.Random(f"{seed}:paraneg")
    out = []
    for r, rw in zip(items, rewrites):
        if not rw or not rw.strip():
            continue
        others = [x for x in by_fam[r["family"]] if x["base_id"] != r["base_id"]]
        out.append({"a": r["text"], "b": rw.strip(), "label": 1, "group": r["base_id"], "ids": (r["id"], "rewrite"), "kind": "paraphrase"})
        out.append({"a": rw.strip(), "b": rng.choice(others)["text"], "label": 0, "group": r["base_id"], "ids": ("rewrite", "other"), "kind": "negative"})
    return out


# ---- scoring -----------------------------------------------------------------------------------------------


def _auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    return float(roc_auc_score(labels, scores)) if len(set(labels.tolist())) == 2 else None


def _group_boot(groups: list[str], fn, n: int = baseline.N_BOOTSTRAP, seed: int = baseline.SEED) -> list[float] | None:
    by: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        by.setdefault(g, []).append(i)
    gl = [np.array(v) for v in by.values()]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        v = fn(np.concatenate([gl[j] for j in rng.integers(0, len(gl), len(gl))]))
        if v is not None:
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else None


def calibrated_prf(labels: np.ndarray, scores: np.ndarray, groups: list[str]) -> dict[str, Any]:
    """Threshold chosen on one half of the groups, applied to the other half (both orders); decisions pooled."""
    gs = sorted(set(groups))
    fold = {g: i % 2 for i, g in enumerate(gs)}
    f = np.array([fold[g] for g in groups])
    pred = np.zeros(len(labels), dtype=bool)
    cuts = []
    for k in (0, 1):
        cal, test = f != k, f == k
        cand = np.unique(scores[cal])
        best = max(cand, key=lambda t: _f1(labels[cal], scores[cal] >= t)) if len(cand) else 1.0
        pred[test] = scores[test] >= best
        cuts.append(float(best))
    return {**_prf(labels, pred), "thresholds_by_fold": cuts}


def _f1(labels, pred) -> float:
    return _prf(labels, pred)["f1"]


def _prf(labels: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    tp, fp, fn = int(((labels == 1) & pred).sum()), int(((labels == 0) & pred).sum()), int(((labels == 1) & ~pred).sum())
    p, r = tp / (tp + fp) if tp + fp else 0.0, tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def score_set(pairs: list[dict[str, Any]], sk_scores: dict[str, np.ndarray], llm_levels: np.ndarray | None, positive_rule: str, n_boot: int = baseline.N_BOOTSTRAP) -> dict[str, Any]:
    labels = np.array([p["label"] for p in pairs])
    groups = [p["group"] for p in pairs]
    arms: dict[str, Any] = {}
    for name, s in sk_scores.items():
        arms[name] = {"auc": _auc(labels, s), "auc_ci95": _group_boot(groups, lambda idx, s=s: _auc(labels[idx], s[idx]), n_boot), "decision": calibrated_prf(labels, s, groups)}
    if llm_levels is not None:
        thr = 2 if positive_rule == "same" else 1
        arms["llm"] = {"auc": _auc(labels, llm_levels), "auc_ci95": _group_boot(groups, lambda idx: _auc(labels[idx], llm_levels[idx]), n_boot),
                       "decision": {**_prf(labels, llm_levels >= thr), "positive_rule": positive_rule}}
        if "char" in sk_scores:
            c = sk_scores["char"]
            d = lambda idx: (lambda a, b: None if a is None or b is None else a - b)(_auc(labels[idx], llm_levels[idx]), _auc(labels[idx], c[idx]))
            arms["llm_minus_char_auc"] = {"value": d(np.arange(len(labels))), "ci95": _group_boot(groups, d, n_boot)}
    return {"n_pairs": len(pairs), "n_positive": int(labels.sum()), "arms": arms}


# ---- running ------------------------------------------------------------------------------------------------


def _run(client: CachedClient, reqs: list[LLMRequest], workers: int = 4) -> list[LLMResponse | None]:
    stop = {"fatal": None}

    def one(req):
        if stop["fatal"] is not None:
            return None
        try:
            return client.complete(req)
        except FatalLLMError as e:
            stop["fatal"] = stop["fatal"] or e
        except LLMError:
            pass
        return None

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        out = list(pool.map(one, reqs))
    if stop["fatal"] is not None:
        raise stop["fatal"]
    return out


def _parse(resp: LLMResponse | None, key: str) -> str | None:
    if resp is None or resp.stop_reason in ("refusal", "max_tokens") or not resp.text:
        return None
    try:
        v = json.loads(resp.text)[key]
    except (ValueError, KeyError, TypeError):
        return None
    return v if isinstance(v, str) else None


def add_parser(sub) -> None:
    p = sub.add_parser("overlap", help="duplicate detection: scikit-learn against an LLM (pre-registered)")
    p.add_argument("--plan", action="store_true")
    p.add_argument("--skip-llm", action="store_true", help="run only the scikit-learn arms (no API)")
    p.add_argument("--model", default="claude-sonnet-5-5")
    p.add_argument("--budget-usd", type=float, default=5.0)
    p.add_argument("--yes", action="store_true")
    p.add_argument("--allow-cis", action="store_true", help="allow CIS text to be sent to the API (needed for the real pair sets)")
    p.add_argument("--replay", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--workspace-id", default=None)
    p.add_argument("--cache", type=Path, default=None)
    p.add_argument("--store", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)


def _write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")


def run(a) -> int:
    out_dir = a.out or paths.scriptorium_home() / "evals" / "overlap"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    try:
        store = MappingStore.load(a.store or store_dir())
        nist = cis._nist_rows()
        sets = build_real_sets(store, nist)
        judge = load_prompt("overlap", "judge-v1")
        cache = ResponseCache(a.cache or (paths.scriptorium_home() / "evals" / "cache.db"))
        price_for(a.model)
        p_items, p_reqs = paraphrase_requests(nist, a.model)
        p_task, p_prompt = "overlap_paraphrase", load_prompt("overlap", "paraphrase-v1")
        real_pairs = [p for k in ("real_equivalent", "real_overlap") for p in sets[k]]
        j_reqs_real = [LLMRequest(a.model, judge.template, pair_message(p["a"], p["b"]), JUDGE_SCHEMA, 256, "low", "adaptive") for p in real_pairs]
        unc_p = sum(cache.get(cache_key(q, p_task, p_prompt.version, p_prompt.sha256)) is None for q in p_reqs)
        unc_r = sum(cache.get(cache_key(q, "overlap_judge", judge.version, judge.sha256)) is None for q in j_reqs_real)
        proj = (projected_cost(a.model, p_reqs, unc_p)[0] + projected_cost(a.model, j_reqs_real, unc_r)[0] + projected_cost(a.model, j_reqs_real[:1] * 400, 400)[0]) if not a.skip_llm else 0.0
    except (ValueError, FileNotFoundError, RuntimeError, cis.CisDataError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"overlap | real-equivalent {len(sets['real_equivalent'])} pairs, real-overlap {len(sets['real_overlap'])} pairs, paraphrase up to {2 * len(p_items)} pairs | LLM calls to make ~{unc_p + unc_r + 400}")
    print(f"rough projection (offline estimate, not a quote): ${proj:.4f} | cap ${a.budget_usd:.2f}")
    if a.plan:
        print("No API call was made.")
        return 0

    llm_on = not a.skip_llm
    budget, client_kw = None, {}
    if llm_on and not a.replay:
        if (unc_p + unc_r) and not a.allow_cis:
            print("Not running the LLM arm: the real pair sets send CIS text to the Anthropic API. Re-run with --allow-cis (and --yes), or use --skip-llm.", file=sys.stderr)
            return 3
        if not a.yes:
            print("Not running: this would spend money. Re-run with --yes to allow spending up to the cap.", file=sys.stderr)
            return 3
        if proj > a.budget_usd:
            print(f"Not running: projected ${proj:.2f} exceeds the cap ${a.budget_usd:.2f}.", file=sys.stderr)
            return 3
        budget = Budget(a.budget_usd)
    inner = None if (a.replay or not llm_on) else LazyClient(lambda: AnthropicClient(workspace_id=a.workspace_id))
    spent_before = 0.0
    corpus = nist_corpus() + [cis.strip_cis_refs(e["text"])[0] for (fw, _), e in store.elements.items() if fw == "cis-8.1" and e["text"]]
    scorers = {v: Scorer(v, corpus) for v in SK_ARMS}

    try:
        t0 = time.time()
        sets["paraphrase"] = []
        llm_t = 0.0
        if llm_on:
            pc = CachedClient(inner, cache, p_task, p_prompt.version, p_prompt.sha256, budget=budget, replay=a.replay)
            t1 = time.time()
            rewrites = [_parse(r, "rewrite") for r in _run(pc, p_reqs, a.workers)]
            llm_t += time.time() - t1
            sets["paraphrase"] = build_paraphrase_pairs(p_items, rewrites, nist)
        levels: dict[str, np.ndarray] = {}
        jc = CachedClient(inner, cache, "overlap_judge", judge.version, judge.sha256, budget=budget, replay=a.replay) if llm_on else None
        for name, pairs in sets.items():
            if llm_on and pairs:
                reqs = [LLMRequest(a.model, judge.template, pair_message(p["a"], p["b"]), JUDGE_SCHEMA, 256, "low", "adaptive") for p in pairs]
                t1 = time.time()
                resp = _run(jc, reqs, a.workers)
                llm_t += time.time() - t1
                levels[name] = np.array([RELATIONS.index(r) if (r := _parse(x, "relation")) in RELATIONS else 0 for x in resp])
    except LLMError as e:
        print(f"error: {e}", file=sys.stderr)
        return 4

    results = {}
    sk_time = {}
    for name, pairs in sets.items():
        if not pairs:
            continue
        t1 = time.time()
        sk = {v: sc.pair_scores([p["a"] for p in pairs], [p["b"] for p in pairs]) for v, sc in scorers.items()}
        sk_time[name] = time.time() - t1
        rule = "same_or_overlap" if name == "real_overlap" else "same"
        results[name] = score_set(pairs, sk, levels.get(name), rule)
    summary = {"experiment": "overlap", "preregistration": PREREG, "created_utc": stamp, "model": a.model if llm_on else None,
               "mode": "scikit-learn only" if not llm_on else ("replay (cache only)" if a.replay else "live"), "allow_cis": bool(a.allow_cis),
               "prompts": {"judge": {"version": judge.version, "sha256": judge.sha256}, "paraphrase": {"version": p_prompt.version, "sha256": p_prompt.sha256}},
               "pairs": {k: {"n": len(v), "positives": sum(p["label"] for p in v)} for k, v in sets.items()},
               "cost_and_time": {"scikit_learn_seconds_score_only": {k: round(v, 3) for k, v in sk_time.items()}, "llm_wall_seconds": round(llm_t, 1) if llm_on else None,
                                 "spent_this_run_usd": round(budget.spent, 6) if budget else 0.0, "llm_calls_live": (jc.calls + pc.calls) if llm_on else 0,
                                 "answers_from_cache": (jc.hits + pc.hits) if llm_on else 0},
               "results": results, "h1_llm_beats_char_on_real_equivalent": None}
    d = results.get("real_equivalent", {}).get("arms", {}).get("llm_minus_char_auc")
    if d and d["ci95"]:
        summary["h1_llm_beats_char_on_real_equivalent"] = bool(d["ci95"][0] > 0)
    local = {"note": "LOCAL ONLY: pair identifiers and scores (CIS-derived)", "sets": {k: [{"ids": p["ids"], "label": p["label"], "kind": p["kind"]} for p in v] for k, v in sets.items()}}
    spath, lpath = out_dir / f"overlap-{stamp}-summary.json", out_dir / f"overlap-{stamp}-pairs.json"
    _write(spath, summary)
    _write(lpath, local)
    print(json.dumps(summary, indent=1))
    print(f"wrote {spath}\nwrote {lpath} (local only)")
    return 0
