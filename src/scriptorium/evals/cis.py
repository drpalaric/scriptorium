"""CIS comparison experiments E1, E2 and E3 (see docs/experiments/cis.md, pre-registered).

    E2 (primary)  CIS safeguard text -> NIST family, scored against the official CIS-to-NIST pairs
    E1            CIS safeguard text -> its CIS Control (1-18)
    E3            TF-IDF retrieval between CIS safeguards and NIST controls, both directions (no LLM)

Inputs are user-supplied and local only. CIS text reaches the API only with an explicit opt-in (`--allow-cis`).
Aggregate summaries contain no CIS text or IDs and are the only thing meant for public documents; the
per-item file stays local.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold

from scriptorium import baseline, crosswalk, ingest
from scriptorium.crosswalks import MappingStore
from scriptorium.evals import context, mapping
from scriptorium.evals.prompts import Prompt

SEED = baseline.SEED
CV_SEED = 20261025
CIS_CV_CONFIG = {"clf": "linear_svm", "ngram_max": 1, "C": 10.0, "class_weight": "balanced"}  # the NIST baseline's, not tuned on CIS
_REF_RE = re.compile(r"\b(Safeguards?|Controls?)\s+\d{1,2}(?:\.\d{1,2})?", re.IGNORECASE)


class CisDataError(ValueError):
    pass


def strip_cis_refs(text: str) -> tuple[str, bool]:
    """Replace cross-references such as 'Safeguard 4.1' so a number cannot leak the label."""
    new = _REF_RE.sub(lambda m: f"[{m.group(1).lower().rstrip('s')}]", text)
    return new, new != text


def load_cis(store: MappingStore | None) -> tuple[dict[str, dict], dict[str, str]]:
    """Safeguards (id -> element with text and control) and Control titles (number -> title) from the local store."""
    if store is None:
        raise CisDataError("no crosswalk store; build it first (python -m scriptorium.crosswalks build ...)")
    safeguards, controls = {}, {}
    for (fw, eid), e in store.elements.items():
        if fw != "cis-8.1":
            continue
        meta = e.get("meta", {})
        if meta.get("level") == "control":
            controls[eid] = e["title"]
        elif meta.get("control"):
            safeguards[eid] = e
    if not safeguards or not all(e["text"] for e in safeguards.values()):
        raise CisDataError("CIS safeguard descriptions are missing; build the store with --cis-controls")
    if not controls:
        raise CisDataError("CIS Control titles are missing; build the store with --cis-controls")
    return safeguards, controls


def _nist_rows() -> list[dict[str, Any]]:
    return baseline.load_jsonl(ingest.PROCESSED_DIR / "train.jsonl") + baseline.load_jsonl(ingest.PROCESSED_DIR / "test.jsonl")


def _cis_pairs(store: MappingStore) -> list[dict]:
    return [m for m in store.mappings if m["from_framework"] == "cis-8.1" and m["to_framework"] == "nist-800-53r5" and m["source"] == "cis-nist-mapping"]


def _ctrl_key(c: str) -> tuple[int, str]:
    return (int(c), c) if c.isdigit() else (10**6, c)


# ---- E2 --------------------------------------------------------------------------------------------------


def build_e2(store: MappingStore | None, nist_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Items, acceptable-family sets and reference baselines for E2."""
    safeguards, _ = load_cis(store)
    nist = nist_rows if nist_rows is not None else _nist_rows()
    fam_of = {r["id"]: r["family"] for r in nist}
    acceptable: dict[str, set[str]] = {}
    for m in _cis_pairs(store):
        if m["to_id"] in fam_of and m["from_id"] in safeguards:
            acceptable.setdefault(m["from_id"], set()).add(fam_of[m["to_id"]])
    items, stripped = [], 0
    for sid in sorted(acceptable, key=lambda s: tuple(int(x) for x in s.split("."))):
        e = safeguards[sid]
        text, changed = strip_cis_refs(e["text"])
        stripped += changed
        items.append({"id": f"cis:{sid}", "base_id": e["meta"]["control"], "family": ",".join(sorted(acceptable[sid])), "title": "", "is_enhancement": False, "text": text})
    families = sorted({r["family"] for r in nist})
    sets = [acceptable[i["id"][4:]] for i in items]
    counts = Counter(f for s in sets for f in s)
    top = counts.most_common(1)[0][0]
    return {"items": items, "acceptable": {i["id"]: acceptable[i["id"][4:]] for i in items}, "references_stripped": stripped,
            "random_lenient": float(np.mean([len(s) / len(families) for s in sets])), "most_frequent_family": top,
            "most_frequent_lenient": float(np.mean([top in s for s in sets])), "n_single_family": sum(len(s) == 1 for s in sets)}


def _boot_clusters(clusters: list[str], fn, n: int = baseline.N_BOOTSTRAP, seed: int = SEED) -> dict[str, list[float]]:
    by: dict[str, list[int]] = {}
    for i, c in enumerate(clusters):
        by.setdefault(c, []).append(i)
    groups = [np.array(v) for v in by.values()]
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {}
    for _ in range(n):
        idx = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        for k, v in fn(idx).items():
            draws.setdefault(k, []).append(v)
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in draws.items()}


def score_e2(outcomes: list[mapping.Outcome], sk_pred: dict[str, str], acceptable: dict[str, set[str]], clusters: dict[str, str],
             n_boot: int = baseline.N_BOOTSTRAP) -> dict[str, Any]:
    """Lenient and strict accuracy for the LLM and sklearn, and the paired cluster-bootstrap difference (the registered H1)."""
    done = [o for o in outcomes if o.status != "not_run"]
    if not done:
        return {"n_scored": 0}
    A = [acceptable[o.id] for o in done]
    llm = np.array([(o.pred in a) if o.pred else False for o, a in zip(done, A)], dtype=float)
    sk = np.array([sk_pred[o.id] in a for o, a in zip(done, A)], dtype=float)
    single = np.array([len(a) == 1 for a in A])
    cl = [clusters[o.id] for o in done]
    diff = lambda idx: {"lenient_diff": float(llm[idx].mean() - sk[idx].mean())}
    point = float(llm.mean() - sk.mean())
    ci = _boot_clusters(cl, diff, n_boot)["lenient_diff"]
    strict = None
    if single.any():
        sc = [c for c, s in zip(cl, single) if s]
        l_s, k_s = llm[single], sk[single]
        sd = lambda idx: {"strict_diff": float(l_s[idx].mean() - k_s[idx].mean())}
        strict = {"n": int(single.sum()), "llm": float(l_s.mean()), "sklearn": float(k_s.mean()), "llm_minus_sklearn": float(l_s.mean() - k_s.mean()),
                  "llm_minus_sklearn_ci95": _boot_clusters(sc, sd, n_boot)["strict_diff"]}
    return {
        "n_scored": len(done), "n_clusters": len(set(cl)),
        "lenient": {"llm": float(llm.mean()), "sklearn": float(sk.mean()), "llm_minus_sklearn": point, "llm_minus_sklearn_ci95": ci,
                    "only_llm": int(((llm == 1) & (sk == 0)).sum()), "only_sklearn": int(((llm == 0) & (sk == 1)).sum()),
                    "both": int(((llm == 1) & (sk == 1)).sum()), "neither": int(((llm == 0) & (sk == 0)).sum())},
        "strict_single_family": strict,
        "h1_supported": bool(ci[0] > 0),  # the registered rule: lower bound of the 95% interval above 0
        "status_counts": dict(Counter(o.status for o in done)),
    }


# ---- E1 --------------------------------------------------------------------------------------------------


def build_e1(store: MappingStore | None) -> dict[str, Any]:
    safeguards, controls = load_cis(store)
    items, stripped = [], 0
    for sid in sorted(safeguards, key=lambda s: tuple(int(x) for x in s.split("."))):
        e = safeguards[sid]
        text, changed = strip_cis_refs(e["text"])
        stripped += changed
        items.append({"id": f"cis:{sid}", "base_id": f"cis:{sid}", "family": e["meta"]["control"], "title": "", "is_enhancement": False, "text": text})
    families = {c: controls[c] for c in sorted(controls, key=_ctrl_key)}
    return {"items": items, "families": families, "references_stripped": stripped}


def e1_system_prompt(prompt: Prompt, families: dict[str, str]) -> str:
    return prompt.render(family_list="\n".join(f"{c}: {t}" for c, t in sorted(families.items(), key=lambda kv: _ctrl_key(kv[0]))))


def sklearn_cv_predictions(items: list[dict[str, Any]], seed: int = CV_SEED, folds: int = 5) -> dict[str, str]:
    """Out-of-fold predictions: every item is predicted by a model that never saw it."""
    X = np.array([r["text"] for r in items], dtype=object)
    y = np.array([r["family"] for r in items])
    out: dict[str, str] = {}
    for tr, te in StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(X, y):
        model = baseline.build_pipeline(CIS_CV_CONFIG).fit(X[tr], y[tr])
        for i, p in zip(te, model.predict(X[te])):
            out[items[i]["id"]] = str(p)
    return out


# ---- E3 --------------------------------------------------------------------------------------------------


def run_e3(store: MappingStore | None, nist_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Retrieval in both directions against the official pairs, overall and for Equivalent pairs only."""
    safeguards, _ = load_cis(store)
    nist = nist_rows if nist_rows is not None else _nist_rows()
    nist_text = {r["id"]: r["text"] for r in nist}
    cis_q = [{"id": sid, "text": strip_cis_refs(e["text"])[0]} for sid, e in safeguards.items()]
    nist_q = [{"id": i, "text": t} for i, t in nist_text.items()]
    pairs = [m for m in _cis_pairs(store) if m["to_id"] in nist_text and m["from_id"] in safeguards]

    def truth(rows):
        fwd: dict[str, set[str]] = {}
        for m in rows:
            fwd.setdefault(m["from_id"], set()).add(m["to_id"])
        return fwd, crosswalk.invert(fwd)

    out = {}
    for name, rows in (("all_pairs", pairs), ("equivalent_only", [m for m in pairs if m["relationship"] == "equivalent"])):
        fwd, inv = truth(rows)
        out[name] = {
            "pairs": len(rows),
            "cis_to_nist": crosswalk.score(cis_q, nist_q, fwd) if fwd else {"n_queries": 0},
            "nist_to_cis": crosswalk.score(nist_q, cis_q, inv) if inv else {"n_queries": 0},
        }
    return out


def user_messages(items: list[dict[str, Any]]) -> list[str]:
    """The only text sent to a model: the description, quoted as untrusted data. No id, title, control or relationship."""
    return context.build_user_messages("none", items)
