"""Read-only service layer: classify a requirement and map it across catalogs.

This is the logic behind the MCP server (and any future CLI or HTTP wrapper). It uses
only the scikit-learn models and data already built by `ingest`, `scf` and `baseline`.

Design constraints (see README, "Security"):
  * No tool argument is ever used as a file path. Catalogs are chosen from a fixed
    allowlist built at start-up; control IDs and free text are validated and length-capped.
  * Queries never write to disk or train anything. The only thing that changes a model is
    an admin applying approved corrections (see `scriptorium.feedback`); the model is then
    refit from the catalog's training split plus those corrections.
  * Every response carries a disclaimer: results are suggestions that need human review.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from scriptorium import baseline, crosswalk, ingest, paths
from scriptorium.feedback import FeedbackStore, normalize_text

MAX_TEXT_CHARS = 4000
MAX_TOP_K = 10
SNIPPET_CHARS = 300
DISCLAIMER = (
    "Suggestions from a TF-IDF/linear-model baseline, not authoritative mappings. "
    "Similarity is not proof of coverage. Review before relying on any result."
)
DEFAULT_CONFIG = {"clf": "linear_svm", "ngram_max": 1, "C": 1.0, "class_weight": None}

_CATALOG_NAME_RE = re.compile(r"^[a-z0-9_-]{1,32}$")
_CONTROL_ID_RE = re.compile(r"^[A-Za-z0-9().\-]{1,24}$")


class ServiceError(ValueError):
    """A request was invalid. The message is safe to show to the caller."""


@dataclass(frozen=True)
class CatalogSource:
    name: str
    data_dir: Path  # contains train.jsonl and test.jsonl
    results_path: Path | None = None  # baseline_metrics.json with the selected config
    nist_mappings_path: Path | None = None  # SCF-style raw NIST mapping file, if any


@dataclass
class _Model:
    pipeline: Pipeline
    version: str
    config: dict[str, Any]
    low_conf_margin: float
    oof: dict[str, Any]
    titles: dict[str, str]
    corrections: dict[str, dict[str, Any]]  # normalized text -> {"id", "label"} for applied corrections
    key: tuple = ()


def _sha(*parts: bytes) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.hexdigest()


def _margin_and_correct(pipeline: Pipeline, rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    scores = pipeline.decision_function([r["text"] for r in rows])
    order = np.argsort(-scores, axis=1)
    top = scores[np.arange(len(rows)), order[:, 0]]
    second = scores[np.arange(len(rows)), order[:, 1]]
    pred = pipeline.classes_[order[:, 0]]
    return top - second, pred == np.array([r["family"] for r in rows])


class Service:
    def __init__(self, sources: dict[str, CatalogSource], feedback: FeedbackStore | None = None):
        for name in sources:
            if not _CATALOG_NAME_RE.match(name):
                raise ValueError(f"invalid catalog name: {name!r}")
        self._sources = dict(sources)
        self.feedback = feedback
        self._rows: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
        self._models: dict[str, _Model] = {}
        self._calibration: dict[str, tuple[float, dict[str, Any]]] = {}
        self._index: tuple[TfidfVectorizer, dict[str, Any]] | None = None
        self._truth: dict[str, dict[str, set[str]]] | None = None

    @classmethod
    def default(cls) -> "Service":
        """NIST from the repo's committed data, plus SCF if the user built it locally."""
        sources = {
            "nist": CatalogSource("nist", ingest.PROCESSED_DIR, ingest.ROOT / "results" / "baseline_metrics.json")
        }
        scf_dir = paths.scf_dir()
        if (scf_dir / "manifest.json").exists() and (scf_dir / "train.jsonl").exists():
            sources["scf"] = CatalogSource(
                "scf", scf_dir, scf_dir / "results" / "baseline_metrics.json", scf_dir / "nist_r5_2_mappings.json"
            )
        return cls(sources, FeedbackStore(paths.scriptorium_home() / "feedback"))

    # ---- validation -------------------------------------------------------------------

    def _catalog(self, name: Any) -> CatalogSource:
        if not isinstance(name, str) or not _CATALOG_NAME_RE.match(name) or name not in self._sources:
            raise ServiceError(f"unknown catalog; available: {sorted(self._sources)}")
        return self._sources[name]

    @staticmethod
    def _text(text: Any) -> str:
        if not isinstance(text, str):
            raise ServiceError("text must be a string")
        text = text.strip()
        if not text:
            raise ServiceError("text must not be empty")
        if len(text) > MAX_TEXT_CHARS:
            raise ServiceError(f"text is too long (max {MAX_TEXT_CHARS} characters)")
        if "\x00" in text:
            raise ServiceError("text contains a null byte")
        return text

    @staticmethod
    def _top_k(k: Any) -> int:
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= MAX_TOP_K:
            raise ServiceError(f"top_k must be an integer from 1 to {MAX_TOP_K}")
        return k

    @staticmethod
    def _control_id(cid: Any) -> str:
        if not isinstance(cid, str) or not _CONTROL_ID_RE.match(cid):
            raise ServiceError("control_id has an invalid format")
        return cid.upper()

    # ---- data and models --------------------------------------------------------------

    def _load(self, name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if name not in self._rows:
            d = self._sources[name].data_dir
            self._rows[name] = (baseline.load_jsonl(d / "train.jsonl"), baseline.load_jsonl(d / "test.jsonl"))
        return self._rows[name]

    def _config(self, src: CatalogSource) -> dict[str, Any]:
        if src.results_path and src.results_path.exists():
            return json.loads(src.results_path.read_text(encoding="utf-8"))["selected_config"]
        return dict(DEFAULT_CONFIG)

    def labels(self, name: str) -> set[str]:
        return {r["family"] for r in self._load(name)[0]}

    def fit(self, name: str, corrections: list[dict[str, Any]]) -> Pipeline:
        """Fit the catalog's model on its training split plus weighted corrections."""
        train, _ = self._load(name)
        X = [r["text"] for r in train] + [c["text"] for c in corrections]
        y = [r["family"] for r in train] + [c["label"] for c in corrections]
        w = [1.0] * len(train) + [float(c["weight"]) for c in corrections]
        return baseline.build_pipeline(self._config(self._sources[name])).fit(X, y, clf__sample_weight=w)

    def heldout_scores(self, name: str, pipeline: Pipeline) -> dict[str, float]:
        """Scores on the frozen test split. Used by the apply gate, never for training."""
        _, test = self._load(name)
        y = [r["family"] for r in test]
        pred = pipeline.predict([r["text"] for r in test])
        return {
            "accuracy": round(float(accuracy_score(y, pred)), 4),
            "macro_f1": round(float(f1_score(y, pred, average="macro", zero_division=0)), 4),
        }

    def _calibrate(self, name: str) -> tuple[float, dict[str, Any]]:
        """Low-confidence threshold from out-of-fold predictions on the training split only:
        the median score margin among wrong predictions. Corrections are not included."""
        if name not in self._calibration:
            train, _ = self._load(name)
            cfg = self._config(self._sources[name])
            margins, correct = [], []
            for tr, va in baseline.grouped_folds(train):
                fold = baseline.build_pipeline(cfg).fit([train[i]["text"] for i in tr], [train[i]["family"] for i in tr])
                m, c = _margin_and_correct(fold, [train[i] for i in va])
                margins.append(m)
                correct.append(c)
            margins, correct = np.concatenate(margins), np.concatenate(correct)
            wrong = margins[~correct]
            threshold = float(np.median(wrong)) if len(wrong) else 0.0
            low = margins < threshold
            self._calibration[name] = (
                threshold,
                {
                    "threshold_margin": threshold,
                    "oof_accuracy": float(correct.mean()),
                    "oof_accuracy_if_confident": float(correct[~low].mean()) if (~low).any() else None,
                    "oof_accuracy_if_low_confidence": float(correct[low].mean()) if low.any() else None,
                },
            )
        return self._calibration[name]

    def _model(self, name: str) -> _Model:
        """The active model: training split plus applied corrections. Refit when those change."""
        corrections = sorted(self.feedback.applied(name), key=lambda e: e["id"]) if self.feedback else []
        key = tuple((c["id"], c["weight"]) for c in corrections)
        cached = self._models.get(name)
        if cached is not None and cached.key == key:
            return cached
        src = self._sources[name]
        train, _ = self._load(name)
        cfg = self._config(src)
        pipe = self.fit(name, corrections)
        threshold, oof = self._calibrate(name)
        version = _sha(
            json.dumps(cfg, sort_keys=True).encode(),
            *(json.dumps(r, sort_keys=True).encode() for r in train),
            *(json.dumps([c["id"], c["text"], c["label"], c["weight"]]).encode() for c in corrections),
        )[:12]
        titles = {r["family"]: r.get("family_title", r["family"]) for r in train}
        lookup = {normalize_text(c["text"]): {"id": c["id"], "label": c["label"]} for c in corrections}
        self._models[name] = _Model(pipe, version, cfg, threshold, oof, titles, lookup, key)
        return self._models[name]

    def _retrieval_index(self) -> tuple[TfidfVectorizer, dict[str, Any]]:
        """One vectorizer fitted on the text of every available catalog (no labels or mappings)."""
        if self._index is None:
            recs = {n: self._load(n)[0] + self._load(n)[1] for n in self._sources}
            vec = crosswalk.make_vectorizer().fit([r["text"] for rows in recs.values() for r in rows])
            self._index = (vec, {n: (rows, vec.transform([r["text"] for r in rows])) for n, rows in recs.items()})
        return self._index

    def _published(self) -> dict[str, dict[str, set[str]]]:
        """Published SCF->NIST mapping and its inverse, when both catalogs are available."""
        if self._truth is None:
            self._truth = {}
            scf = self._sources.get("scf")
            if scf and scf.nist_mappings_path and scf.nist_mappings_path.exists() and "nist" in self._sources:
                nist_ids = {r["id"] for rows in self._load("nist") for r in rows}
                raw = json.loads(scf.nist_mappings_path.read_text(encoding="utf-8"))
                fwd, _ = crosswalk.normalized_mapping(raw, nist_ids)
                self._truth = {"scf->nist": fwd, "nist->scf": crosswalk.invert(fwd)}
        return self._truth

    # ---- public tools -----------------------------------------------------------------

    def list_catalogs(self) -> dict[str, Any]:
        out = []
        for name, src in self._sources.items():
            train, test = self._load(name)
            m = self._model(name)
            out.append(
                {
                    "name": name,
                    "records": len(train) + len(test),
                    "labels": sorted(m.titles),
                    "model_version": m.version,
                    "model": m.config,
                    "confidence_calibration": m.oof,
                }
            )
        return {"catalogs": out, "disclaimer": DISCLAIMER}

    def classify(self, text: Any, catalog: Any = "nist", top_k: Any = 3) -> dict[str, Any]:
        text, k = self._text(text), self._top_k(top_k)
        src = self._catalog(catalog)
        m = self._model(src.name)
        scores = m.pipeline.decision_function([text])[0]
        order = np.argsort(-scores)
        margin = float(scores[order[0]] - scores[order[1]])
        return {
            "catalog": src.name,
            "model_version": m.version,
            "predictions": [
                {"label": str(m.pipeline.classes_[i]), "label_title": m.titles.get(str(m.pipeline.classes_[i])), "score": round(float(scores[i]), 4)}
                for i in order[:k]
            ],
            "margin": round(margin, 4),
            "low_confidence": margin < m.low_conf_margin,
            # Exact-text match with an applied correction: shown alongside, never hidden.
            "applied_correction": m.corrections.get(normalize_text(text)),
            "disclaimer": DISCLAIMER,
        }

    def _nearest(self, query_vec, to: str, k: int, exclude_id: str | None = None) -> list[dict[str, Any]]:
        rows, mat = self._retrieval_index()[1][to]
        sims = (query_vec @ mat.T).toarray()[0]
        out = []
        for i in np.argsort(-sims, kind="stable"):
            r = rows[i]
            if exclude_id and r["id"] == exclude_id:
                continue
            out.append(
                {
                    "id": r["id"],
                    "title": r["title"],
                    "family": r["family"],
                    "score": round(float(sims[i]), 4),
                    "text": r["text"][:SNIPPET_CHARS],
                }
            )
            if len(out) == k:
                break
        return out

    def map_requirement(self, text: Any, to_catalog: Any, top_k: Any = 5) -> dict[str, Any]:
        text, k = self._text(text), self._top_k(top_k)
        dst = self._catalog(to_catalog)
        vec, _ = self._retrieval_index()
        return {
            "to_catalog": dst.name,
            "matches": self._nearest(vec.transform([text]), dst.name, k),
            "method": "TF-IDF cosine similarity",
            "disclaimer": DISCLAIMER,
        }

    def map_control(self, control_id: Any, from_catalog: Any, to_catalog: Any, top_k: Any = 5) -> dict[str, Any]:
        cid, k = self._control_id(control_id), self._top_k(top_k)
        src, dst = self._catalog(from_catalog), self._catalog(to_catalog)
        train, test = self._load(src.name)
        rec = next((r for r in train + test if r["id"].upper() == cid), None)
        if rec is None:
            raise ServiceError(f"control {cid} not found in catalog {src.name}")
        vec, _ = self._retrieval_index()
        published = self._published().get(f"{src.name}->{dst.name}")
        return {
            "control": {"id": rec["id"], "title": rec["title"], "family": rec["family"], "catalog": src.name},
            "to_catalog": dst.name,
            "published_mapping": sorted(published.get(rec["id"], [])) if published is not None else None,
            "published_mapping_source": "NIST SP 800-53 Rev 5.2 mappings in the user's SCF catalog" if published is not None else None,
            "similarity_matches": self._nearest(vec.transform([rec["text"]]), dst.name, k, exclude_id=rec["id"] if src is dst else None),
            "method": "TF-IDF cosine similarity",
            "disclaimer": DISCLAIMER,
        }
