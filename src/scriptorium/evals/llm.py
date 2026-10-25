"""One request/response shape for every model call, a real Anthropic client, and a cache wrapper.

Request parameters follow the current Claude 5 API: structured output through
`output_config.format`, effort through `output_config.effort`, thinking left adaptive or set to
`between_tools` (the thinking-off setting on Sonnet 5.5). Sampling parameters, forced tool choice
and assistant prefill are rejected by these models, so none is ever sent.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from scriptorium.evals.budget import Budget, cost_usd

EFFORTS = ("low", "medium", "high")
THINKING_MODES = ("adaptive", "off")


class LLMError(RuntimeError):
    """A model call failed. The message is safe to print (it never contains credentials or request text)."""


class FatalLLMError(LLMError):
    """A failure that would affect every request (bad credentials, unknown model, rejected request shape). Stops a run."""


class CacheMiss(LLMError):
    pass


@dataclass(frozen=True)
class LLMRequest:
    model: str
    system: str
    user: str
    schema: dict[str, Any]
    max_tokens: int = 1024
    effort: str = "low"
    thinking: str = "adaptive"  # "adaptive" (default behaviour) or "off" (between_tools)

    def __post_init__(self) -> None:
        if self.effort not in EFFORTS:
            raise ValueError(f"effort must be one of {EFFORTS}")
        if self.thinking not in THINKING_MODES:
            raise ValueError(f"thinking must be one of {THINKING_MODES}")
        if not 16 <= self.max_tokens <= 64_000:
            raise ValueError("max_tokens must be between 16 and 64000")

    def params(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LLMResponse:
    text: str | None
    stop_reason: str | None
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    request_id: str | None = None
    from_cache: bool = field(default=False, compare=False)

    def cost(self) -> float:
        return cost_usd(self.model, self.input_tokens, self.output_tokens, self.cache_read_tokens, self.cache_write_tokens)


class LLMClient(Protocol):
    def complete(self, req: LLMRequest) -> LLMResponse: ...


_WORKSPACE_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_KEY_RE = re.compile(r"sk-[A-Za-z0-9_\-]{6,}")


def _safe(message: object, limit: int = 300) -> str:
    """API error text made safe to print: key-shaped strings removed, length capped."""
    return _KEY_RE.sub("[redacted]", str(message))[:limit]


def build_messages_kwargs(req: LLMRequest) -> dict[str, Any]:
    """The exact `messages.create` arguments for a request. Kept separate so tests can assert on it."""
    kwargs: dict[str, Any] = {
        "model": req.model,
        "max_tokens": req.max_tokens,
        "system": [{"type": "text", "text": req.system, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": req.user}],
        "output_config": {"effort": req.effort, "format": {"type": "json_schema", "schema": req.schema}},
    }
    if req.thinking == "off":
        kwargs["thinking"] = {"type": "between_tools"}
    return kwargs


class AnthropicClient:
    """Live client. Credentials come from the environment or the SDK's own profile; nothing is stored here."""

    def __init__(self, client: Any | None = None, workspace_id: str | None = None):
        if client is None:
            import anthropic

            client = anthropic.Anthropic(max_retries=3, timeout=120.0)
        self._client = client
        # A key that is not scoped to one workspace needs the workspace named on every request.
        wid = workspace_id or os.environ.get("ANTHROPIC_WORKSPACE_ID") or None
        if wid is not None and not _WORKSPACE_RE.match(wid):
            raise ValueError("workspace id must be 1-64 characters: letters, digits, dash, underscore")
        self.workspace_id = wid

    def complete(self, req: LLMRequest) -> LLMResponse:
        import anthropic

        no_credentials = FatalLLMError("no credentials found: set ANTHROPIC_API_KEY in your environment (or log in with the Anthropic CLI) and retry")
        try:
            kwargs = build_messages_kwargs(req)
            if self.workspace_id:
                kwargs["workspace_id"] = self.workspace_id  # sent as the anthropic-workspace-id header
            r = self._client.messages.create(**kwargs)
        except anthropic.CredentialsError:
            raise no_credentials from None
        except TypeError as e:  # the SDK raises this before any network call when nothing is configured
            if "authentication method" in str(e):
                raise no_credentials from None
            raise
        except anthropic.AuthenticationError:
            raise FatalLLMError("authentication failed: set ANTHROPIC_API_KEY in your environment (or log in with the Anthropic CLI) and retry") from None
        except anthropic.PermissionDeniedError:
            raise FatalLLMError("the credentials are not permitted to use this model") from None
        except anthropic.NotFoundError:
            raise FatalLLMError(f"model {req.model!r} was not found") from None
        except anthropic.BadRequestError as e:
            raise FatalLLMError(f"the API rejected the request: {_safe(getattr(e, 'message', 'bad request'))}") from None
        except anthropic.RateLimitError:
            raise LLMError("rate limited after retries; try again later or lower concurrency") from None
        except anthropic.APIConnectionError:
            raise LLMError("could not reach the API (network error)") from None
        except anthropic.APIStatusError as e:
            raise LLMError(f"API error (status {e.status_code})") from None
        text = next((b.text for b in r.content if getattr(b, "type", None) == "text"), None)
        u = r.usage
        return LLMResponse(
            text=text,
            stop_reason=r.stop_reason,
            model=r.model,
            input_tokens=int(getattr(u, "input_tokens", 0) or 0),
            output_tokens=int(getattr(u, "output_tokens", 0) or 0),
            cache_read_tokens=int(getattr(u, "cache_read_input_tokens", 0) or 0),
            cache_write_tokens=int(getattr(u, "cache_creation_input_tokens", 0) or 0),
            request_id=getattr(r, "_request_id", None),
        )


class LazyClient:
    """Builds the real client only if a call actually reaches it, so a fully cached run needs no credentials."""

    def __init__(self, factory):
        self._factory, self._client = factory, None

    def complete(self, req: LLMRequest) -> LLMResponse:
        if self._client is None:
            self._client = self._factory()
        return self._client.complete(req)


# ---- response cache ----------------------------------------------------------------------------


def cache_key(req: LLMRequest, task: str, prompt_version: str, prompt_sha256: str) -> str:
    """Hash of everything that determines the answer: model, prompt version and hash, parameters and input."""
    body = json.dumps(
        {"task": task, "prompt_version": prompt_version, "prompt_sha256": prompt_sha256, "request": req.params()},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._ready = False

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.path.parent, 0o700)
        if not self.path.exists():
            os.close(os.open(self.path, os.O_WRONLY | os.O_CREAT, 0o600))
        os.chmod(self.path, 0o600)
        conn = sqlite3.connect(self.path, timeout=30)
        conn.execute("PRAGMA busy_timeout=30000")
        if not self._ready:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, created TEXT NOT NULL, task TEXT NOT NULL, "
                "model TEXT NOT NULL, prompt_version TEXT NOT NULL, response TEXT NOT NULL)"
            )
            self._ready = True
        return conn

    def get(self, key: str) -> LLMResponse | None:
        if not self.path.exists():
            return None
        conn = self._connect()
        try:
            row = conn.execute("SELECT response FROM responses WHERE key=?", (key,)).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        data = json.loads(row[0])
        data.pop("from_cache", None)
        return LLMResponse(**data, from_cache=True)

    def put(self, key: str, task: str, prompt_version: str, resp: LLMResponse) -> None:
        data = asdict(resp)
        data.pop("from_cache", None)
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO responses (key, created, task, model, prompt_version, response) VALUES (?,?,?,?,?,?)",
                    (key, datetime.now(timezone.utc).isoformat(timespec="seconds"), task, resp.model, prompt_version, json.dumps(data)),
                )
                conn.commit()
            finally:
                conn.close()


class CachedClient:
    """Cache first. In replay mode a miss is an error and no network call is ever made."""

    def __init__(self, inner: LLMClient | None, cache: ResponseCache, task: str, prompt_version: str, prompt_sha256: str,
                 budget: Budget | None = None, replay: bool = False, refresh: bool = False):
        if replay and refresh:
            raise ValueError("replay and refresh cannot be combined")
        if inner is None and not replay:
            raise ValueError("a live client is required unless replaying")
        self.inner, self.cache, self.task = inner, cache, task
        self.prompt_version, self.prompt_sha256 = prompt_version, prompt_sha256
        self.budget, self.replay, self.refresh = budget, replay, refresh
        self.hits = self.calls = 0
        self._lock = threading.Lock()

    def complete(self, req: LLMRequest) -> LLMResponse:
        key = cache_key(req, self.task, self.prompt_version, self.prompt_sha256)
        if not self.refresh:
            hit = self.cache.get(key)
            if hit is not None:
                with self._lock:
                    self.hits += 1
                return hit
        if self.replay:
            raise CacheMiss("replay mode: no cached response for this request")
        if self.budget is not None:
            self.budget.check()
        resp = self.inner.complete(req)  # type: ignore[union-attr]
        self.cache.put(key, self.task, self.prompt_version, resp)
        with self._lock:
            self.calls += 1
        if self.budget is not None:
            self.budget.add(resp.cost())
        return resp
