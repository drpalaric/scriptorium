"""Prices and a hard spend cap.

Prices are USD per million tokens as listed in Anthropic's published pricing, copied here from
the claude-api reference dated 2026-10-06. They can go stale: check
https://platform.claude.com/docs/en/about-claude/pricing before relying on a projected cost.
Cache-write rates marked derived use the documented 1.25x-input multiplier. A model that is
not listed is refused rather than guessed at, so a run can never spend unpriced money.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

PRICES_AS_OF = "2026-10-06"
# Output tokens per call used in projections. "low" is measured (about 11 tokens per answer across earlier runs, so 40 leaves
# margin); "medium" and "high" are unmeasured guesses for answers that include some thinking.
EST_OUTPUT_BY_EFFORT = {"low": 40, "medium": 300, "high": 800}


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cache_read: float
    cache_write: float  # 5-minute cache writes


PRICES: dict[str, Price] = {
    "claude-sonnet-5-5": Price(2.00, 10.00, 0.20, 2.50),
    "claude-opus-5-5": Price(4.00, 20.00, 0.20, 5.00),  # cache write derived (1.25x input)
    "claude-haiku-5-5": Price(0.10, 0.50, 0.01, 0.125),  # prompts up to 100K tokens; cache rates derived
}


class BudgetExceeded(RuntimeError):
    pass


def price_for(model: str) -> Price:
    if model not in PRICES:
        raise ValueError(f"no price on file for model {model!r}; known models: {sorted(PRICES)}")
    return PRICES[model]


def cost_usd(model: str, input_tokens: int, output_tokens: int, cache_read: int = 0, cache_write: int = 0) -> float:
    p = price_for(model)
    return (input_tokens * p.input + output_tokens * p.output + cache_read * p.cache_read + cache_write * p.cache_write) / 1_000_000


def estimate_tokens_offline(text: str) -> int:
    """A rough count (about 4 characters per token) used only when no API is available. Always labelled an estimate."""
    return max(1, len(text) // 4)


def projected_cost(model: str, requests: list, n_calls: int, est_output_tokens: int | None = None, schema_overhead_tokens: int = 120) -> tuple[float, int]:
    """Offline projection for `n_calls` uncached requests, returning (usd, estimated input tokens per call).

    The system prompt is written to the prompt cache once and then read at the cache-read rate; only the per-item
    text, the schema and the output are charged in full. Still a rough character-based estimate, not a quote.
    """
    if not requests or n_calls <= 0:
        return 0.0, 0
    p = price_for(model)
    if est_output_tokens is None:
        est_output_tokens = EST_OUTPUT_BY_EFFORT.get(getattr(requests[0], "effort", "low"), 300)
    system_tokens = estimate_tokens_offline(requests[0].system)
    user_tokens = int(sum(estimate_tokens_offline(q.user) for q in requests) / len(requests)) + schema_overhead_tokens
    per_call = user_tokens * p.input + system_tokens * p.cache_read + est_output_tokens * p.output
    return (n_calls * per_call + system_tokens * p.cache_write) / 1_000_000, user_tokens + system_tokens


class Budget:
    """Thread-safe running total with a hard stop."""

    def __init__(self, limit_usd: float):
        if not 0 < limit_usd <= 10_000:
            raise ValueError("budget must be greater than 0 and at most 10,000 USD")
        self.limit = float(limit_usd)
        self.spent = 0.0
        self._lock = threading.Lock()

    def check(self) -> None:
        with self._lock:
            if self.spent >= self.limit:
                raise BudgetExceeded(f"spend cap reached (${self.spent:.4f} of ${self.limit:.2f})")

    def add(self, amount: float) -> None:
        with self._lock:
            self.spent += amount
