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
