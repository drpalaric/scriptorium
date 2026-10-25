"""Small in-process sliding-window rate limiter."""

from __future__ import annotations

import time
from collections import deque


class RateLimited(Exception):
    pass


class RateLimiter:
    def __init__(self, limit: int, window: float = 60.0, clock=time.monotonic):
        self.limit, self.window, self.clock = limit, window, clock
        self.hits: deque[float] = deque()

    def allow(self) -> bool:
        """Record a hit and return True if under the limit; return False (recording nothing) otherwise."""
        now = self.clock()
        while self.hits and now - self.hits[0] > self.window:
            self.hits.popleft()
        if len(self.hits) >= self.limit:
            return False
        self.hits.append(now)
        return True

    def check(self) -> None:
        if not self.allow():
            raise RateLimited(f"too many submissions; limit is {self.limit} per minute")
