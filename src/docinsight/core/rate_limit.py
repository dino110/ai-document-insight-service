"""Rate limiting: a sliding window of recent requests per key, kept in memory."""

import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Protocol


class RateLimiter(Protocol):
    def hit(self, key: str, limit: int) -> float | None:
        """Record a request for ``key``. None if it is allowed, otherwise the seconds until
        the next one would be (the request is then not recorded)."""
        ...


class InMemoryRateLimiter:
    """Timestamps of the requests in the last ``window_s`` seconds, per key.

    Memory only: limits reset on restart and are not shared between API instances; a
    Redis implementation can replace it behind the same protocol.
    """

    # Keys without a request in the last window are dropped every this many hits.
    SWEEP_EVERY = 1000

    def __init__(self, window_s: float = 60, clock: Callable[[], float] = time.monotonic) -> None:
        self._window_s = window_s
        self._clock = clock  # tests pass a fake clock
        self._hits: dict[str, deque[float]] = {}
        self._calls = 0
        # Sync endpoints run in a threadpool, so two requests can count at the same time.
        self._lock = threading.Lock()

    def hit(self, key: str, limit: int) -> float | None:
        now = self._clock()
        start = now - self._window_s
        with self._lock:
            self._calls += 1
            if self._calls % self.SWEEP_EVERY == 0:
                self._sweep(start)
            hits = self._hits.setdefault(key, deque())
            while hits and hits[0] <= start:
                hits.popleft()
            if len(hits) >= limit:
                return hits[0] - start  # when the oldest request leaves the window
            hits.append(now)
            return None

    def _sweep(self, start: float) -> None:
        for key in [k for k, hits in self._hits.items() if not hits or hits[-1] <= start]:
            del self._hits[key]
