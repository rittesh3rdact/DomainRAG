"""Proactive client-side rate limiting.

Free-tier Gemini quotas enforce a requests-per-minute cap per model. Firing
calls in a tight loop (e.g. one contextualization call per chunk during
ingestion) bursts past that cap, causing a wave of 429s that then have to
be recovered with exponential backoff -- slow and noisy. Pacing calls to
stay under the limit up front avoids hitting 429s in the first place.
"""

import threading
import time
from collections import deque


class RateLimiter:
    def __init__(self, max_per_minute: int):
        self._max_per_minute = max_per_minute
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def wait(self) -> None:
        """Block, if necessary, until another call is allowed under the budget."""
        if self._max_per_minute <= 0:
            return

        with self._lock:
            self._drop_stale()
            if len(self._timestamps) >= self._max_per_minute:
                sleep_for = 60 - (time.monotonic() - self._timestamps[0])
                if sleep_for > 0:
                    time.sleep(sleep_for)
                self._drop_stale()
            self._timestamps.append(time.monotonic())

    def try_acquire(self) -> bool:
        """Non-blocking variant: returns False instead of waiting if the
        budget is exhausted. Used to reject excess incoming requests (e.g.
        at the API layer) rather than queueing them, which would just delay
        an eventual quota-exhaustion error instead of preventing it."""
        if self._max_per_minute <= 0:
            return True

        with self._lock:
            self._drop_stale()
            if len(self._timestamps) >= self._max_per_minute:
                return False
            self._timestamps.append(time.monotonic())
            return True

    def _drop_stale(self) -> None:
        now = time.monotonic()
        while self._timestamps and now - self._timestamps[0] >= 60:
            self._timestamps.popleft()
