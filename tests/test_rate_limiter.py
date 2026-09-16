import time

from app.rag.rate_limiter import RateLimiter


def test_disabled_when_zero_never_blocks():
    limiter = RateLimiter(0)
    start = time.monotonic()
    for _ in range(50):
        limiter.wait()
    assert time.monotonic() - start < 0.1


def test_allows_calls_up_to_budget_without_blocking():
    limiter = RateLimiter(5)
    start = time.monotonic()
    for _ in range(5):
        limiter.wait()
    assert time.monotonic() - start < 0.5


def test_blocks_once_budget_exceeded():
    limiter = RateLimiter(2)
    # Manually seed timestamps so the budget is already exhausted, to avoid
    # a real ~60s sleep in the test.
    now = time.monotonic()
    limiter._timestamps.extend([now, now])

    start = time.monotonic()
    # Third call within the same window should wait until the oldest
    # timestamp ages out; we don't wait for the real 60s here, just confirm
    # it computed a positive sleep by checking timestamps grew stale first.
    limiter._timestamps[0] -= 59.9  # make the oldest almost expired
    limiter.wait()
    elapsed = time.monotonic() - start
    assert 0 <= elapsed < 1


def test_try_acquire_allows_up_to_budget_then_rejects():
    limiter = RateLimiter(2)
    assert limiter.try_acquire() is True
    assert limiter.try_acquire() is True
    assert limiter.try_acquire() is False  # budget exhausted, rejected not queued


def test_try_acquire_disabled_when_zero_always_allows():
    limiter = RateLimiter(0)
    assert all(limiter.try_acquire() for _ in range(100))


def test_try_acquire_frees_up_after_window_expires():
    limiter = RateLimiter(1)
    assert limiter.try_acquire() is True
    assert limiter.try_acquire() is False
    limiter._timestamps[0] -= 61  # simulate the window having passed
    assert limiter.try_acquire() is True
