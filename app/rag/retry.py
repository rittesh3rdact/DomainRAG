"""Small retry helper for transient Gemini API errors (rate limits, 5xx)."""

import logging
import time
from functools import wraps
from typing import Callable, TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _is_daily_quota_exhausted(exc: Exception) -> bool:
    """A per-day quota 429 won't recover within a retry loop's timescale,
    unlike a per-minute rate limit or a transient overload -- so retrying it
    just wastes time before failing anyway."""
    message = str(exc)
    return "RESOURCE_EXHAUSTED" in message and "PerDay" in message


def with_retry(max_attempts: int = 5, base_delay: float = 1.0) -> Callable:
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001 - re-raised after retries
                    status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
                    last_exc = exc
                    if _is_daily_quota_exhausted(exc):
                        logger.error(
                            "%s hit a daily quota limit; won't recover from retrying. "
                            "Aborting immediately instead of wasting time.",
                            func.__name__,
                        )
                        raise
                    if status not in RETRYABLE_STATUS_CODES or attempt == max_attempts:
                        raise
                    delay = base_delay * (2 ** (attempt - 1))
                    logger.warning(
                        "%s failed (attempt %d/%d): %s. Retrying in %.1fs",
                        func.__name__,
                        attempt,
                        max_attempts,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
            raise last_exc  # pragma: no cover - unreachable

        return wrapper

    return decorator
