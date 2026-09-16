from unittest.mock import MagicMock

import pytest

from app.rag.retry import with_retry


class _DailyQuotaError(Exception):
    def __init__(self):
        super().__init__(
            "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, "
            "'message': 'Quota exceeded', 'status': 'RESOURCE_EXHAUSTED', "
            "'details': [{'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'}]}}"
        )
        self.code = 429


class _TransientError(Exception):
    def __init__(self):
        super().__init__("503 UNAVAILABLE")
        self.code = 503


def test_daily_quota_error_fails_fast_without_retrying(monkeypatch):
    monkeypatch.setattr("app.rag.retry.time.sleep", MagicMock())
    calls = MagicMock(side_effect=_DailyQuotaError())

    @with_retry(max_attempts=5, base_delay=0.01)
    def flaky():
        return calls()

    with pytest.raises(_DailyQuotaError):
        flaky()

    assert calls.call_count == 1  # no retries attempted


def test_transient_error_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr("app.rag.retry.time.sleep", MagicMock())
    calls = MagicMock(side_effect=[_TransientError(), _TransientError(), "ok"])

    @with_retry(max_attempts=5, base_delay=0.01)
    def flaky():
        return calls()

    assert flaky() == "ok"
    assert calls.call_count == 3


def test_non_retryable_error_raises_immediately(monkeypatch):
    monkeypatch.setattr("app.rag.retry.time.sleep", MagicMock())
    calls = MagicMock(side_effect=ValueError("not an API error"))

    @with_retry(max_attempts=5, base_delay=0.01)
    def flaky():
        return calls()

    with pytest.raises(ValueError):
        flaky()

    assert calls.call_count == 1
