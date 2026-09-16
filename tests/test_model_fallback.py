import pytest

from app.rag.model_fallback import ModelFallback


def test_uses_primary_model_when_it_succeeds():
    fallback = ModelFallback(["primary", "backup"])
    calls = []

    def fn(model):
        calls.append(model)
        return "ok"

    assert fallback.call(fn) == "ok"
    assert calls == ["primary"]


def test_falls_back_to_next_model_on_failure():
    fallback = ModelFallback(["primary", "backup"])
    calls = []

    def fn(model):
        calls.append(model)
        if model == "primary":
            raise RuntimeError("exhausted")
        return "ok from backup"

    assert fallback.call(fn) == "ok from backup"
    assert calls == ["primary", "backup"]


def test_prefers_last_successful_model_next_time():
    fallback = ModelFallback(["primary", "backup"])

    def fails_primary(model):
        if model == "primary":
            raise RuntimeError("exhausted")
        return "ok"

    fallback.call(fails_primary)  # falls back to "backup", remembers it

    calls = []

    def fn(model):
        calls.append(model)
        return "ok"

    fallback.call(fn)
    assert calls == ["backup"]  # tried first this time, no wasted attempt on primary


def test_raises_last_error_when_all_models_fail():
    fallback = ModelFallback(["a", "b"])

    def always_fails(model):
        raise RuntimeError(f"{model} failed")

    with pytest.raises(RuntimeError, match="b failed"):
        fallback.call(always_fails)


def test_empty_model_list_rejected():
    with pytest.raises(ValueError):
        ModelFallback([])
