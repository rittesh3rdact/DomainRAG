from app.config import Settings


def _settings(**overrides) -> Settings:
    return Settings(google_api_key="test-key", **overrides)


def test_generation_models_prepends_primary():
    settings = _settings(
        generation_model="primary", generation_model_fallbacks="fallback-a,fallback-b"
    )
    assert settings.generation_models == ["primary", "fallback-a", "fallback-b"]


def test_generation_models_dedupes_primary_from_fallbacks():
    settings = _settings(
        generation_model="primary", generation_model_fallbacks="primary,fallback-a"
    )
    assert settings.generation_models == ["primary", "fallback-a"]


def test_generation_models_handles_empty_fallbacks():
    settings = _settings(generation_model="primary", generation_model_fallbacks="")
    assert settings.generation_models == ["primary"]


def test_utility_models_prepends_primary():
    settings = _settings(
        utility_model="primary", utility_model_fallbacks="fallback-a, fallback-b"
    )
    assert settings.utility_models == ["primary", "fallback-a", "fallback-b"]
