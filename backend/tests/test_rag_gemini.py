"""GeminiProvider unit tests - fully mocked, never touching the real API.

Covers: factory wiring, missing/invalid API key handling, network and
service errors, empty responses, availability probing, prompt/config
propagation, and a guarantee that the secret key is never leaked through
error messages or reprs.
"""

from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors as genai_errors

from app.core.config import get_settings
from app.rag.llm.base import (
    LLMResponseError,
    LLMUnavailableError,
    create_provider,
)
from app.rag.llm.gemini import GeminiProvider

SECRET = "test-secret-key-do-not-leak"


class _FakeModels:
    """Stands in for ``client.models``; records calls, raises on demand."""

    def __init__(self, text="```sql\nSELECT 1\n```", error=None):
        self.text = text
        self.error = error
        self.generate_calls: list[dict] = []
        self.get_calls: list[str] = []

    def generate_content(self, *, model, contents, config):
        self.generate_calls.append(
            {"model": model, "contents": contents, "config": config}
        )
        if self.error is not None:
            raise self.error
        return SimpleNamespace(text=self.text)

    def get(self, *, model):
        self.get_calls.append(model)
        if isinstance(self.error, genai_errors.APIError):
            raise self.error
        return SimpleNamespace(name=model)


def _provider(monkeypatch, models=None, api_key=SECRET, **kwargs) -> GeminiProvider:
    provider = GeminiProvider(model="gemini-test", api_key=api_key, **kwargs)
    fake_models = models if models is not None else _FakeModels()
    monkeypatch.setattr(provider, "_client", lambda: SimpleNamespace(models=fake_models))
    return provider


# ---------------------------------------------------------------------------
# Factory / configuration wiring
# ---------------------------------------------------------------------------
def test_factory_creates_gemini_provider():
    provider = create_provider("gemini", "gemini-2.5-flash")
    assert isinstance(provider, GeminiProvider)
    assert provider.name == "gemini"
    assert provider.model == "gemini-2.5-flash"


@pytest.fixture()
def secret_key_env(monkeypatch):
    """Set a fake key for Settings, then fully restore the cached settings."""
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    get_settings.cache_clear()
    yield
    monkeypatch.undo()
    get_settings.cache_clear()


def test_provider_reads_model_and_key_from_settings(secret_key_env):
    settings = get_settings()
    assert settings.gemini_api_key.get_secret_value() == SECRET
    provider = GeminiProvider(model=settings.active_llm_model)
    assert provider.api_key == SECRET
    assert provider.model == settings.gemini_model


def test_settings_default_provider_is_gemini():
    assert get_settings().llm_provider == "gemini"
    assert get_settings().active_llm_model == get_settings().gemini_model


# ---------------------------------------------------------------------------
# Missing / invalid key handling
# ---------------------------------------------------------------------------
def test_missing_key_raises_unavailable_on_generate(monkeypatch):
    provider = GeminiProvider(model="gemini-test", api_key="")
    with pytest.raises(LLMUnavailableError, match="GEMINI_API_KEY"):
        provider.generate("prompt")


def test_missing_key_means_not_available():
    provider = GeminiProvider(model="gemini-test", api_key="")
    assert provider.is_available() is False


def test_invalid_api_key_maps_to_unavailable(monkeypatch):
    err = genai_errors.ClientError(400, {"error": {"message": "API key not valid"}})
    provider = _provider(monkeypatch, _FakeModels(error=err))
    with pytest.raises(LLMUnavailableError, match="API key"):
        provider.generate("prompt")


def test_forbidden_key_maps_to_unavailable(monkeypatch):
    err = genai_errors.ClientError(403, {"error": {"message": "Permission denied"}})
    provider = _provider(monkeypatch, _FakeModels(error=err))
    with pytest.raises(LLMUnavailableError):
        provider.generate("prompt")


def test_other_client_errors_map_to_response_error(monkeypatch):
    err = genai_errors.ClientError(429, {"error": {"message": "quota"}})
    provider = _provider(monkeypatch, _FakeModels(error=err))
    with pytest.raises(LLMResponseError, match="429"):
        provider.generate("prompt")


# ---------------------------------------------------------------------------
# Network / service failures
# ---------------------------------------------------------------------------
def test_network_error_maps_to_unavailable(monkeypatch):
    err = httpx.ConnectError("connection refused")
    provider = _provider(monkeypatch, _FakeModels(error=err))
    with pytest.raises(LLMUnavailableError, match="unreachable"):
        provider.generate("prompt")


def test_server_error_maps_to_unavailable(monkeypatch):
    err = genai_errors.ServerError(503, {"error": {"message": "overloaded"}})
    provider = _provider(monkeypatch, _FakeModels(error=err))
    with pytest.raises(LLMUnavailableError, match="503"):
        provider.generate("prompt")


def test_unknown_model_is_not_available(monkeypatch):
    err = genai_errors.ClientError(404, {"error": {"message": "not found"}})
    provider = _provider(monkeypatch, _FakeModels(error=err))
    assert provider.is_available() is False


def test_valid_model_is_available(monkeypatch):
    provider = _provider(monkeypatch, _FakeModels())
    assert provider.is_available() is True


# ---------------------------------------------------------------------------
# Response handling + prompt/config propagation
# ---------------------------------------------------------------------------
def test_generate_returns_stripped_text(monkeypatch):
    provider = _provider(monkeypatch, _FakeModels(text="  SELECT 1  "))
    assert provider.generate("p") == "SELECT 1"


def test_generate_sends_system_instruction_and_determinism(monkeypatch):
    models = _FakeModels()
    provider = _provider(monkeypatch, models)
    provider.generate("the prompt", system="be safe", temperature=0.0, max_tokens=256)
    call = models.generate_calls[0]
    assert call["model"] == "gemini-test"
    assert call["contents"] == "the prompt"
    assert call["config"].system_instruction == "be safe"
    assert call["config"].temperature == 0.0
    # Thinking headroom: thinking models consume hidden tokens from the
    # output budget, so the visible-answer cap must be exceeded.
    assert call["config"].max_output_tokens == 256 + GeminiProvider._THINKING_HEADROOM_TOKENS


@pytest.mark.parametrize("text", [None, "", "   "])
def test_empty_responses_raise_response_error(monkeypatch, text):
    provider = _provider(monkeypatch, _FakeModels(text=text))
    with pytest.raises(LLMResponseError, match="empty response"):
        provider.generate("prompt")


# ---------------------------------------------------------------------------
# Secret hygiene
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("models_kwargs"),
    [
        {"error": genai_errors.ClientError(400, {})},
        {"error": httpx.ConnectError("boom")},
        {"text": None},
    ],
)
def test_api_key_never_leaks_into_errors(monkeypatch, models_kwargs):
    provider = _provider(monkeypatch, _FakeModels(**models_kwargs))
    with pytest.raises(Exception) as excinfo:  # noqa: PT011 - any LLM failure path
        provider.generate("prompt")
    assert SECRET not in str(excinfo.value)
    assert SECRET not in repr(excinfo.value)


def test_provider_repr_does_not_leak_key():
    provider = GeminiProvider(model="m", api_key=SECRET)
    assert SECRET not in repr(provider)
