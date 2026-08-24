"""LLM provider abstraction for Text-to-SQL.

Supported providers: Google Gemini (cloud, ``google-genai`` SDK) and Ollama
(local fallback). Providers raise :class:`LLMUnavailableError` when their
backend cannot be reached/used so callers can degrade cleanly; secrets are
always sourced from configuration, never from code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMError(RuntimeError):
    """Base class for LLM-related failures."""


class LLMUnavailableError(LLMError):
    """The configured LLM server or model cannot be reached/used."""


class LLMResponseError(LLMError):
    """The LLM returned a malformed HTTP payload."""


class LLMProvider(ABC):
    """Minimal interface every provider must implement."""

    #: short provider identifier ("gemini", "ollama", ...)
    name: str = "base"
    #: the concrete model name this instance generates with
    model: str = "unknown"

    @abstractmethod
    def is_available(self) -> bool:
        """Cheap check whether the server and model are usable right now."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        """Generate one completion for *prompt* and return it as plain text."""


def create_provider(provider_name: str, model: str) -> LLMProvider:
    """Factory used by configuration; extensible with more backends."""
    if provider_name == "gemini":
        # Imported lazily to keep import time low for non-LLM code paths.
        from app.rag.llm.gemini import GeminiProvider

        return GeminiProvider(model=model)
    if provider_name == "ollama":
        from app.rag.llm.ollama import OllamaProvider

        return OllamaProvider(model=model)
    raise LLMUnavailableError(f"Unknown LLM provider {provider_name!r}")
