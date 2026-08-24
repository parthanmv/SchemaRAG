"""LLM provider package (Gemini cloud + Ollama local fallback).

Provider modules are imported lazily: ``google.genai`` alone costs ~2s of
import time, which would otherwise be paid by every non-LLM code path.
"""

from typing import TYPE_CHECKING

from app.rag.llm.base import (
    LLMError,
    LLMProvider,
    LLMResponseError,
    LLMUnavailableError,
    create_provider,
)

if TYPE_CHECKING:  # pragma: no cover
    from app.rag.llm.gemini import GeminiProvider
    from app.rag.llm.ollama import OllamaProvider

_LAZY_EXPORTS = {
    "GeminiProvider": "app.rag.llm.gemini",
    "OllamaProvider": "app.rag.llm.ollama",
}

__all__ = [
    *_LAZY_EXPORTS,
    "LLMError",
    "LLMProvider",
    "LLMResponseError",
    "LLMUnavailableError",
    "create_provider",
]


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        from importlib import import_module

        return getattr(import_module(_LAZY_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
