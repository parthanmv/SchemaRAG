"""Ollama-backed LLM provider (local server, no secrets, no auto-downloads).

Uses ``POST {base_url}/api/generate`` with ``stream=false``. Availability is
checked against both the server (``GET /api/tags``) and the configured model
name; a running server without the pulled model still counts as unavailable.
"""

from __future__ import annotations

import httpx

from app.core.config import get_settings
from app.rag.llm.base import (
    LLMProvider,
    LLMResponseError,
    LLMUnavailableError,
)


class OllamaProvider(LLMProvider):
    """Talks to a local Ollama server."""

    name = "ollama"

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        settings = get_settings()
        self.model = model
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.timeout_seconds = (
            timeout_seconds if timeout_seconds is not None else settings.llm_timeout_seconds
        )

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------
    def _client(self, timeout: float | None = None) -> httpx.Client:
        return httpx.Client(
            base_url=self.base_url,
            timeout=timeout if timeout is not None else self.timeout_seconds,
        )

    def is_available(self) -> bool:
        """True only when the server responds *and* lists the model."""
        try:
            with self._client(timeout=2.0) as client:
                response = client.get("/api/tags")
                response.raise_for_status()
                names = [m.get("name", "") for m in response.json().get("models", [])]
        except (httpx.HTTPError, ValueError):
            return False
        return any(n == self.model or n.split(":")[0] == self.model for n in names)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system:
            payload["system"] = system

        try:
            with self._client() as client:
                response = client.post("/api/generate", json=payload)
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(
                f"Ollama server at {self.base_url} is unreachable: {exc!r}"
            ) from exc

        if response.status_code == 404:
            raise LLMUnavailableError(
                f"Model {self.model!r} is not available on the Ollama server; "
                f"pull it first (e.g. 'ollama pull {self.model}')."
            )
        try:
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise LLMResponseError(f"Unexpected Ollama response: {exc!r}") from exc

        text = body.get("response")
        if not isinstance(text, str):
            raise LLMResponseError("Ollama response missing 'response' text field")
        return text.strip()
