"""Google Gemini-backed LLM provider using the official ``google-genai`` SDK.

The API key is read from configuration (``GEMINI_API_KEY``) and is never
hardcoded, logged, or embedded in error messages. Failures map onto the
project's existing LLM exception hierarchy:

- missing/invalid key, unreachable network, 4xx auth problems, 5xx service
  errors -> :class:`LLMUnavailableError`
- malformed/empty model output -> :class:`LLMResponseError`
"""

from __future__ import annotations

import httpx
import requests

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from app.core.config import get_settings
from app.rag.llm.base import (
    LLMProvider,
    LLMResponseError,
    LLMUnavailableError,
)

#: Network-level failures that mean "cannot reach the Gemini endpoint".
_NETWORK_ERRORS = (httpx.HTTPError, requests.RequestException)


class GeminiProvider(LLMProvider):
    """Talks to the Google Gemini API (cloud, key-based)."""

    name = "gemini"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        settings = get_settings()
        self.model = model
        self.timeout_seconds = (
            timeout_seconds if timeout_seconds is not None else settings.llm_timeout_seconds
        )
        # SecretStr keeps the key out of reprs/logs; unwrap only here.
        configured = (
            api_key if api_key is not None else settings.gemini_api_key.get_secret_value()
        )
        self.api_key = (configured or "").strip()

    # ------------------------------------------------------------------
    # Client / availability
    # ------------------------------------------------------------------
    def _client(self) -> genai.Client:
        """Build an SDK client; raises when no API key is configured."""
        if not self.api_key:
            raise LLMUnavailableError(
                "Gemini API key is missing; set GEMINI_API_KEY in the environment "
                "or .env file (never hardcode it in source code)."
            )
        return genai.Client(
            api_key=self.api_key,
            http_options={"timeout": int(self.timeout_seconds * 1000)},
        )

    def is_available(self) -> bool:
        """True when a key exists *and* the configured model is reachable."""
        try:
            client = self._client()
            client.models.get(model=self.model)
        except LLMUnavailableError:
            return False
        except (genai_errors.APIError, ValueError, RuntimeError, *_NETWORK_ERRORS):
            return False
        return True

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    #: Gemini 3.x models "think" before answering and those hidden tokens
    #: count against ``max_output_tokens``. We request minimal thinking
    #: (deterministic SQL generation needs no deep reasoning) and budget
    #: extra room so the visible answer is never truncated mid-statement.
    _THINKING_HEADROOM_TOKENS = 1024

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        client = self._client()  # LLMUnavailableError when the key is absent
        config = genai_types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens + self._THINKING_HEADROOM_TOKENS,
            thinking_config=genai_types.ThinkingConfig(
                include_thoughts=False,
                thinking_level=genai_types.ThinkingLevel.MINIMAL,
            ),
        )

        try:
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
        except genai_errors.ClientError as exc:
            if exc.code in (400, 401, 403):
                raise LLMUnavailableError(
                    f"Gemini rejected the request (HTTP {exc.code}); the API key "
                    f"may be invalid or lacks access to model {self.model!r}."
                ) from exc
            raise LLMResponseError(
                f"Gemini API client error (HTTP {exc.code})."
            ) from exc
        except genai_errors.ServerError as exc:
            raise LLMUnavailableError(
                f"Gemini service is currently unavailable (HTTP {exc.code}); "
                "retry later."
            ) from exc
        except _NETWORK_ERRORS as exc:
            raise LLMUnavailableError(
                f"Gemini endpoint is unreachable ({exc.__class__.__name__})."
            ) from exc

        text = response.text  # Optional[str]: None when no text parts came back
        if not isinstance(text, str) or not text.strip():
            raise LLMResponseError("Gemini returned an empty response.")
        return text.strip()
