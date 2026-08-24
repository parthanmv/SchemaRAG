"""Phase 7: query preprocessing - the first stage of the product pipeline.

Normalises the raw user question BEFORE retrieval and prompting so that
semantically identical inputs ("Which  department…?", curly quotes, stray
whitespace) embed and retrieve consistently. The stage is purely textual and
deterministic: no LLM calls, no schema knowledge, no SQL handling.

The ORIGINAL question is always preserved alongside the processed form; API
contracts keep echoing the original text while retrieval/prompting run on the
processed one.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

#: Hard ceiling mirrors the API question limit (3-500 chars); longer input is
#: truncated defensively so downstream embedding/prompt budgets hold even for
#: direct service callers that bypass HTTP validation.
MAX_QUESTION_CHARS = 500

#: Typographic characters commonly pasted into questions that would otherwise
#: fragment embeddings. Mapping is explicit (no locale-dependent transforms).
_PUNCTUATION_MAP = {
    ord("\u2018"): "'",  # left single quote
    ord("\u2019"): "'",  # right single quote
    ord("\u201c"): '"',  # left double quote
    ord("\u201d"): '"',  # right double quote
    ord("\u2013"): "-",  # en dash
    ord("\u2014"): "-",  # em dash
    ord("\u00a0"): " ",  # non-breaking space
    ord("\u200b"): "",   # zero-width space
}

_WHITESPACE_RE = re.compile(r"\s+")


class ProcessedQuestion(BaseModel):
    """Original vs processed question plus the transformations applied."""

    original: str
    processed: str
    transformations: list[str] = Field(default_factory=list)


class QueryPreprocessor:
    """Deterministic text normalisation for incoming questions."""

    def preprocess(self, question: str) -> ProcessedQuestion:
        """Return the cleaned question and a record of what was changed.

        Raises ``ValueError`` when nothing usable remains after cleaning.
        The transformation is idempotent: processing an already-processed
        question yields the same text and an empty transformation list.
        """
        original = question if isinstance(question, str) else ""
        current = original
        applied: list[str] = []

        def apply(name: str, transform) -> None:
            nonlocal current
            updated = transform(current)
            if updated != current:
                applied.append(name)
                current = updated

        apply("normalize_unicode_punctuation", self._normalize_punctuation)
        apply("collapse_whitespace", self._collapse_whitespace)
        apply("strip", str.strip)
        apply("truncate_to_limit", self._truncate)

        if not current:
            raise ValueError("question must be a non-empty string")

        return ProcessedQuestion(
            original=original, processed=current, transformations=applied
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_punctuation(text: str) -> str:
        return text.translate(_PUNCTUATION_MAP)

    @staticmethod
    def _collapse_whitespace(text: str) -> str:
        return _WHITESPACE_RE.sub(" ", text)

    @staticmethod
    def _truncate(text: str) -> str:
        return text[:MAX_QUESTION_CHARS]
