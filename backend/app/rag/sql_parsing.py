"""Extraction and light parsing of SQL from LLM responses.

This is deliberately NOT the Phase 5 security validator: it only extracts a
single candidate statement from the model output and performs structural
sanity checks (exactly one statement, SELECT/WITH only, non-empty). Full
security validation and execution belong to Phase 5.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import sqlglot
from sqlglot import expressions as exp

#: Statement keywords accepted as "a read query".
_ALLOWED_STARTS = ("SELECT", "WITH")
_INSUFFICIENT_MARKER_RE = re.compile(r"INSUFFICIENT_CONTEXT\s*[:\-]?\s*(.*)", re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


class InvalidSQLResponseError(ValueError):
    """The LLM response did not contain exactly one readable read-only query."""


@dataclass(frozen=True)
class ParsedSQL:
    """Result of extracting + parsing one candidate statement."""

    sql: str
    ast: object  # sqlglot Expression


def extract_sql(response: str) -> str:
    """Extract one SQL statement string from an LLM response.

    Accepts fenced (```` ```sql ... ``` ````) or bare responses. Raises
    :class:`InvalidSQLResponseError` for empty responses, multiple
    statements, or statements that do not start with SELECT/WITH.
    """
    if not response or not response.strip():
        raise InvalidSQLResponseError("LLM returned an empty response")

    text = response.strip()

    insufficient = _INSUFFICIENT_MARKER_RE.search(text)
    if insufficient:
        reason = (insufficient.group(1) or "").strip() or "context was insufficient"
        raise InvalidSQLResponseError(f"insufficient context: {reason}")

    fences = _FENCE_RE.findall(text)
    if fences:
        text = max(fences, key=len).strip()

    text = text.strip().strip(";").strip()
    if not text:
        raise InvalidSQLResponseError("LLM response contained no SQL text")

    first_word = text.split(None, 1)[0].upper().rstrip(";")
    if first_word not in _ALLOWED_STARTS:
        raise InvalidSQLResponseError(
            f"response does not look like a SELECT query (starts with {first_word!r})"
        )

    # Multiple statements: split on semicolons; at most one non-empty part.
    parts = [p.strip() for p in text.split(";") if p.strip()]
    if len(parts) > 1:
        raise InvalidSQLResponseError(
            f"multiple SQL statements are not allowed ({len(parts)} found)"
        )

    return parts[0]


def parse_sql(sql: str) -> ParsedSQL:
    """Parse *sql* with the PostgreSQL dialect; raises on syntax errors."""
    try:
        ast = sqlglot.parse_one(sql, dialect="postgres")
    except sqlglot.errors.ParseError as exc:
        raise InvalidSQLResponseError(f"SQL does not parse: {exc}") from exc
    if ast is None:
        raise InvalidSQLResponseError("SQL parses to an empty expression")
    return ParsedSQL(sql=sql, ast=ast)


def validate_single_select(sql: str) -> ParsedSQL:
    """Extract + parse + assert the AST really is a read query."""
    extracted = extract_sql(sql)
    parsed = parse_sql(extracted)
    root = parsed.ast
    if isinstance(root, exp.With):
        root = root.this  # CTE wrapper: the actual SELECT sits below
    if not isinstance(root, (exp.Select, exp.Union)):
        raise InvalidSQLResponseError(
            f"only SELECT queries are allowed (got {type(root).__name__})"
        )
    return parsed
