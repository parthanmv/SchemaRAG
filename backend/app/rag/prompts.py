"""Grounded Text-to-SQL prompt construction.

The prompt contains only retrieved, verified knowledge-base content plus the
user question. The instruction block forbids non-SELECT statements and
invented schema information; when the context is insufficient the model is
told to answer with an explicit ``INSUFFICIENT_CONTEXT`` marker instead of
guessing.
"""

from __future__ import annotations

from app.rag.context import AssembledContext, SECTION_TITLES

INSTRUCTION = (
    "Generate exactly one PostgreSQL SELECT query using ONLY the provided "
    "tables, columns and relationships. Never invent schema information. "
    "Do not generate INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, "
    "GRANT or REVOKE. Do not execute SQL. If context is insufficient, report "
    "insufficient context by replying with exactly 'INSUFFICIENT_CONTEXT: "
    "<reason>' and nothing else. Respond with the SQL inside a ```sql code "
    "block."
)

SECTION_ORDER = ("schema", "relationship", "constraint", "business_rule", "query_example")


def build_text_to_sql_prompt(question: str, context: AssembledContext) -> str:
    """Render the full prompt: instruction, context sections, question.

    All five context headers are always emitted in fixed order so the layout
    is stable; sections without retrieved documents show an explicit
    placeholder instead of silently disappearing.
    """
    parts: list[str] = [INSTRUCTION, ""]
    for key in SECTION_ORDER:
        entries = context.sections.get(key, ())
        title = SECTION_TITLES[key]  # plural display title, e.g. DATABASE SCHEMA
        if entries:
            body = "\n".join(f"- {entry}" for entry in entries)
            parts.append(f"=== {title} ===")
            parts.append(body)
        else:
            parts.append(f"=== {title} ===")
            parts.append("(no retrieved documents)")
        parts.append("")

    parts.append("=== USER QUESTION ===")
    parts.append(question.strip())
    return "\n".join(parts)


def build_system_message() -> str:
    """System message reinforcing the read-only, grounded behaviour."""
    return (
        "You are a careful PostgreSQL assistant that only writes SELECT "
        "queries strictly grounded in the provided database schema. You never "
        "execute SQL and never invent tables or columns."
    )
