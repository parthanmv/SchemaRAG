"""Context assembly for Text-to-SQL prompts.

Takes the ranked results from the Phase 3 :class:`KnowledgeRetriever` and
builds a deduplicated, priority-ordered, character-budgeted context grouped
by document type:

    schema > relationship > constraint > business_rule > query_example

Nothing is invented here: only retrieved document content is emitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.rag.retriever import RetrievalResult

#: Lower rank == higher priority.
TYPE_PRIORITY: dict[str, int] = {
    "schema": 0,
    "relationship": 1,
    "constraint": 2,
    "business_rule": 3,
    "query_example": 4,
}

SECTION_TITLES: dict[str, str] = {
    "schema": "DATABASE SCHEMA",
    "relationship": "RELATIONSHIPS",
    "constraint": "CONSTRAINTS",
    "business_rule": "BUSINESS RULES",
    "query_example": "QUERY EXAMPLES",
}


@dataclass
class AssembledContext:
    """Grouped prompt-ready context plus bookkeeping."""

    sections: dict[str, tuple[str, ...]] = field(default_factory=dict)
    used_documents: tuple[str, ...] = field(default_factory=tuple)
    dropped_documents: tuple[str, ...] = field(default_factory=tuple)
    total_chars: int = 0

    def render(self) -> str:
        """Render the context blocks in fixed priority order."""
        blocks: list[str] = []
        for doc_type in sorted(self.sections, key=lambda t: TYPE_PRIORITY.get(t, 99)):
            title = SECTION_TITLES.get(doc_type, doc_type.upper())
            entries = self.sections[doc_type]
            if not entries:
                continue
            body = "\n".join(f"- {entry}" for entry in entries)
            blocks.append(f"=== {title} ===\n{body}")
        return "\n\n".join(blocks)


class ContextAssembler:
    """Builds the grounded LLM context from retrieval results."""

    def __init__(self, max_chars: int | None = None) -> None:
        from app.core.config import get_settings

        settings = get_settings()
        self.max_chars = (
            max_chars if max_chars is not None else settings.max_context_chars
        )

    def assemble(
        self, results: list[RetrievalResult]
    ) -> AssembledContext:
        """Group, prioritise and budget retrieved documents.

        * duplicates (same ``document_id``) are removed, best score kept
        * documents are considered in priority order, score-descending within
          a type; once the character budget is exhausted, remaining documents
          are recorded as dropped (never silently merged into the prompt)
        """
        unique: dict[str, RetrievalResult] = {}
        for result in results:
            current = unique.get(result.document_id)
            if current is None or result.score > current.score:
                unique[result.document_id] = result

        ordered = sorted(
            unique.values(),
            key=lambda r: (TYPE_PRIORITY.get(r.document_type, 99), -r.score,
                           r.document_id),
        )

        sections: dict[str, list[str]] = {}
        used: list[str] = []
        dropped: list[str] = []
        budget = self.max_chars

        for result in ordered:
            text = " ".join(result.content.split())
            if len(text) > budget:
                dropped.append(result.document_id)
                continue
            budget -= len(text)
            sections.setdefault(result.document_type, []).append(text)
            used.append(result.document_id)

        return AssembledContext(
            sections={t: tuple(v) for t, v in sections.items()},
            used_documents=tuple(used),
            dropped_documents=tuple(dropped),
            total_chars=sum(len(x) for v in sections.values() for x in v),
        )
