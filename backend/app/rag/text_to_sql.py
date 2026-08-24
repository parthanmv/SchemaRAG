"""Text-to-SQL orchestration (generation only - never execution).

Pipeline:
    question -> KnowledgeRetriever -> ContextAssembler -> prompt -> LLM
             -> SQL extraction/parsing -> grounding check -> GeneratedSQL

The generated SQL is returned to the caller untouched (no repair); executing
it is explicitly out of scope until Phase 5.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Sequence

from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.rag.context import AssembledContext, ContextAssembler
from app.rag.grounding import GroundingReport, ground_sql
from app.rag.llm.base import LLMProvider, create_provider
from app.rag.models import SchemaMetadata
from app.rag.preprocessing import ProcessedQuestion, QueryPreprocessor
from app.rag.prompts import build_system_message, build_text_to_sql_prompt
from app.rag.retriever import KnowledgeRetriever, get_embedding_service
from app.rag.sql_parsing import (
    InvalidSQLResponseError,
    validate_single_select,
)

logger = logging.getLogger(__name__)


class GeneratedSQL(BaseModel):
    """Typed end-to-end result of one Text-to-SQL generation attempt."""

    question: str
    #: Phase 7: the normalised question actually used for retrieval/prompting
    #: (None for legacy call sites that construct results directly).
    processed_question: str | None = None
    sql: str | None = None
    model: str
    grounded: bool = False
    retrieved_documents: list[str] = Field(default_factory=list)
    retrieval_scores: list[float] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    error: str | None = None


def load_schema_metadata_snapshot(path=None) -> SchemaMetadata:  # noqa: ANN001
    """Load the Phase 2 metadata snapshot written by ``extract_metadata``."""
    import pathlib

    settings = get_settings()
    metadata_path = pathlib.Path(
        path or (settings.rag_output_dir / "metadata" / "schema_metadata.json")
    )
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Schema metadata snapshot not found at {metadata_path}; run "
            f"'python -m app.scripts.extract_metadata' first."
        )
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    return SchemaMetadata.model_validate(payload)


class TextToSQLService:
    """Reusable facade over retrieval, prompting, generation and grounding."""

    def __init__(
        self,
        provider: LLMProvider | None = None,
        retriever: KnowledgeRetriever | None = None,
        assembler: ContextAssembler | None = None,
        schema_metadata: SchemaMetadata | None = None,
        top_k: int | None = None,
        preprocessor: QueryPreprocessor | None = None,
    ) -> None:
        settings = get_settings()
        self.provider = provider or create_provider(
            settings.llm_provider, settings.active_llm_model
        )
        self.retriever = retriever or KnowledgeRetriever(
            embedding_service=get_embedding_service()
        )
        self.assembler = assembler or ContextAssembler()
        self.schema_metadata = schema_metadata or load_schema_metadata_snapshot()
        self.top_k = top_k if top_k is not None else settings.rag_top_k
        # Phase 7 pipeline stage 1: query preprocessing.
        self.preprocessor = preprocessor or QueryPreprocessor()

    # ------------------------------------------------------------------
    def generate(self, question: str) -> GeneratedSQL:
        """Run the full pipeline for *question* and return a typed result."""
        cleaned = (question or "").strip()
        if not cleaned:
            raise ValueError("question must be a non-empty string")

        processed: ProcessedQuestion = self.preprocessor.preprocess(cleaned)
        results = self.retriever.retrieve(processed.processed, top_k=self.top_k)
        context = self.assembler.assemble(results)

        base_kwargs = {
            "question": cleaned,
            "processed_question": processed.processed,
            "model": f"{self.provider.name}:{self.provider.model}",
            "retrieved_documents": [r.document_id for r in results],
            "retrieval_scores": [r.score for r in results],
        }

        prompt = build_text_to_sql_prompt(processed.processed, context)
        response = self.provider.generate(
            prompt, system=build_system_message(), temperature=0.0
        )

        try:
            parsed = validate_single_select(response)
        except InvalidSQLResponseError as exc:
            message = str(exc)
            error = (
                "insufficient_context" if message.startswith("insufficient context")
                else "invalid_response"
            )
            logger.info("Generation rejected (%s): %s", error, message)
            return GeneratedSQL(
                grounded=False, error=error, issues=[message], **base_kwargs
            )

        report: GroundingReport = ground_sql(parsed.sql, self.schema_metadata)
        return GeneratedSQL(
            question=cleaned,
            processed_question=processed.processed,
            sql=parsed.sql,
            model=f"{self.provider.name}:{self.provider.model}",
            grounded=report.grounded,
            retrieved_documents=list(base_kwargs["retrieved_documents"]),
            retrieval_scores=list(base_kwargs["retrieval_scores"]),
            issues=list(report.issues),
            error=None if report.grounded else "not_grounded",
        )


@lru_cache(maxsize=1)
def _cached_service() -> TextToSQLService:
    """Process-wide service (embedding model/index/metadata loaded once)."""
    return TextToSQLService()


def get_sql_generation_service() -> TextToSQLService:
    """FastAPI dependency returning the cached service instance."""
    return _cached_service()


def format_generated(result: GeneratedSQL) -> str:
    """Human-friendly multi-line rendering used by the CLI."""
    lines = [
        f"Question : {result.question}",
        f"Model    : {result.model}",
    ]
    lines.append("Retrieved documents:")
    for doc_id, score in zip(
        result.retrieved_documents, result.retrieval_scores, strict=False
    ):
        lines.append(f"  - {doc_id} ({score:.4f})")
    if result.sql:
        lines.append("Generated SQL:")
        lines.append(f"  {result.sql}")
    else:
        lines.append(f"Generated SQL: <none> ({result.error})")
    lines.append(f"Grounded : {'yes' if result.grounded else 'no'}")
    for issue in result.issues:
        lines.append(f"  ! {issue}")
    return "\n".join(lines)


def summarize_documents(results: Sequence) -> list[str]:  # noqa: ANN001 - RetrievalResult
    return [f"{r.document_id} ({r.score:.4f})" for r in results]
