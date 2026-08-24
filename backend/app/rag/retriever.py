"""Semantic retrieval over the knowledge base.

Pipeline: query -> embedding -> FAISS inner-product search -> ranked
:class:`RetrievalResult` objects with full document metadata. The retriever
loads the model and index once; per-query work is a single encode + search.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Sequence

from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.rag.embeddings import EmbeddingService, get_embedding_service
from app.rag.models import DocumentType, KnowledgeDocument
from app.rag.vector_store import (
    FaissVectorStore,
    StaleIndexError,
    embedding_text_for,
    load_knowledge_documents,
    sha256_of_file,
)

logger = logging.getLogger(__name__)


class RetrievalResult(BaseModel):
    """One ranked retrieval hit with its full document metadata."""

    document_id: str
    score: float
    document_type: str
    source: str
    tables: list[str] = Field(default_factory=list)
    content: str


class KnowledgeRetriever:
    """Reusable retrieval facade over the embedding service + vector store."""

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_store: FaissVectorStore | None = None,
        knowledge_path=None,  # Path | None -> settings default
    ) -> None:
        settings = get_settings()
        self.knowledge_path = knowledge_path or (settings.rag_output_dir / "documents" / "knowledge.jsonl")
        self.embedding_service = embedding_service or get_embedding_service()

        if vector_store is None:
            documents = load_knowledge_documents(self.knowledge_path)
            vector_store = FaissVectorStore.load(
                settings.rag_index_dir,
                expected_knowledge_sha256=sha256_of_file(self.knowledge_path),
                expected_embedding_model=self.embedding_service.model_name,
            )
            if len(documents) != vector_store.size:
                raise StaleIndexError(
                    f"Index holds {vector_store.size} vectors for "
                    f"{len(documents)} knowledge documents."
                )
        self.vector_store = vector_store

    # ------------------------------------------------------------------
    # Retrieval API
    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        document_type: str | DocumentType | None = None,
    ) -> list[RetrievalResult]:
        """Return the ``top_k`` most similar knowledge documents.

        Graceful degradation: an empty/whitespace-only query or a non-positive
        ``top_k`` yields an empty result list rather than raising. Results are
        unique by document and ordered best-first. An optional
        ``document_type`` filters results *after* semantic search so the
        default path always searches the complete knowledge base.
        """
        cleaned = (query or "").strip()
        if not cleaned or top_k is None or top_k <= 0:
            return []

        query_vector = self.embedding_service.embed_query(cleaned)

        filter_type = None
        if document_type is not None:
            filter_type = (
                document_type.value if isinstance(document_type, DocumentType)
                else str(document_type)
            )

        # With filtering we rank every document then slice, guaranteeing up to
        # top_k filtered hits; without it FAISS returns exactly top_k.
        fetch_k = self.vector_store.size if filter_type else min(top_k, self.vector_store.size)
        hits = self.vector_store.search(query_vector, fetch_k)

        results: list[RetrievalResult] = []
        seen_ids: set[str] = set()
        for position, score in hits:
            doc = self.vector_store.documents[position]
            if filter_type and doc.document_type.value != filter_type:
                continue
            assert doc.document_id not in seen_ids, "duplicate retrieval result"
            seen_ids.add(doc.document_id)
            results.append(_to_result(doc, score))
            if len(results) == top_k:
                break
        return results


def _to_result(doc: KnowledgeDocument, score: float) -> RetrievalResult:
    return RetrievalResult(
        document_id=doc.document_id,
        score=round(score, 6),
        document_type=doc.document_type.value,
        source=doc.source.value,
        tables=list(doc.tables),
        content=doc.content,
    )


def build_documents_text_map(documents: Sequence[KnowledgeDocument]) -> dict[str, str]:
    """Helper for tooling: document_id -> exact text that was embedded."""
    return {doc.document_id: embedding_text_for(doc) for doc in documents}


@lru_cache(maxsize=1)
def _default_retriever() -> KnowledgeRetriever:
    """Process-wide retriever: model + index loaded once, reused per query."""
    return KnowledgeRetriever()


def retrieve(
    query: str,
    top_k: int = 5,
    document_type: str | DocumentType | None = None,
) -> list[RetrievalResult]:
    """Module-level convenience API using a process-wide cached retriever."""
    return _default_retriever().retrieve(query, top_k=top_k, document_type=document_type)
