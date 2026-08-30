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


#: Minimum number of documents to surface from each content type on the
#: default (all-corpus) retrieval path.
#
# ``None`` means "every document of that type".
#
# Text-to-SQL needs the grounded join backbone: the table definitions
# (``schema``) and the relationships between them (``relationship``) plus a
# couple of worked ``query_example``s. Purely score-driven top-k selection lets
# a cluster of semantically similar query examples crowd these information-dense
# documents out of context, so complex multi-table questions (e.g. "students who
# scored >80% in every subject") can no longer be constructed or grounded. The
# full schema is always included because it is small, authoritative and the
# definitive grounding context - no table referenced by a relationship or
# worked example can ever be missing. These guarantees are GENERIC: they apply
# to every query, not to any particular question.
TYPE_FLOORS: dict[str, int | None] = {
    "schema": None,  # all table definitions, always
    "relationship": 4,
    "constraint": 1,
    "business_rule": None,  # all (few, small; mark-scale rule resolves "80%" etc.)
    "query_example": 2,
}


class KnowledgeRetriever:
    """Reusable retrieval facade over the embedding service + vector store."""

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_store: FaissVectorStore | None = None,
        knowledge_path=None,  # Path | None -> settings default
        type_floors: dict[str, int | None] | None = None,
    ) -> None:
        settings = get_settings()
        self.knowledge_path = knowledge_path or (settings.rag_output_dir / "documents" / "knowledge.jsonl")
        self.embedding_service = embedding_service or get_embedding_service()
        self.type_floors = dict(type_floors or TYPE_FLOORS)

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
        unique by document and (on the default path) returned best-first by
        score. An optional ``document_type`` filters results *after* semantic
        search so the default path always searches the complete knowledge base.

        On the default path (``document_type is None``) the selection is
        type-stratified: each content type contributes up to its
        :data:`TYPE_FLOORS` minimum (scaled down to fit ``top_k`` when needed)
        so the schema + relationship backbone required for grounded multi-table
        SQL is never crowded out by a run of similar query examples. The chosen
        set is still ordered best-first by semantic score.
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
        # top_k filtered hits; without it we rank everything so the type floors
        # can draw from the full corpus.
        fetch_k = self.vector_store.size
        hits = self.vector_store.search(query_vector, fetch_k)

        if filter_type is not None:
            results: list[RetrievalResult] = []
            seen_ids: set[str] = set()
            for position, score in hits:
                doc = self.vector_store.documents[position]
                if doc.document_type.value != filter_type:
                    continue
                assert doc.document_id not in seen_ids, "duplicate retrieval result"
                seen_ids.add(doc.document_id)
                results.append(_to_result(doc, score))
                if len(results) == top_k:
                    break
            return results

        ranked = [_to_result(self.vector_store.documents[pos], score)
                  for pos, score in hits]
        return _select_with_type_cover(ranked, top_k, self.type_floors)


def _to_result(doc: KnowledgeDocument, score: float) -> RetrievalResult:
    return RetrievalResult(
        document_id=doc.document_id,
        score=round(score, 6),
        document_type=doc.document_type.value,
        source=doc.source.value,
        tables=list(doc.tables),
        content=doc.content,
    )


def _select_with_type_cover(
    ranked: list[RetrievalResult],
    top_k: int,
    floors: dict[str, int | None],
) -> list[RetrievalResult]:
    """Choose ``top_k`` documents guaranteeing per-type representation.

    ``ranked`` is the full semantic ranking (best-first). Documents are grouped
    by type (score descending within each type). ``floors`` is a
    :data:`TYPE_FLOORS`-style minimum to draw from each type, where ``None``
    means "every document of that type"; floors are scaled down
    proportionally if their sum exceeds ``top_k``. Selection proceeds greedily
    in type-priority order, then any remaining slots are filled with the
    best-scoring documents not yet chosen (from any type). The final set is
    deterministic, deduplicated and returned best-first by score.
    """
    by_type: dict[str, list[RetrievalResult]] = {}
    for result in ranked:
        by_type.setdefault(result.document_type, []).append(result)

    types_in_order = [t for t in (
        "schema", "relationship", "constraint", "business_rule", "query_example"
    ) if by_type.get(t)]

    scaled = _scaled_floors(floors, by_type, top_k, types_in_order)

    locked: list[RetrievalResult] = []
    for doc_type in types_in_order:
        for _ in range(scaled.get(doc_type, 0)):
            locked.append(by_type[doc_type].pop(0))

    chosen: list[RetrievalResult] = list(locked)
    if len(chosen) < top_k:
        pool: list[RetrievalResult] = []
        for doc_type in types_in_order:
            pool.extend(by_type[doc_type])
        pool.sort(key=lambda r: r.score, reverse=True)
        for result in pool:
            if len(chosen) >= top_k:
                break
            chosen.append(result)

    chosen.sort(key=lambda r: r.score, reverse=True)
    return chosen[:top_k]


def _scaled_floors(
    floors: dict[str, int | None],
    by_type: dict[str, list[RetrievalResult]],
    top_k: int,
    types_in_order: list[str],
) -> dict[str, int]:
    """Scale per-type floors so they never exceed what each type / top_k allows.

    ``None`` floors mean "all documents of that type". Returns a dict of
    feasible floor counts given (a) how many documents each type actually has
    and (b) the overall ``top_k`` budget.
    """
    out: dict[str, int] = {}
    for doc_type in types_in_order:
        desired = floors.get(doc_type, 0)
        if desired is None:
            desired = len(by_type[doc_type])
        out[doc_type] = min(desired, len(by_type[doc_type]))

    total = sum(out.values())
    if total <= top_k:
        return out

    # Over-budget: keep priority order, dropping whole-type floors from the
    # bottom of the priority list until the sum fits (preserving the highest
    # priority types first).
    priority = list(reversed(types_in_order))
    while sum(out.values()) > top_k and priority:
        low = priority.pop(0)
        if out.get(low, 0):
            out[low] = out[low] - 1
    return out


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
