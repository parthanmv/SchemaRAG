"""FAISS vector store for knowledge documents.

The store keeps two artefacts on disk:

* ``index.faiss``        – a flat inner-product index over normalised vectors
* ``document_store.json`` – the documents in vector order plus a manifest
  recording the embedding model, dimension, FAISS index type, document count
  and the SHA-256 of the ``knowledge.jsonl`` the vectors were built from.

FAISS itself stores no application metadata: position ``i`` in the index
always corresponds to ``documents[i]``. Loading validates freshness against
the current knowledge base so a stale index can never be used silently.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import faiss
import numpy as np

from app.rag.models import KnowledgeDocument

logger = logging.getLogger(__name__)

INDEX_FILENAME = "index.faiss"
STORE_FILENAME = "document_store.json"

SIMILARITY_METRIC = "cosine (inner product over L2-normalised vectors)"


class StaleIndexError(RuntimeError):
    """Raised when an on-disk index does not match the current knowledge base."""


def sha256_of_file(path: Path) -> str:
    """Stable content digest used for index-freshness checks."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def embedding_text_for(document: KnowledgeDocument) -> str:
    """Deterministic text that gets embedded for a document."""
    return f"{document.title}\n{document.content}"


class IndexManifest:
    """Provenance/freshness information for one built index."""

    def __init__(
        self,
        embedding_model: str,
        embedding_dimension: int,
        faiss_index_type: str,
        document_count: int,
        knowledge_sha256: str,
    ) -> None:
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension
        self.faiss_index_type = faiss_index_type
        self.document_count = document_count
        self.knowledge_sha256 = knowledge_sha256

    def to_dict(self) -> dict:
        return {
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "faiss_index_type": self.faiss_index_type,
            "document_count": self.document_count,
            "knowledge_sha256": self.knowledge_sha256,
            "similarity_metric": SIMILARITY_METRIC,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> IndexManifest:
        return cls(
            embedding_model=payload["embedding_model"],
            embedding_dimension=int(payload["embedding_dimension"]),
            faiss_index_type=payload["faiss_index_type"],
            document_count=int(payload["document_count"]),
            knowledge_sha256=payload["knowledge_sha256"],
        )


class FaissVectorStore:
    """FAISS index plus its deterministic position -> document mapping."""

    def __init__(
        self,
        index: faiss.Index,
        documents: tuple[KnowledgeDocument, ...],
        manifest: IndexManifest,
    ) -> None:
        if index.ntotal != len(documents):
            raise ValueError(
                f"Index holds {index.ntotal} vectors but {len(documents)} documents"
            )
        self.index = index
        self.documents = documents
        self.manifest = manifest

    # ------------------------------------------------------------------
    # Build / persistence
    # ------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        documents: tuple[KnowledgeDocument, ...],
        embeddings: np.ndarray,
        embedding_model: str,
        knowledge_sha256: str = "",
    ) -> FaissVectorStore:
        """Create a flat inner-product index over normalised embeddings."""
        ids = [doc.document_id for doc in documents]
        if len(set(ids)) != len(ids):
            duplicates = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"Duplicate document_ids are not allowed: {duplicates}")
        expected_shape = (len(documents), int(embeddings.shape[1]))
        if embeddings.shape != expected_shape:
            raise ValueError(
                f"Embeddings shape {embeddings.shape} does not match "
                f"documents {expected_shape}"
            )
        norms = np.linalg.norm(embeddings.astype(np.float32), axis=1)
        if len(documents) and not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError("Embeddings must be L2-normalised before indexing")

        index = faiss.IndexFlatIP(int(embeddings.shape[1]))
        index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
        manifest = IndexManifest(
            embedding_model=embedding_model,
            embedding_dimension=int(embeddings.shape[1]),
            faiss_index_type=type(index).__name__,
            document_count=len(documents),
            knowledge_sha256=knowledge_sha256,
        )
        return cls(index=index, documents=tuple(documents), manifest=manifest)

    def save(self, directory: Path) -> tuple[Path, Path]:
        """Write ``index.faiss`` and ``document_store.json`` into *directory*."""
        directory.mkdir(parents=True, exist_ok=True)
        index_path = directory / INDEX_FILENAME
        store_path = directory / STORE_FILENAME
        faiss.write_index(self.index, str(index_path))
        payload = {
            "manifest": self.manifest.to_dict(),
            "documents": [
                json.loads(doc.model_dump_json()) for doc in self.documents
            ],
        }
        store_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True),
            encoding="utf-8",
            newline="\n",
        )
        logger.info("Saved FAISS index to %s", index_path)
        logger.info("Saved document store to %s", store_path)
        return index_path, store_path

    @classmethod
    def load(
        cls,
        directory: Path,
        expected_knowledge_sha256: str | None = None,
        expected_embedding_model: str | None = None,
    ) -> FaissVectorStore:
        """Load a previously saved store.

        When expectations are provided they are enforced strictly so callers
        never silently search stale vectors.
        """
        index_path = directory / INDEX_FILENAME
        store_path = directory / STORE_FILENAME
        if not index_path.is_file() or not store_path.is_file():
            raise FileNotFoundError(
                f"No vector index found in {directory}; run "
                f"'python -m app.scripts.build_vector_index' first."
            )

        payload = json.loads(store_path.read_text(encoding="utf-8"))
        manifest = IndexManifest.from_dict(payload["manifest"])
        documents = tuple(KnowledgeDocument.model_validate(d) for d in payload["documents"])

        if expected_embedding_model and manifest.embedding_model != expected_embedding_model:
            raise StaleIndexError(
                f"Index was built with model {manifest.embedding_model!r} but the "
                f"configured model is {expected_embedding_model!r}. Rebuild via "
                f"'python -m app.scripts.build_vector_index'."
            )
        if expected_knowledge_sha256 and (
            manifest.knowledge_sha256 != expected_knowledge_sha256
            or manifest.document_count != len(documents)
        ):
            raise StaleIndexError(
                "The existing index was built from a different knowledge.jsonl "
                f"(index digest {manifest.knowledge_sha256[:12]}... != current "
                f"{expected_knowledge_sha256[:12]}...). Rebuild via "
                f"'python -m app.scripts.build_vector_index'."
            )

        index = faiss.read_index(str(index_path))
        store = cls(index=index, documents=documents, manifest=manifest)
        if store.dimension != manifest.embedding_dimension:
            raise StaleIndexError(
                f"Index dimension {store.dimension} disagrees with manifest "
                f"{manifest.embedding_dimension}"
            )
        return store

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    @property
    def dimension(self) -> int:
        return int(self.index.d)

    @property
    def size(self) -> int:
        return int(self.index.ntotal)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        """Return up to ``top_k`` unique ``(position, score)`` pairs, best first."""
        if top_k <= 0 or self.size == 0:
            return []
        vector = np.ascontiguousarray(
            np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        )
        k = min(top_k, self.size)
        scores, positions = self.index.search(vector, k)
        results: list[tuple[int, float]] = []
        seen: set[int] = set()
        for position, score in zip(positions[0], scores[0]):
            pos = int(position)
            if pos < 0 or pos in seen:
                continue  # safety: FAISS returns -1 when fewer than k vectors exist
            seen.add(pos)
            results.append((pos, float(score)))
        return results


def load_knowledge_documents(path: Path) -> tuple[KnowledgeDocument, ...]:
    """Load and structurally validate documents from ``knowledge.jsonl``."""
    if not path.is_file():
        raise FileNotFoundError(f"Knowledge file not found: {path}")
    documents: list[KnowledgeDocument] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        missing = {"document_id", "document_type", "content", "source"} - set(payload)
        if missing:
            raise ValueError(f"knowledge.jsonl line {line_number} missing fields: {sorted(missing)}")
        doc = KnowledgeDocument.model_validate(payload)
        if doc.document_id in seen:
            raise ValueError(f"Duplicate document_id at line {line_number}: {doc.document_id}")
        if not doc.content.strip():
            raise ValueError(f"Empty content for document {doc.document_id!r}")
        seen.add(doc.document_id)
        documents.append(doc)
    if not documents:
        raise ValueError(f"No documents found in {path}")
    return tuple(documents)
