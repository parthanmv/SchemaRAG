"""Local sentence-transformer embedding service.

The model is loaded once per :class:`EmbeddingService` instance; the
process-wide instance is available via :func:`get_embedding_service`. All
vectors are L2-normalised so cosine similarity equals the inner product,
matching the FAISS ``IndexFlatIP`` used by the vector store.

No external embedding API is involved: the model runs locally and no API
keys are required. The embedding dimension is discovered from the model —
never hardcoded.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np

# Cosmetic: keep model-loading quiet and silence the Windows symlink notice.
import logging
import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from sentence_transformers import SentenceTransformer


def _discover_model_dimension(model: SentenceTransformer) -> int:
    """Discover dimension across sentence-transformers versions."""
    getter = getattr(model, "get_embedding_dimension", None)
    if getter is None:  # pre-6.0 name
        getter = model.get_sentence_embedding_dimension
    return int(getter())  # noqa: E402

from app.core.config import get_settings


class EmbeddingService:
    """Thin, reusable wrapper around a local SentenceTransformer model."""

    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self._batch_size = settings.embedding_batch_size
        # eval-only inference on CPU; deterministic for a fixed input.
        self._model: SentenceTransformer = SentenceTransformer(self.model_name)
        dimension = _discover_model_dimension(self._model)
        if not dimension or dimension <= 0:
            raise RuntimeError(
                f"Could not discover embedding dimension for model {self.model_name!r}"
            )
        self.dimension: int = int(dimension)

    # ------------------------------------------------------------------
    # Embedding API
    # ------------------------------------------------------------------
    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed many texts in one batched call; returns (n, dim) float32, L2-normalised."""
        if len(texts) == 0:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_document(self, text: str) -> np.ndarray:
        """Embed one document text; returns (dim,) float32, L2-normalised."""
        return self.embed_documents([text])[0]

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a user query; returns (dim,) float32, L2-normalised."""
        return self.embed_documents([query])[0]


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """Process-wide cached embedding service (model loaded once)."""
    return EmbeddingService()
