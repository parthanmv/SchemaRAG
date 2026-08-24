"""Build the FAISS vector index from the Phase 2 knowledge base.

Run from the ``backend`` directory:

    python -m app.scripts.build_vector_index

Steps: load knowledge.jsonl -> structurally validate -> embed (batched,
normalised, model loaded once) -> build FAISS IndexFlatIP -> save
index.faiss + document_store.json (with a freshness manifest) -> print a
summary. Safe to rerun; output is deterministic for unchanged inputs.
"""

from __future__ import annotations

import argparse
import logging
import sys

from app.core.config import get_settings
from app.rag.embeddings import EmbeddingService
from app.rag.vector_store import (
    FaissVectorStore,
    embedding_text_for,
    load_knowledge_documents,
    sha256_of_file,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger("build_vector_index")


def build_index(index_dir=None, knowledge_path=None) -> FaissVectorStore:  # noqa: ANN001
    """Full build pipeline; returns the built store."""
    settings = get_settings()
    knowledge_path = knowledge_path or (
        settings.rag_output_dir / "documents" / "knowledge.jsonl"
    )
    index_dir = index_dir or settings.rag_index_dir

    documents = load_knowledge_documents(knowledge_path)
    digest = sha256_of_file(knowledge_path)
    logger.info("Loaded %d knowledge document(s) from %s", len(documents), knowledge_path)

    service = EmbeddingService()
    logger.info(
        "Embedding with %s (dimension %d) ...", service.model_name, service.dimension
    )
    embeddings = service.embed_documents([embedding_text_for(doc) for doc in documents])

    store = FaissVectorStore.build(
        documents=documents,
        embeddings=embeddings,
        embedding_model=service.model_name,
        knowledge_sha256=digest,
    )
    index_path, store_path = store.save(index_dir)

    print("Index build summary")
    print(f"  Documents        : {len(documents)}")
    print(f"  Embedding model  : {service.model_name}")
    print(f"  Embedding dim    : {service.dimension}")
    print(f"  FAISS vectors    : {store.size}")
    print(f"  Index type       : {store.manifest.faiss_index_type} "
          f"({store.manifest.to_dict()['similarity_metric']})")
    print(f"  Knowledge sha256 : {digest[:16]}...")
    print(f"  Output           : {index_dir}")
    logger.info("Files: %s, %s", index_path.name, store_path.name)
    return store


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Embed the RAG knowledge base and build the FAISS index."
    )
    parser.add_argument(
        "--index-dir", type=str, default=None,
        help="Override index output directory (default: <repo>/rag/index).",
    )
    args = parser.parse_args(argv)

    from pathlib import Path

    override = Path(args.index_dir) if args.index_dir else None
    build_index(index_dir=override)
    return 0


if __name__ == "__main__":
    sys.exit(main())
