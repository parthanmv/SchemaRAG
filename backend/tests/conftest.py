"""Shared pytest fixtures for SchemaRAG tests.

Tests run against the *actual* configured PostgreSQL environment (the same
``college_db`` used by the application) as required by the project specs.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.db.session import SessionLocal, engine
from app.main import app
from app.rag.extractor import extract_schema_metadata
from app.rag.knowledge import KnowledgeGenerator
from app.rag.models import KnowledgeDocument, SchemaMetadata


@pytest.fixture(scope="session")
def client() -> TestClient:
    """HTTP client bound to the FastAPI application."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def db_session():
    """A short-lived database session rolled back after each test."""
    session = SessionLocal()
    try:
        yield session
        session.rollback()
    finally:
        session.close()


@pytest.fixture(scope="session")
def table_counts() -> dict[str, int]:
    """Row counts of every seeded table, queried once per test session."""
    tables = (
        "departments", "courses", "students",
        "enrollments", "marks", "attendance",
    )
    with engine.connect() as conn:
        return {
            t: conn.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar_one()
            for t in tables
        }


# ---------------------------------------------------------------------------
# Phase 2 fixtures: live metadata extraction (never mocked)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def rag_metadata() -> SchemaMetadata:
    """Metadata reflected once per test session from the live database."""
    return extract_schema_metadata()


@pytest.fixture(scope="session")
def rag_documents(rag_metadata: SchemaMetadata) -> tuple[KnowledgeDocument, ...]:
    """Knowledge documents generated from the reflected metadata."""
    return KnowledgeGenerator().generate(rag_metadata)


# ---------------------------------------------------------------------------
# Phase 3 fixtures: shared real embedding model + FAISS store (never mocked)
# ---------------------------------------------------------------------------
import numpy as np  # noqa: E402

from app.rag.embeddings import EmbeddingService  # noqa: E402
from app.rag.retriever import KnowledgeRetriever  # noqa: E402
from app.rag.vector_store import FaissVectorStore, embedding_text_for  # noqa: E402


@pytest.fixture(scope="session")
def embedder() -> EmbeddingService:
    """One shared embedding service for the whole session (model loads once)."""
    return EmbeddingService()


@pytest.fixture(scope="session")
def rag_store_dir(tmp_path_factory, rag_documents, embedder):
    """Directory holding a saved FAISS store built from live documents."""
    directory = tmp_path_factory.mktemp("rag_index")
    embeddings = embedder.embed_documents(
        [embedding_text_for(doc) for doc in rag_documents]
    )
    store = FaissVectorStore.build(
        documents=rag_documents,
        embeddings=embeddings,
        embedding_model=embedder.model_name,
        knowledge_sha256="deadbeef" * 8,
    )
    store.save(directory)
    return directory


@pytest.fixture(scope="session")
def rag_store(rag_store_dir) -> FaissVectorStore:
    return FaissVectorStore.load(rag_store_dir)


@pytest.fixture(scope="session")
def rag_retriever(rag_store, embedder) -> KnowledgeRetriever:
    return KnowledgeRetriever(embedding_service=embedder, vector_store=rag_store)
