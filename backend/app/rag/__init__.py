"""RAG knowledge-base pipeline: metadata extraction, document generation, validation.

Sub-modules:
* :mod:`app.rag.extractor`      – reflect the live PostgreSQL schema
* :mod:`app.rag.knowledge`      – turn metadata into RAG documents
* :mod:`app.rag.validation`     – oracle checks against the Phase 1 contract
* :mod:`app.rag.business_rules` / :mod:`app.rag.query_examples` – curated sources
"""

from app.rag.models import KnowledgeDocument, SchemaMetadata
from app.rag.extractor import extract_schema_metadata

__all__ = ["KnowledgeDocument", "SchemaMetadata", "extract_schema_metadata"]
