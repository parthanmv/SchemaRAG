"""Reproducible knowledge-base build pipeline.

Run from the ``backend`` directory:

    python -m app.scripts.extract_metadata

Steps:
1. Connect to PostgreSQL and reflect the actual schema.
2. Validate the extracted metadata against the Phase 1 contract.
3. Generate RAG knowledge documents.
4. Validate the generated documents against the metadata.
5. Write ``rag/metadata/schema_metadata.json`` and
   ``rag/documents/knowledge.jsonl`` (deterministic, byte-stable).
6. Print a concise summary including a SHA-256 digest of the JSONL output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys

from pydantic import BaseModel

from app.core.config import get_settings
from app.rag.extractor import extract_schema_metadata
from app.rag.knowledge import KnowledgeGenerator
from app.rag.models import DocumentType
from app.rag.validation import validate_all

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger("extract_metadata")


class BuildArtifacts(BaseModel):
    """Paths and counts produced by one pipeline run."""

    schema_metadata_path: str
    knowledge_jsonl_path: str
    document_counts: dict[str, int]
    total_documents: int
    digest: str


def serialize_metadata(metadata) -> str:  # noqa: ANN001 - SchemaMetadata
    """Deterministic pretty-printed JSON for the metadata snapshot."""
    return json.dumps(
        metadata.model_dump(mode="json"),
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
    ) + "\n"


def serialize_documents(documents) -> str:  # noqa: ANN001 - tuple[KnowledgeDocument, ...]
    """Deterministic JSONL payload (one document per line, id-sorted)."""
    lines = [
        json.dumps(doc.model_dump(mode="json"), sort_keys=True, ensure_ascii=True)
        for doc in sorted(documents, key=lambda d: d.document_id)
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def write_artifacts(output_dir, metadata, documents) -> BuildArtifacts:  # noqa: ANN001
    """Write both artifacts atomically-ish (write then verify digest)."""
    documents_dir = output_dir / "documents"
    metadata_dir = output_dir / "metadata"
    documents_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = metadata_dir / "schema_metadata.json"
    jsonl_path = documents_dir / "knowledge.jsonl"

    metadata_text = serialize_metadata(metadata)
    documents_text = serialize_documents(documents)
    metadata_path.write_text(metadata_text, encoding="utf-8", newline="\n")
    jsonl_path.write_text(documents_text, encoding="utf-8", newline="\n")

    counts: dict[str, int] = {}
    for doc in documents:
        key = doc.document_type.value
        counts[key] = counts.get(key, 0) + 1

    return BuildArtifacts(
        schema_metadata_path=str(metadata_path),
        knowledge_jsonl_path=str(jsonl_path),
        document_counts=dict(sorted(counts.items())),
        total_documents=len(documents),
        digest=hashlib.sha256(documents_text.encode("utf-8")).hexdigest(),
    )


def run_build(output_dir=None) -> BuildArtifacts:  # noqa: ANN001 - Path | None
    """Extract -> generate -> validate -> write; raises on validation failure."""
    settings = get_settings()
    target_dir = output_dir if output_dir is not None else settings.rag_output_dir

    logger.info("Reflecting PostgreSQL schema ...")
    metadata = extract_schema_metadata()
    logger.info("Discovered %d table(s).", len(metadata.tables))

    generator = KnowledgeGenerator()
    documents = generator.generate(metadata)

    report = validate_all(metadata, documents)
    if report.warnings:
        for issue in report.warnings:
            logger.warning("%s", issue.message)
    if not report.ok:
        for issue in report.errors:
            logger.error("%s", issue.message)
        raise RuntimeError(
            f"Knowledge base validation failed: {report.summary}"
        )
    logger.info("Validation passed: %s", report.summary)

    artifacts = write_artifacts(target_dir, metadata, documents)
    logger.info("Metadata snapshot : %s", artifacts.schema_metadata_path)
    logger.info("Knowledge JSONL   : %s", artifacts.knowledge_jsonl_path)
    logger.info(
        "Documents by type : %s",
        ", ".join(f"{k}={v}" for k, v in artifacts.document_counts.items()),
    )
    logger.info("Total documents   : %d", artifacts.total_documents)
    logger.info("JSONL sha256      : %s", artifacts.digest)
    return artifacts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the SchemaRAG RAG knowledge base from live PostgreSQL metadata."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override the output directory (default: <repo>/rag).",
    )
    args = parser.parse_args(argv)

    from pathlib import Path

    override = Path(args.output_dir) if args.output_dir else None
    run_build(override)
    logger.info("Knowledge base build completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
