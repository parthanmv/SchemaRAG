"""Phase 2 tests: end-to-end build pipeline, artifacts, and determinism.

Runs the real pipeline (reflection -> generation -> validation -> write)
against the live database into temporary directories.
"""

import hashlib
import json

import pytest

from app.rag.knowledge import KnowledgeGenerator
from app.rag.models import DocumentType
from app.scripts.extract_metadata import (
    serialize_documents,
    serialize_metadata,
    run_build,
)


@pytest.fixture(scope="module")
def first_build(tmp_path_factory):
    return run_build(output_dir=tmp_path_factory.mktemp("rag_run1"))


def test_pipeline_writes_both_artifacts(first_build):
    from pathlib import Path

    metadata_path = Path(first_build.schema_metadata_path)
    jsonl_path = Path(first_build.knowledge_jsonl_path)
    assert metadata_path.is_file()
    assert jsonl_path.is_file()
    assert jsonl_path.parent.name == "documents"
    assert metadata_path.parent.name == "metadata"


def test_artifact_counts_match_documents(first_build):
    total = sum(first_build.document_counts.values())
    assert total == first_build.total_documents == 38
    assert first_build.document_counts[DocumentType.SCHEMA.value] == 6
    assert first_build.document_counts[DocumentType.RELATIONSHIP.value] == 8
    assert first_build.document_counts[DocumentType.BUSINESS_RULE.value] == 3
    assert first_build.document_counts[DocumentType.QUERY_EXAMPLE.value] == 5


def test_knowledge_jsonl_is_valid_jsonl_with_metadata(first_build):
    from pathlib import Path

    lines = Path(first_build.knowledge_jsonl_path).read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(lines) == 38
    ids = []
    for line in lines:
        payload = json.loads(line)
        for key in ("document_id", "document_type", "tables", "source", "content"):
            assert key in payload, f"missing {key}"
        ids.append(payload["document_id"])
    assert ids == sorted(ids)


def test_schema_metadata_json_is_valid_and_complete(first_build):
    from pathlib import Path

    payload = json.loads(
        Path(first_build.schema_metadata_path).read_text(encoding="utf-8")
    )
    names = [t["name"] for t in payload["tables"]]
    assert set(names) == {
        "departments", "students", "courses",
        "enrollments", "marks", "attendance",
    }
    students = next(t for t in payload["tables"] if t["name"] == "students")
    assert {"student_id", "roll_number", "email"} <= {
        c["name"] for c in students["columns"]
    }


def test_output_is_deterministic_across_reruns(tmp_path):
    """Rerunning the whole pipeline with an unchanged schema is byte-stable."""
    run1_dir = tmp_path / "run1"
    run2_dir = tmp_path / "run2"
    artifacts1 = run_build(output_dir=run1_dir)
    artifacts2 = run_build(output_dir=run2_dir)

    assert artifacts1.digest == artifacts2.digest

    jsonl_bytes_1 = (run1_dir / "documents" / "knowledge.jsonl").read_bytes()
    jsonl_bytes_2 = (run2_dir / "documents" / "knowledge.jsonl").read_bytes()
    meta_bytes_1 = (run1_dir / "metadata" / "schema_metadata.json").read_bytes()
    meta_bytes_2 = (run2_dir / "metadata" / "schema_metadata.json").read_bytes()

    assert jsonl_bytes_1 == jsonl_bytes_2
    assert meta_bytes_1 == meta_bytes_2
    # The recorded digest really is the sha256 of the JSONL file.
    assert hashlib.sha256(jsonl_bytes_1).hexdigest() == artifacts1.digest


def test_rerun_overwrites_existing_artifacts_safely(tmp_path):
    """Re-extraction into the same directory succeeds (idempotent)."""
    target = tmp_path / "rag"
    run_build(output_dir=target)
    first = (target / "documents" / "knowledge.jsonl").read_bytes()
    run_build(output_dir=target)
    second = (target / "documents" / "knowledge.jsonl").read_bytes()
    assert first == second


def test_serializers_are_stable(rag_metadata, rag_documents):
    """Serialisation of identical inputs yields identical strings."""
    documents_again = KnowledgeGenerator().generate(rag_metadata)
    assert serialize_documents(rag_documents) == serialize_documents(documents_again)
    assert serialize_metadata(rag_metadata) == serialize_metadata(rag_metadata)


def test_serialized_documents_have_no_timestamps_or_secrets(rag_documents):
    text = serialize_documents(rag_documents).lower()
    for forbidden in ("password=", "postgres://", "postgresql://", "generated_at"):
        assert forbidden not in text
