"""Phase 2 tests: knowledge document generation and validation.

Documents are generated from live-reflected metadata (see ``rag_documents``
fixture in conftest.py); no part of the pipeline is mocked.
"""

import json

import pytest

from app.rag.models import DocumentSource, DocumentType, KnowledgeDocument
from app.rag.validation import (
    EXPECTED_TABLES,
    validate_all,
    validate_documents,
    validate_metadata,
)


@pytest.fixture()
def docs_by_id(rag_documents) -> dict[str, KnowledgeDocument]:
    return {d.document_id: d for d in rag_documents}


# ---------------------------------------------------------------------------
# 7. Documents are generated
# ---------------------------------------------------------------------------
def test_every_document_type_is_generated(rag_documents):
    types = {d.document_type for d in rag_documents}
    assert types == set(DocumentType)


def test_document_counts(rag_documents):
    counts: dict[DocumentType, int] = {}
    for doc in rag_documents:
        counts[doc.document_type] = counts.get(doc.document_type, 0) + 1
    assert counts[DocumentType.SCHEMA] == 6
    assert counts[DocumentType.RELATIONSHIP] == 8
    assert counts[DocumentType.CONSTRAINT] >= 16  # 8 checks + 8 uniques
    assert counts[DocumentType.BUSINESS_RULE] == 3
    assert counts[DocumentType.QUERY_EXAMPLE] == 5


def test_one_schema_document_per_table(rag_metadata, rag_documents):
    schema_ids = {
        d.document_id for d in rag_documents if d.document_type is DocumentType.SCHEMA
    }
    assert schema_ids == {f"schema_{t}" for t in EXPECTED_TABLES}


def test_schema_document_lists_all_columns(rag_metadata, docs_by_id):
    doc = docs_by_id["schema_students"]
    students = rag_metadata.get_table("students")
    for col_name in students.column_names:
        assert f"- {col_name}: " in doc.content


def test_schema_document_marks_primary_key_and_fk(rag_metadata, docs_by_id):
    content = docs_by_id["schema_students"].content
    assert "- student_id: integer, primary key" in content
    assert "references departments.department_id" in content


# ---------------------------------------------------------------------------
# 8-9. Documents reference only real database objects
# ---------------------------------------------------------------------------
def test_schema_documents_reference_real_tables(rag_metadata, rag_documents):
    table_names = set(rag_metadata.table_names)
    for doc in rag_documents:
        if doc.document_type is DocumentType.SCHEMA:
            assert len(doc.tables) == 1
            assert doc.tables[0] in table_names


def test_relationship_documents_reference_real_objects(rag_metadata, rag_documents):
    tables = {t.name: t for t in rag_metadata.tables}
    relationship_docs = [
        d for d in rag_documents if d.document_type is DocumentType.RELATIONSHIP
    ]
    assert len(relationship_docs) == 8
    for doc in relationship_docs:
        assert set(doc.tables).issubset(tables), doc.document_id
        key = doc.extra["relationship_key"]
        src_cols, dst = [part.strip() for part in key.split(" -> ")]
        src_table = src_cols.split(".")[0]
        dst_table, dst_col = dst.rsplit(".", 1)
        assert src_table in tables and dst_table in tables
        assert src_cols.split(".")[1] in tables[src_table].column_names
        assert dst_col in tables[dst_table].column_names


def test_constraint_documents_reference_real_constraints(rag_metadata, rag_documents):
    checks = {c.name for t in rag_metadata.tables for c in t.check_constraints}
    uniques = {u.name for t in rag_metadata.tables for u in t.unique_constraints}
    for doc in rag_documents:
        if doc.document_type is not DocumentType.CONSTRAINT:
            continue
        name = doc.extra["constraint_name"]
        kind = doc.extra["constraint_kind"]
        pool = checks if kind == "check" else uniques
        assert name in pool, f"{doc.document_id} references unknown constraint"


def test_no_document_mentions_nonexistent_table_names(rag_metadata, rag_documents):
    known = set(EXPECTED_TABLES) | set(rag_metadata.table_names)
    for doc in rag_documents:
        assert set(doc.tables).issubset(known), doc.document_id


# ---------------------------------------------------------------------------
# 10. Document metadata validity
# ---------------------------------------------------------------------------
REQUIRED_METADATA_KEYS = {"document_id", "document_type", "tables", "source"}


def test_every_document_has_required_metadata(rag_documents):
    seen_ids = set()
    for doc in rag_documents:
        payload = json.loads(doc.model_dump_json())
        assert REQUIRED_METADATA_KEYS.issubset(payload.keys()), doc.document_id
        assert payload["document_type"] in {t.value for t in DocumentType}
        assert payload["source"] in {s.value for s in DocumentSource}
        assert isinstance(payload["tables"], list)
        assert doc.content.strip(), f"{doc.document_id} has empty content"
        assert doc.document_id not in seen_ids, f"duplicate id {doc.document_id}"
        seen_ids.add(doc.document_id)


def test_source_matches_document_type(rag_documents):
    expected_source = {
        DocumentType.SCHEMA: DocumentSource.POSTGRESQL_METADATA,
        DocumentType.RELATIONSHIP: DocumentSource.POSTGRESQL_METADATA,
        DocumentType.CONSTRAINT: DocumentSource.POSTGRESQL_METADATA,
        DocumentType.BUSINESS_RULE: DocumentSource.DOMAIN_RULES,
        DocumentType.QUERY_EXAMPLE: DocumentSource.CURATED_EXAMPLES,
    }
    for doc in rag_documents:
        assert doc.source == expected_source[doc.document_type], doc.document_id


def test_documents_sorted_by_id_for_deterministic_output(rag_documents):
    ids = [d.document_id for d in rag_documents]
    assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# 11. Business rules vs database constraints
# ---------------------------------------------------------------------------
def test_business_rules_use_domain_source_not_postgresql(rag_documents):
    rules = [d for d in rag_documents if d.document_type is DocumentType.BUSINESS_RULE]
    assert len(rules) == 3
    for doc in rules:
        assert doc.source is DocumentSource.DOMAIN_RULES
        assert doc.content.startswith("Business rule (domain source):")


def test_unenforced_business_rule_states_it_clearly(docs_by_id):
    doc = docs_by_id["business_rule_pass_mark"]
    assert "NOT enforced by any PostgreSQL constraint" in doc.content
    assert doc.extra["enforced_by_postgresql"] == "no"
    assert doc.extra["backing_check_constraint"] == ""


def test_backed_business_rule_references_existing_constraint(rag_metadata, docs_by_id):
    doc = docs_by_id["business_rule_marks_scale"]
    assert doc.extra["enforced_by_postgresql"] == "yes"
    backing = doc.extra["backing_check_constraint"]
    assert backing == "ck_marks_range"
    marks = rag_metadata.get_table("marks")
    assert any(c.name == backing for c in marks.check_constraints)


# ---------------------------------------------------------------------------
# 12. Query examples reference valid tables
# ---------------------------------------------------------------------------
EXPECTED_EXAMPLE_QUESTIONS = {
    "Which department has the highest average marks?",
    "Which students have attendance below 75%?",
    "Show the top 5 students by marks.",
    "Which departments have an average mark above 75?",
    "Which students scored above their department average?",
}


def test_curated_query_examples_present(rag_documents):
    questions = {
        d.extra["question"]
        for d in rag_documents
        if d.document_type is DocumentType.QUERY_EXAMPLE
    }
    assert questions == EXPECTED_EXAMPLE_QUESTIONS


def test_query_examples_reference_valid_tables_with_concepts(rag_metadata, rag_documents):
    tables = set(rag_metadata.table_names)
    examples = [
        d for d in rag_documents if d.document_type is DocumentType.QUERY_EXAMPLE
    ]
    for doc in examples:
        assert set(doc.tables).issubset(tables), doc.document_id
        assert doc.extra["concepts"], f"{doc.document_id} lacks concepts"
        # The question text itself must appear in the embeddable content.
        assert doc.extra["question"] in doc.content


# ---------------------------------------------------------------------------
# Validators as a whole
# ---------------------------------------------------------------------------
def test_full_validation_passes_against_live_database(rag_metadata, rag_documents):
    report = validate_all(rag_metadata, rag_documents)
    assert report.ok, [i.message for i in report.errors]
    assert not report.warnings, [i.message for i in report.warnings]


def test_validator_rejects_nonexistent_table_reference(rag_metadata, rag_documents):
    from app.rag.models import KnowledgeDocument

    forged = KnowledgeDocument(
        document_id="schema_ghost",
        document_type=DocumentType.SCHEMA,
        title="ghost",
        content="The ghost table stores nothing.",
        tables=("ghost",),
        source=DocumentSource.POSTGRESQL_METADATA,
        extra={"table": "ghost"},
    )
    report = validate_documents((*rag_documents, forged), rag_metadata)
    assert not report.ok
    assert any("nonexistent table 'ghost'" in i.message for i in report.errors)


def test_metadata_validator_rejects_missing_expected_fk():
    from app.rag.models import (
        CheckConstraintInfo,
        ColumnInfo,
        ForeignKeyInfo,
        SchemaMetadata,
        TableInfo,
        UniqueConstraintInfo,
    )

    # Valid snapshot minus the courses.department_id foreign key.
    departments = TableInfo(
        name="departments",
        columns=(
            ColumnInfo(name="department_id", data_type="integer", nullable=False,
                       primary_key=True),
            ColumnInfo(name="department_name", data_type="text", nullable=False),
            ColumnInfo(name="department_code", data_type="varchar(10)", nullable=False),
        ),
        primary_keys=("department_id",),
        unique_constraints=(
            UniqueConstraintInfo(name="departments_department_name_key",
                                 columns=("department_name",)),
            UniqueConstraintInfo(name="departments_department_code_key",
                                 columns=("department_code",)),
        ),
    )
    students = TableInfo(
        name="students",
        columns=tuple(
            ColumnInfo(name=n, data_type="integer", nullable=False, primary_key=(n == "student_id"))
            if n in ("student_id", "department_id")
            else ColumnInfo(name=n, data_type="text", nullable=False)
            for n in ("student_id", "name", "department_id")
        ),
        primary_keys=("student_id",),
        foreign_keys=(
            ForeignKeyInfo(
                constraint_name="students_department_id_fkey",
                columns=("department_id",),
                referred_table="departments",
                referred_columns=("department_id",),
            ),
        ),
        check_constraints=(CheckConstraintInfo(name="ck_x", expression="1 > 0"),),
    )
    broken = SchemaMetadata(
        tables=(
            departments,
            students,
            # every other table omitted on purpose -> many errors expected
        )
    )
    report = validate_metadata(broken)
    assert not report.ok
    messages = " | ".join(i.message for i in report.errors)
    assert "Expected table not found" in messages
    assert "courses" in messages
    assert "attendance" in messages
