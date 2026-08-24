"""Validation of extracted metadata and generated knowledge documents.

``validate_metadata`` cross-checks a reflected snapshot against the known
Phase 1 college-database contract (tables/columns/keys/constraints). The
expectations below are *test oracle* values only — they never feed the
extractor, which discovers everything from PostgreSQL.

``validate_documents`` guarantees generated documents only ever reference
real database objects and carry complete, correctly-typed provenance.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from app.rag import domain_glossary
from app.rag.models import (
    DocumentSource,
    DocumentType,
    KnowledgeDocument,
    SchemaMetadata,
    TableInfo,
    ValidationIssue,
)

# ---------------------------------------------------------------------------
# Phase 1 database contract (validation oracle, not extractor input)
# ---------------------------------------------------------------------------
EXPECTED_TABLES: tuple[str, ...] = (
    "attendance",
    "courses",
    "departments",
    "enrollments",
    "marks",
    "students",
)

EXPECTED_COLUMNS: dict[str, tuple[str, ...]] = {
    "departments": ("department_id", "department_name", "department_code"),
    "students": (
        "student_id", "roll_number", "name", "email",
        "department_id", "semester", "admission_year",
    ),
    "courses": ("course_id", "course_code", "course_name", "credits", "department_id"),
    "enrollments": ("enrollment_id", "student_id", "course_id", "academic_year", "semester"),
    "marks": (
        "mark_id", "student_id", "course_id", "exam_type",
        "marks", "academic_year", "semester",
    ),
    "attendance": (
        "attendance_id", "student_id", "course_id", "classes_held",
        "classes_attended", "attendance_percentage", "academic_year", "semester",
    ),
}

EXPECTED_FOREIGN_KEYS: tuple[tuple[str, str, str, str], ...] = (
    # (table, column, referred_table, referred_column)
    ("students", "department_id", "departments", "department_id"),
    ("courses", "department_id", "departments", "department_id"),
    ("enrollments", "student_id", "students", "student_id"),
    ("enrollments", "course_id", "courses", "course_id"),
    ("marks", "student_id", "students", "student_id"),
    ("marks", "course_id", "courses", "course_id"),
    ("attendance", "student_id", "students", "student_id"),
    ("attendance", "course_id", "courses", "course_id"),
)

EXPECTED_CHECK_CONSTRAINTS: tuple[str, ...] = (
    "ck_students_semester",
    "ck_students_admission_year",
    "ck_marks_range",
    "ck_marks_semester",
    "ck_attendance_counts",
    "ck_attendance_percentage",
    "ck_courses_credits",
    "ck_enrollments_semester",
)

EXPECTED_UNIQUE_CONSTRAINTS: tuple[str, ...] = (
    "departments_department_name_key",
    "departments_department_code_key",
    "students_roll_number_key",
    "students_email_key",
    "courses_course_code_key",
    "uq_enrollments_student_course_term",
    "uq_marks_student_course_exam_term",
    "uq_attendance_student_course_term",
)

_ALLOWED_SOURCES: dict[DocumentType, frozenset[DocumentSource]] = {
    DocumentType.SCHEMA: frozenset({DocumentSource.POSTGRESQL_METADATA}),
    DocumentType.RELATIONSHIP: frozenset({DocumentSource.POSTGRESQL_METADATA}),
    DocumentType.CONSTRAINT: frozenset({DocumentSource.POSTGRESQL_METADATA}),
    DocumentType.BUSINESS_RULE: frozenset({DocumentSource.DOMAIN_RULES}),
    DocumentType.QUERY_EXAMPLE: frozenset({DocumentSource.CURATED_EXAMPLES}),
}


class ValidationReport(BaseModel):
    """Aggregated result of validating metadata and/or documents."""

    issues: tuple[ValidationIssue, ...] = ()

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(i for i in self.issues if i.severity == "error")

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(i for i in self.issues if i.severity == "warning")

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def summary(self) -> str:
        return (
            f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)"
            f"{' - PASSED' if self.ok else ' - FAILED'}"
        )


def _error(message: str) -> ValidationIssue:
    return ValidationIssue(severity="error", message=message)


def _warning(message: str) -> ValidationIssue:
    return ValidationIssue(severity="warning", message=message)


def validate_metadata(metadata: SchemaMetadata) -> ValidationReport:
    """Validate a reflected snapshot against the Phase 1 contract."""
    issues: list[ValidationIssue] = []
    discovered_tables = set(metadata.table_names)

    missing_tables = sorted(set(EXPECTED_TABLES) - discovered_tables)
    for name in missing_tables:
        issues.append(_error(f"Expected table not found in PostgreSQL: {name!r}"))

    unexpected_tables = sorted(discovered_tables - set(EXPECTED_TABLES))
    for name in unexpected_tables:
        issues.append(
            _warning(f"Unexpected table discovered in public schema: {name!r}")
        )

    discovered_fk_pairs: set[tuple[str, str, str, str]] = set()
    discovered_check_names: set[str] = set()
    discovered_unique_names: set[str] = set()

    for table in metadata.tables:
        # Columns --------------------------------------------------------
        actual_cols = set(table.column_names)
        for col_name in EXPECTED_COLUMNS.get(table.name, ()):
            if col_name not in actual_cols:
                issues.append(
                    _error(f"Table {table.name!r} is missing expected column {col_name!r}")
                )
        # Primary keys ---------------------------------------------------
        if not table.primary_keys:
            issues.append(_error(f"Table {table.name!r} has no detected primary key"))
        pk_missing_columns = [
            c for c in table.primary_keys if c not in actual_cols
        ]
        for col_name in pk_missing_columns:
            issues.append(
                _error(
                    f"Primary key of {table.name!r} references unknown "
                    f"column {col_name!r}"
                )
            )
        # Foreign keys ---------------------------------------------------
        for fk in table.foreign_keys:
            if fk.referred_table not in discovered_tables:
                issues.append(
                    _error(
                        f"Foreign key on {table.name!r}.{fk.columns} references "
                        f"unknown table {fk.referred_table!r}"
                    )
                )
            for col in (*fk.columns, *fk.referred_columns):
                owner = table.name if col in fk.columns else fk.referred_table
                if owner == table.name and col not in actual_cols:
                    issues.append(
                        _error(
                            f"Foreign key on {table.name!r} uses unknown column {col!r}"
                        )
                    )
            if len(fk.columns) == 1 and len(fk.referred_columns) == 1:
                discovered_fk_pairs.add(
                    (
                        table.name,
                        fk.columns[0],
                        fk.referred_table,
                        fk.referred_columns[0],
                    )
                )
        # Constraints ----------------------------------------------------
        for ck in table.check_constraints:
            if ck.name:
                discovered_check_names.add(ck.name)
        for uc in table.unique_constraints:
            if uc.name:
                discovered_unique_names.add(uc.name)
            unknown = [c for c in uc.columns if c not in actual_cols]
            for col_name in unknown:
                issues.append(
                    _error(
                        f"Unique constraint {uc.name!r} on {table.name!r} uses "
                        f"unknown column {col_name!r}"
                    )
                )

    for pair in EXPECTED_FOREIGN_KEYS:
        if pair not in discovered_fk_pairs:
            issues.append(_error(f"Expected foreign-key relationship not found: {pair}"))
    for name in EXPECTED_CHECK_CONSTRAINTS:
        if name not in discovered_check_names:
            issues.append(_error(f"Expected CHECK constraint not found: {name!r}"))
    for name in EXPECTED_UNIQUE_CONSTRAINTS:
        if name not in discovered_unique_names:
            issues.append(_error(f"Expected UNIQUE constraint not found: {name!r}"))

    return ValidationReport(issues=tuple(issues))


def validate_documents(
    documents: tuple[KnowledgeDocument, ...], metadata: SchemaMetadata
) -> ValidationReport:
    """Validate generated documents against the metadata snapshot."""
    issues: list[ValidationIssue] = []
    tables = {t.name: t for t in metadata.tables}
    seen_ids: set[str] = set()

    for doc in documents:
        did = doc.document_id
        if did in seen_ids:
            issues.append(_error(f"Duplicate document_id: {did!r}"))
        seen_ids.add(did)

        if not doc.content.strip():
            issues.append(_error(f"Document {did!r} has empty content"))
        allowed = _ALLOWED_SOURCES.get(doc.document_type, frozenset())
        if doc.source not in allowed:
            issues.append(
                _error(
                    f"Document {did!r} ({doc.document_type.value}) has illegal "
                    f"source {doc.source.value!r}; allowed: "
                    f"{sorted(s.value for s in allowed)}"
                )
            )

        unknown_tables = [t for t in doc.tables if t not in tables]
        for table_name in unknown_tables:
            issues.append(
                _error(f"Document {did!r} references nonexistent table {table_name!r}")
            )

        if doc.document_type is DocumentType.SCHEMA:
            for table_name in doc.tables:
                table = tables.get(table_name)
                if table is None:
                    continue  # already reported as nonexistent above
                if doc.extra.get("table") != table.name:
                    issues.append(
                        _error(f"Schema document {did!r} has inconsistent table metadata")
                    )
                mentioned = _mentioned_columns(doc.content, table)
                for col_name in mentioned - set(table.column_names):
                    issues.append(
                        _error(
                            f"Schema document {did!r} mentions unknown column "
                            f"{col_name!r} of {table_name!r}"
                        )
                    )

        elif doc.document_type is DocumentType.RELATIONSHIP:
            key = doc.extra.get("relationship_key", "")
            matched = any(
                fk.relationship_key(t.name) == key
                for t in metadata.tables
                for fk in t.foreign_keys
            )
            if not matched:
                issues.append(
                    _error(
                        f"Relationship document {did!r} does not match any "
                        f"discovered foreign key"
                    )
                )

        elif doc.document_type is DocumentType.CONSTRAINT:
            table = tables.get(doc.tables[0]) if doc.tables else None
            if table is None:
                continue
            cname = doc.extra.get("constraint_name", "")
            kind = doc.extra.get("constraint_kind")
            known = (
                {ck.name for ck in table.check_constraints}
                if kind == "check"
                else {uc.name for uc in table.unique_constraints}
            )
            if cname not in known:
                issues.append(
                    _error(
                        f"Constraint document {did!r} references unknown "
                        f"{kind} constraint {cname!r} on {table.name!r}"
                    )
                )

        elif doc.document_type is DocumentType.BUSINESS_RULE:
            backing = doc.extra.get("backing_check_constraint") or None
            if backing is None:
                if doc.extra.get("enforced_by_postgresql") != "no":
                    issues.append(
                        _error(
                            f"Business rule {did!r} without backing constraint "
                            f"must be marked unenforced"
                        )
                    )
            elif not any(
                ck.name == backing
                for table in metadata.tables
                for ck in table.check_constraints
            ):
                issues.append(
                    _error(
                        f"Business rule {did!r} claims backing CHECK "
                        f"constraint {backing!r} which does not exist in PostgreSQL"
                    )
                )

        elif doc.document_type is DocumentType.QUERY_EXAMPLE:
            if not doc.extra.get("question"):
                issues.append(_error(f"Query example {did!r} has no question text"))

    # Glossary hygiene -----------------------------------------------------
    for key in domain_glossary.TABLE_DESCRIPTIONS:
        if key not in tables:
            issues.append(
                _warning(f"Glossary describes table {key!r} which does not exist")
            )
    known_pairs = {
        f"{t.name}.{c}" for t in metadata.tables for c in t.column_names
    }
    for key in domain_glossary.COLUMN_DESCRIPTIONS:
        if key not in known_pairs:
            issues.append(
                _warning(f"Glossary describes column {key!r} which does not exist")
            )

    return ValidationReport(issues=tuple(issues))


_COLUMN_LINE_RE = re.compile(r"- ([a-z_][a-z0-9_]*): ")


def _mentioned_columns(content: str, table: TableInfo) -> set[str]:
    """Extract column names listed in a schema document's ``Columns:`` block."""
    names: set[str] = set()
    in_columns_block = False
    for line in content.splitlines():
        if line.strip() == "Columns:":
            in_columns_block = True
            continue
        if in_columns_block:
            match = _COLUMN_LINE_RE.match(line)
            if match:
                names.add(match.group(1))
            elif line.startswith("- ") or line == "":
                break
    return names


def validate_all(
    metadata: SchemaMetadata, documents: tuple[KnowledgeDocument, ...]
) -> ValidationReport:
    """Run both validators and merge their reports."""
    metadata_report = validate_metadata(metadata)
    documents_report = validate_documents(documents, metadata)
    return ValidationReport(issues=metadata_report.issues + documents_report.issues)
