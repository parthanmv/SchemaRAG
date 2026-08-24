"""Phase 2 tests: metadata extraction from live PostgreSQL.

These tests intentionally do NOT mock the extractor — they reflect the real
``college_db`` schema and assert the Phase 2 contract.
"""

import pytest

from app.rag.extractor import extract_schema_metadata
from app.rag.validation import (
    EXPECTED_CHECK_CONSTRAINTS,
    EXPECTED_COLUMNS,
    EXPECTED_FOREIGN_KEYS,
    EXPECTED_TABLES,
    EXPECTED_UNIQUE_CONSTRAINTS,
)

pytestmark = pytest.mark.usefixtures("rag_metadata")


# ---------------------------------------------------------------------------
# 1. Tables
# ---------------------------------------------------------------------------
def test_all_expected_tables_discovered(rag_metadata):
    assert set(EXPECTED_TABLES).issubset(set(rag_metadata.table_names))


def test_table_count_matches_contract(rag_metadata):
    # Exactly the six college tables exist in the public schema.
    assert set(rag_metadata.table_names) == set(EXPECTED_TABLES)


# ---------------------------------------------------------------------------
# 2. Columns
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("table_name", sorted(EXPECTED_COLUMNS))
def test_expected_columns_discovered(rag_metadata, table_name):
    table = rag_metadata.get_table(table_name)
    missing = set(EXPECTED_COLUMNS[table_name]) - set(table.column_names)
    assert not missing, f"{table_name} missing columns: {sorted(missing)}"


def test_total_column_count(rag_metadata):
    total = sum(len(t.columns) for t in rag_metadata.tables)
    assert total == sum(len(c) for c in EXPECTED_COLUMNS.values()) == 35


def test_column_data_types_reflected(rag_metadata):
    students = rag_metadata.get_table("students")
    marks = rag_metadata.get_table("marks")
    assert students.column("roll_number").data_type == "varchar(20)"
    assert students.column("semester").data_type == "smallint"
    assert marks.column("marks").data_type == "numeric(5, 2)"
    assert marks.column("exam_type").data_type == "varchar(20)"


def test_nullable_information_reflected(rag_metadata):
    students = rag_metadata.get_table("students")
    assert students.column("name").nullable is False
    assert students.column("department_id").nullable is False


def test_no_duplicate_columns_within_tables(rag_metadata):
    for table in rag_metadata.tables:
        names = [c.name for c in table.columns]
        assert len(names) == len(set(names)), f"duplicate columns in {table.name}"


# ---------------------------------------------------------------------------
# 3. Primary keys
# ---------------------------------------------------------------------------
EXPECTED_PRIMARY_KEYS = {
    "departments": ("department_id",),
    "students": ("student_id",),
    "courses": ("course_id",),
    "enrollments": ("enrollment_id",),
    "marks": ("mark_id",),
    "attendance": ("attendance_id",),
}


@pytest.mark.parametrize("table_name", sorted(EXPECTED_PRIMARY_KEYS))
def test_primary_key_detected(rag_metadata, table_name):
    table = rag_metadata.get_table(table_name)
    expected = EXPECTED_PRIMARY_KEYS[table_name]
    assert tuple(table.primary_keys) == expected
    pk_col = table.column(expected[0])
    assert pk_col.primary_key is True


def test_primary_keys_are_flagged_on_exactly_one_column_each(rag_metadata):
    for table in rag_metadata.tables:
        flagged = [c.name for c in table.columns if c.primary_key]
        assert set(flagged) == set(EXPECTED_PRIMARY_KEYS[table.name])


# ---------------------------------------------------------------------------
# 4-5. Foreign keys / relationships
# ---------------------------------------------------------------------------
def _fk_pairs(metadata):
    return {
        (t.name, col, fk.referred_table, ref_col)
        for t in metadata.tables
        for fk in t.foreign_keys
        for col, ref_col in zip(fk.columns, fk.referred_columns)
    }


def test_all_expected_foreign_keys_detected(rag_metadata):
    discovered = _fk_pairs(rag_metadata)
    missing = set(EXPECTED_FOREIGN_KEYS) - discovered
    assert not missing, f"missing FK relationships: {sorted(missing)}"


def test_foreign_key_count(rag_metadata):
    assert len(rag_metadata.relationships()) == len(EXPECTED_FOREIGN_KEYS)


def test_relationship_representation(rag_metadata):
    students = rag_metadata.get_table("students")
    fk = next(f for f in students.foreign_keys if "department_id" in f.columns)
    assert fk.constraint_name == "students_department_id_fkey"
    assert fk.referred_table == "departments"
    assert fk.referred_columns == ("department_id",)
    assert fk.on_delete == "RESTRICT"


def test_marks_cascade_delete_discovered(rag_metadata):
    marks = rag_metadata.get_table("marks")
    student_fk = next(f for f in marks.foreign_keys if f.columns == ("student_id",))
    assert student_fk.on_delete == "CASCADE"


def test_every_fk_references_existing_table_and_column(rag_metadata):
    tables = {t.name: t for t in rag_metadata.tables}
    for table in rag_metadata.tables:
        for fk in table.foreign_keys:
            assert fk.referred_table in tables
            referred = tables[fk.referred_table]
            for col in fk.referred_columns:
                assert col in referred.column_names


# ---------------------------------------------------------------------------
# 6. Constraints
# ---------------------------------------------------------------------------
def test_expected_check_constraints_extracted(rag_metadata):
    found = {
        ck.name for t in rag_metadata.tables for ck in t.check_constraints
    }
    assert set(EXPECTED_CHECK_CONSTRAINTS).issubset(found)


def test_marks_range_check_expression(rag_metadata):
    marks = rag_metadata.get_table("marks")
    ck = next(c for c in marks.check_constraints if c.name == "ck_marks_range")
    assert "marks >= 0" in ck.expression and "marks <= 100" in ck.expression


def test_attendance_counts_check_prevents_attended_gt_held(rag_metadata):
    attendance = rag_metadata.get_table("attendance")
    ck = next(
        c for c in attendance.check_constraints if c.name == "ck_attendance_counts"
    )
    assert "classes_attended <= classes_held" in ck.expression.replace("  ", " ")


def test_credits_check_constraint_extracted(rag_metadata):
    courses = rag_metadata.get_table("courses")
    ck = next(c for c in courses.check_constraints if c.name == "ck_courses_credits")
    assert "credits >= 1" in ck.expression and "credits <= 6" in ck.expression


def test_expected_unique_constraints_extracted(rag_metadata):
    found = {
        uc.name for t in rag_metadata.tables for uc in t.unique_constraints
    }
    assert set(EXPECTED_UNIQUE_CONSTRAINTS).issubset(found)


def test_unique_roll_number_email_course_code(rag_metadata):
    students = rag_metadata.get_table("students")
    courses = rag_metadata.get_table("courses")
    assert {"roll_number", "email"} == {
        uc.columns[0]
        for uc in students.unique_constraints
        if len(uc.columns) == 1
    }
    assert {"course_code"} == {
        uc.columns[0] for uc in courses.unique_constraints if len(uc.columns) == 1
    }


def test_composite_term_uniqueness_discovered(rag_metadata):
    enrollments = rag_metadata.get_table("enrollments")
    uc = next(u for u in enrollments.unique_constraints
              if u.name == "uq_enrollments_student_course_term")
    assert set(uc.columns) == {"student_id", "course_id", "academic_year", "semester"}


# ---------------------------------------------------------------------------
# Extraction robustness
# ---------------------------------------------------------------------------
def test_extraction_is_rerunnable():
    """Two consecutive extractions produce identical snapshots."""
    first = extract_schema_metadata()
    second = extract_schema_metadata()
    assert first == second


def test_extraction_matches_independent_reflection():
    """Extraction must mirror raw SQLAlchemy reflection, not a static copy."""
    from sqlalchemy import inspect as sqlalchemy_inspect

    from app.db.session import engine

    inspector = sqlalchemy_inspect(engine)
    fresh = extract_schema_metadata()

    assert sorted(fresh.table_names) == sorted(inspector.get_table_names())
    students = fresh.get_table("students")
    assert [c["name"] for c in inspector.get_columns("students")] == list(
        students.column_names
    )
    fk = inspector.get_foreign_keys("marks")
    marks_pairs = {
        (f.columns[0], f.referred_table) for f in fresh.get_table("marks").foreign_keys
    }
    assert {(x["constrained_columns"][0], x["referred_table"]) for x in fk} == (
        marks_pairs
    )
