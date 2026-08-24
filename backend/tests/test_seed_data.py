"""Seed data integrity tests: counts, referential integrity, value ranges, determinism."""

import pytest
from sqlalchemy import text

from app.db.session import engine
from app.scripts.seed_database import (
    EXAM_TYPES,
    SEED,
    dataset_digest,
    generate_dataset,
)

# ---------------------------------------------------------------------------
# Row counts (acceptance criteria)
# ---------------------------------------------------------------------------


def test_departments_seeded(table_counts: dict[str, int]) -> None:
    assert table_counts["departments"] == 8


def test_students_seeded(table_counts: dict[str, int]) -> None:
    assert table_counts["students"] == 1_000


def test_courses_seeded(table_counts: dict[str, int]) -> None:
    assert table_counts["courses"] == 50


def test_relationship_tables_meet_minimum_targets(
    table_counts: dict[str, int],
) -> None:
    assert table_counts["enrollments"] >= 5_000
    assert table_counts["marks"] >= 5_000
    assert table_counts["attendance"] >= 5_000


def test_total_relational_records(table_counts: dict[str, int]) -> None:
    total = sum(table_counts.values())
    assert total >= 15_000


# ---------------------------------------------------------------------------
# Foreign-key / referential integrity (no orphan rows)
# ---------------------------------------------------------------------------


def _scalar(conn, sql: str) -> int:
    return conn.execute(text(sql)).scalar_one()


def test_no_orphan_enrollments() -> None:
    with engine.connect() as conn:
        orphans = _scalar(
            conn,
            """
            SELECT COUNT(*) FROM enrollments e
            LEFT JOIN students s ON s.student_id = e.student_id
            LEFT JOIN courses c ON c.course_id = e.course_id
            WHERE s.student_id IS NULL OR c.course_id IS NULL
            """,
        )
    assert orphans == 0


def test_no_orphan_marks() -> None:
    with engine.connect() as conn:
        orphans = _scalar(
            conn,
            """
            SELECT COUNT(*) FROM marks m
            LEFT JOIN students s ON s.student_id = m.student_id
            LEFT JOIN courses c ON c.course_id = m.course_id
            WHERE s.student_id IS NULL OR c.course_id IS NULL
            """,
        )
    assert orphans == 0


def test_no_orphan_attendance() -> None:
    with engine.connect() as conn:
        orphans = _scalar(
            conn,
            """
            SELECT COUNT(*) FROM attendance a
            LEFT JOIN students s ON s.student_id = a.student_id
            LEFT JOIN courses c ON c.course_id = a.course_id
            WHERE s.student_id IS NULL OR c.course_id IS NULL
            """,
        )
    assert orphans == 0


def test_every_student_belongs_to_valid_department() -> None:
    with engine.connect() as conn:
        invalid = _scalar(
            conn,
            """
            SELECT COUNT(*) FROM students s
            LEFT JOIN departments d ON d.department_id = s.department_id
            WHERE d.department_id IS NULL
            """,
        )
    assert invalid == 0


def test_every_course_belongs_to_valid_department() -> None:
    with engine.connect() as conn:
        invalid = _scalar(
            conn,
            """
            SELECT COUNT(*) FROM courses c
            LEFT JOIN departments d ON d.department_id = c.department_id
            WHERE d.department_id IS NULL
            """,
        )
    assert invalid == 0


def test_marks_reference_enrolled_student_course_pairs() -> None:
    """Every mark must correspond to an actual enrollment of that student."""
    with engine.connect() as conn:
        unenrolled = _scalar(
            conn,
            """
            SELECT COUNT(*) FROM marks m
            WHERE NOT EXISTS (
                SELECT 1 FROM enrollments e
                WHERE e.student_id = m.student_id
                  AND e.course_id = m.course_id
                  AND e.academic_year = m.academic_year
                  AND e.semester = m.semester
            )
            """,
        )
    assert unenrolled == 0


def test_attendance_matches_enrollment_terms() -> None:
    with engine.connect() as conn:
        mismatched = _scalar(
            conn,
            """
            SELECT COUNT(*) FROM attendance a
            WHERE NOT EXISTS (
                SELECT 1 FROM enrollments e
                WHERE e.student_id = a.student_id
                  AND e.course_id = a.course_id
                  AND e.academic_year = a.academic_year
                  AND e.semester = a.semester
            )
            """,
        )
    assert mismatched == 0


# ---------------------------------------------------------------------------
# Value-range and business consistency checks
# ---------------------------------------------------------------------------


def test_marks_within_sensible_range() -> None:
    with engine.connect() as conn:
        out_of_range = _scalar(
            conn, "SELECT COUNT(*) FROM marks WHERE marks < 0 OR marks > 100"
        )
        distinct_exams = {
            row[0]
            for row in conn.execute(text("SELECT DISTINCT exam_type FROM marks"))
        }
    assert out_of_range == 0
    assert distinct_exams <= set(EXAM_TYPES)


def test_attendance_percentage_consistent_with_counts() -> None:
    with engine.connect() as conn:
        inconsistent = _scalar(
            conn,
            """
            SELECT COUNT(*) FROM attendance
            WHERE classes_attended > classes_held
               OR attendance_percentage <> ROUND(classes_attended * 100.0 / NULLIF(classes_held, 0), 2)
            """,
        )
    assert inconsistent == 0


def test_low_attendance_students_exist_for_realistic_queries() -> None:
    """The dataset must contain sub-75% cases for realistic questions."""
    with engine.connect() as conn:
        low = _scalar(
            conn,
            "SELECT COUNT(DISTINCT student_id) FROM attendance "
            "WHERE attendance_percentage < 75",
        )
    assert low > 50


def test_roll_numbers_and_emails_are_unique() -> None:
    with engine.connect() as conn:
        dup_rolls = _scalar(
            conn,
            "SELECT COUNT(*) FROM (SELECT roll_number FROM students "
            "GROUP BY roll_number HAVING COUNT(*) > 1) d",
        )
        dup_emails = _scalar(
            conn,
            "SELECT COUNT(*) FROM (SELECT email FROM students "
            "GROUP BY email HAVING COUNT(*) > 1) d",
        )
    assert dup_rolls == 0
    assert dup_emails == 0


# ---------------------------------------------------------------------------
# Determinism of seed generation
# ---------------------------------------------------------------------------


def test_seed_generation_is_reproducible() -> None:
    first = generate_dataset(SEED)
    second = generate_dataset(SEED)
    assert dataset_digest(first) == dataset_digest(second)


def test_different_seed_produces_different_dataset() -> None:
    default = generate_dataset(SEED)
    other = generate_dataset(SEED + 1)
    assert dataset_digest(default) != dataset_digest(other)


@pytest.mark.parametrize("table", ["students", "courses", "enrollments", "marks", "attendance"])
def test_generated_rows_are_unique_per_table(table: str) -> None:
    ds = generate_dataset(SEED)
    rows = getattr(ds, table)
    keys = [tuple(sorted(r.items())) for r in rows]
    assert len(keys) == len(set(keys)), f"Duplicate rows generated in {table}"
