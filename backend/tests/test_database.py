"""Database connectivity, schema and constraint tests against real PostgreSQL."""

import pytest
from sqlalchemy import inspect, text

from app.db.base import Base
from app.db.session import engine

REQUIRED_TABLES = {
    "departments", "students", "courses", "enrollments", "marks", "attendance",
}


def test_database_connectivity() -> None:
    with engine.connect() as conn:
        assert conn.execute(text("SELECT 1")).scalar_one() == 1


def test_database_is_college_db() -> None:
    with engine.connect() as conn:
        name = conn.execute(text("SELECT current_database()")).scalar_one()
    assert name == "college_db"


def test_all_required_tables_exist() -> None:
    inspector = inspect(engine)
    existing = set(inspector.get_table_names())
    missing = REQUIRED_TABLES - existing
    assert not missing, f"Missing tables: {sorted(missing)}"


def test_orm_metadata_matches_database_tables() -> None:
    inspector = inspect(engine)
    for table in Base.metadata.sorted_tables:
        db_columns = {c["name"] for c in inspector.get_columns(table.name)}
        model_columns = {c.name for c in table.columns}
        assert model_columns <= db_columns, (
            f"Table {table.name}: columns missing in DB: "
            f"{sorted(model_columns - db_columns)}"
        )


def test_foreign_keys_are_configured() -> None:
    """Every relationship table must reference its parent tables."""
    inspector = inspect(engine)
    fks = {
        table: {(fk["referred_table"], tuple(sorted(fk["constrained_columns"])))
                for fk in inspector.get_foreign_keys(table)}
        for table in ("students", "courses", "enrollments", "marks", "attendance")
    }

    assert ("departments", ("department_id",)) in fks["students"]
    assert ("departments", ("department_id",)) in fks["courses"]
    assert ("students", ("student_id",)) in fks["enrollments"]
    assert ("courses", ("course_id",)) in fks["enrollments"]
    assert ("students", ("student_id",)) in fks["marks"]
    assert ("courses", ("course_id",)) in fks["marks"]
    assert ("students", ("student_id",)) in fks["attendance"]
    assert ("courses", ("course_id",)) in fks["attendance"]


def test_check_constraints_exist() -> None:
    inspector = inspect(engine)
    marks_checks = " ".join(
        c["sqltext"] for c in inspector.get_check_constraints("marks")
    ).lower()
    attendance_checks = " ".join(
        c["sqltext"] for c in inspector.get_check_constraints("attendance")
    ).lower()

    assert "100" in marks_checks          # ck_marks_range upper bound
    assert "classes_attended" in attendance_checks
