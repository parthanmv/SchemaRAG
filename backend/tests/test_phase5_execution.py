"""Phase 5 tests: SQL execution service gates, limits and error mapping.

The database is faked with a spy engine/connection for the gate tests so we
can *prove* that invalid/ungrounded/security-rejected SQL never reaches
PostgreSQL. Timeout/permission/row-limit paths are simulated by raising the
real SQLAlchemy exception types. Live read-only-role verification lives in
``test_phase5_live_role.py`` (skips when execution credentials are absent).
"""

from __future__ import annotations

import pytest
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.sql.elements import TextClause

from app.rag.sql_security import SQLSecurityValidator
from app.rag.text_to_sql import GeneratedSQL
from app.services.sql_execution import (
    ExecutionStatus,
    QueryResult,
    SqlExecutionService,
)


# ---------------------------------------------------------------------------
# Fakes / spies
# ---------------------------------------------------------------------------
class _FakeResult:
    def __init__(self, columns, rows):
        self._columns = columns
        self._rows = rows

    def keys(self):
        return list(self._columns)

    def fetchmany(self, size):
        return [tuple(r) for r in self._rows[:size]]

    def scalar(self):
        return self._rows[0][0] if self._rows else None

    def close(self):
        pass


class SpyConnection:
    """Records every statement it is asked to execute."""

    def __init__(self, result=None, error=None):
        self.executed: list[str] = []
        self._result = result or _FakeResult(["name"], [("Alice",)])
        self._error = error
        self.execution_options_used = None

    def execution_options(self, **kwargs):
        self.execution_options_used = kwargs
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def execute(self, statement):
        assert isinstance(statement, TextClause), "must be a TextClause"
        sql_text = str(statement)
        self.executed.append(sql_text)
        if "current_user" in sql_text:
            # Identity probe issued on the same connection after the query.
            return _FakeResult(["current_user"], [("schemarag_reader",)])
        if self._error is not None:
            raise self._error
        return self._result


class SpyEngine:
    """Engine stand-in that hands out one shared spy connection."""

    def __init__(self, connection: SpyConnection):
        self.connection = connection

    def connect(self):
        # Each connect() yields a fresh recorder sharing the same list.
        outer = self

        class _Ctx:
            def __enter__(self_inner):
                return outer.connection

            def __exit__(self_inner, *exc):
                return False

        return _Ctx()


def _generated(
    sql: str | None,
    *,
    grounded: bool = True,
    error: str | None = None,
    question: str = "test question?",
) -> GeneratedSQL:
    return GeneratedSQL(
        question=question,
        sql=sql,
        model="fake:test",
        grounded=grounded,
        retrieved_documents=["schema_students"],
        retrieval_scores=[0.9],
        issues=[],
        error=error,
    )


@pytest.fixture()
def spy_connection() -> SpyConnection:
    return SpyConnection()


@pytest.fixture()
def service(spy_connection: SpyConnection) -> SqlExecutionService:
    return SqlExecutionService(
        engine=SpyEngine(spy_connection),  # type: ignore[arg-type]
        max_rows=500,
        statement_timeout_ms=5000,
        schema_metadata=_schema_metadata_stub(),
    )


def _schema_metadata_stub():
    from app.rag.models import CheckConstraintInfo, ColumnInfo, SchemaMetadata, TableInfo

    students = TableInfo(
        name="students",
        columns=(
            ColumnInfo(name="student_id", data_type="integer", nullable=False),
            ColumnInfo(name="name", data_type="character varying", nullable=False),
            ColumnInfo(name="semester", data_type="smallint", nullable=False),
        ),
        primary_keys=("student_id",),
    )
    departments = TableInfo(
        name="departments",
        columns=(
            ColumnInfo(name="department_id", data_type="integer", nullable=False),
            ColumnInfo(name="department_name", data_type="text", nullable=False),
        ),
        primary_keys=("department_id",),
    )
    marks = TableInfo(
        name="marks",
        columns=(
            ColumnInfo(name="mark_id", data_type="integer", nullable=False),
            ColumnInfo(name="marks", data_type="numeric", nullable=False),
        ),
        primary_keys=("mark_id",),
        check_constraints=(CheckConstraintInfo(name="ck_marks_range", expression="marks >= 0"),),
    )
    return SchemaMetadata(tables=(students, departments, marks))


GOOD_SQL = "SELECT name FROM students WHERE semester = 5"


# ---------------------------------------------------------------------------
# Gates: SQL that must NEVER reach PostgreSQL
# ---------------------------------------------------------------------------
def test_generation_error_never_executes(service, spy_connection):
    result = service.execute(_generated(None, error="invalid_response"))
    assert result.execution_status is ExecutionStatus.INVALID_SQL
    assert result.security_allowed is False
    assert spy_connection.executed == []


def test_insufficient_context_never_executes(service, spy_connection):
    result = service.execute(_generated(None, error="insufficient_context"))
    assert result.execution_status is ExecutionStatus.INVALID_SQL
    assert spy_connection.executed == []


def test_empty_sql_never_executes(service, spy_connection):
    result = service.execute(_generated("   "))
    assert result.execution_status is ExecutionStatus.INVALID_SQL
    assert spy_connection.executed == []


def test_ungrounded_sql_never_executes(service, spy_connection):
    result = service.execute(_generated("SELECT potion FROM wizards"))
    assert result.execution_status is ExecutionStatus.UNGROUNDED
    assert result.grounded is False
    assert any("wizards" in i for i in result.security_issues)
    assert spy_connection.executed == []


def test_unknown_column_never_executes(service, spy_connection):
    result = service.execute(_generated("SELECT gpa FROM students"))
    assert result.execution_status is ExecutionStatus.UNGROUNDED
    assert spy_connection.executed == []


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM students",
        "INSERT INTO students VALUES (1)",
        "UPDATE students SET name = 'x'",
        "DROP TABLE students",
        # pg_catalog is unknown to the schema snapshot, so the *grounding* gate
        # rejects it first; either way it must never reach PostgreSQL.
        "SELECT * FROM pg_catalog.pg_tables",
        # pg_sleep parses as grounded-ish? No - it has no FROM clause, so it
        # passes grounding and must be stopped by the security validator.
        "SELECT pg_sleep(5)",
        "SELECT 1; DELETE FROM students",
        "SELECT * FROM students -- comment",
    ],
)
def test_security_rejected_sql_never_executes(service, spy_connection, sql):
    result = service.execute(_generated(sql))
    assert result.execution_status in (
        ExecutionStatus.SECURITY_REJECTED,
        ExecutionStatus.UNGROUNDED,
    )
    assert result.security_allowed is False
    assert result.rows == []
    assert spy_connection.executed == [], "rejected SQL reached the DB!"


def test_valid_select_reaches_database_exactly_once(service, spy_connection):
    result = service.execute(_generated(GOOD_SQL))
    assert result.execution_status is ExecutionStatus.SUCCESS
    assert spy_connection.executed[0] == GOOD_SQL
    assert spy_connection.executed[1] == "SELECT current_user"
    assert result.columns == ["name"]
    assert result.rows == [["Alice"]]
    assert result.row_count == 1


def test_result_reports_executing_role(service, spy_connection):
    """The identity probe on the same connection proves the execution user."""
    result = service.execute(_generated(GOOD_SQL))
    assert result.executed_as == "schemarag_reader"


def test_streaming_and_limits_requested(service, spy_connection):
    service.execute(_generated(GOOD_SQL))
    assert spy_connection.execution_options_used is not None
    assert spy_connection.execution_options_used.get("stream_results") is True


# ---------------------------------------------------------------------------
# Result handling
# ---------------------------------------------------------------------------
def test_empty_result_status(service, spy_connection):
    spy_connection._result = _FakeResult(["n"], [])
    result = service.execute(_generated("SELECT COUNT(*) AS n FROM marks WHERE 1 = 0"))
    assert result.execution_status is ExecutionStatus.EMPTY
    assert result.rows == []
    assert result.row_count == 0
    assert result.error is None


def test_row_limit_truncates_cleanly(spy_connection):
    rows = [(i,) for i in range(100)]
    spy_connection._result = _FakeResult(["id"], rows)
    svc = SqlExecutionService(
        engine=SpyEngine(spy_connection),  # type: ignore[arg-type]
        max_rows=10,
        statement_timeout_ms=5000,
        schema_metadata=_schema_metadata_stub(),
    )
    result = svc.execute(_generated("SELECT student_id AS id FROM students"))
    assert result.execution_status is ExecutionStatus.ROW_LIMIT_EXCEEDED
    assert result.row_count == 10
    assert len(result.rows) == 10
    assert result.error is not None and "truncated" in result.error


def test_statement_timeout_mapped(service, spy_connection):
    orig = type("Orig", (), {"sqlstate": "57014"})()
    err = OperationalError("stmt", {}, orig)
    spy_connection._error = err
    result = service.execute(_generated(GOOD_SQL))
    assert result.execution_status is ExecutionStatus.STATEMENT_TIMEOUT
    assert "timeout" in (result.error or "").lower()
    assert result.rows == []


def test_permission_denied_mapped(service, spy_connection):
    orig = type("Orig", (), {"sqlstate": "42501"})()
    spy_connection._error = OperationalError("stmt", {}, orig)
    result = service.execute(_generated(GOOD_SQL))
    assert result.execution_status is ExecutionStatus.PERMISSION_DENIED


def test_generic_dbapi_error_sanitized(service, spy_connection):
    orig = type("Orig", (), {"sqlstate": "42601"})()  # syntax error
    spy_connection._error = OperationalError("stmt", {}, orig)
    result = service.execute(_generated(GOOD_SQL))
    assert result.execution_status is ExecutionStatus.EXECUTION_ERROR
    # No raw driver text leaks into the message.
    assert result.error == "PostgreSQL rejected the statement"


# ---------------------------------------------------------------------------
# Typed-model hygiene
# ---------------------------------------------------------------------------
def test_query_result_model_has_all_spec_fields():
    fields = set(QueryResult.model_fields)
    required = {
        "question", "sql", "model", "grounded", "security_allowed",
        "security_issues", "execution_status", "columns", "rows",
        "row_count", "execution_time_ms", "retrieved_documents",
    }
    assert required <= fields


def test_result_json_safe_with_decimals_and_dates(service, spy_connection):
    from datetime import date
    from decimal import Decimal

    spy_connection._result = _FakeResult(
        ["avg_marks", "day"], [(Decimal("88.50"), date(2025, 1, 1))]
    )
    result = service.execute(_generated("SELECT AVG(marks) AS avg_marks FROM marks"))
    payload = result.model_dump(mode="json")
    assert payload["rows"] == [[88.50, "2025-01-01"]]
