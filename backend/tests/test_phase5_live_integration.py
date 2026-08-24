"""Phase 5 LIVE integration tests against real PostgreSQL.

Everything runs through the dedicated ``schemarag_reader`` role using the
EXEC_DB_* configuration - never the application/admin credentials. Write
refusals are probed with statements that cannot mutate data even in principle
(permission checks precede execution; TRUNCATE/DDL attempts are rolled back).

These tests skip automatically when the execution role is not configured
(e.g. fresh checkouts without .env), keeping the suite hermetic elsewhere.
"""

from __future__ import annotations

import pytest

from app.core.config import get_settings
from app.rag.models import SchemaMetadata  # noqa: F401 (typing clarity)
from app.rag.text_to_sql import GeneratedSQL, load_schema_metadata_snapshot
from app.services.sql_execution import (
    ExecutionStatus,
    SqlExecutionService,
    create_execution_engine,
)

EXPECTED_COUNTS = {
    "departments": 8,
    "students": 1000,
    "courses": 50,
    "enrollments": 5049,
    "marks": 15147,
    "attendance": 5049,
}


def _exec_creds():
    s = get_settings()
    pw = s.exec_db_password
    pw = pw.get_secret_value() if hasattr(pw, "get_secret_value") else pw
    return s.exec_db_user, pw


@pytest.fixture(scope="module")
def exec_settings():
    user, password = _exec_creds()
    if not user or not password:
        pytest.skip("EXEC_DB_USER/EXEC_DB_PASSWORD not configured")
    return get_settings(), user, password


@pytest.fixture(scope="module")
def live_service(exec_settings):
    settings, user, password = exec_settings
    engine = create_execution_engine(user, password)
    yield SqlExecutionService(
        engine=engine, schema_metadata=load_schema_metadata_snapshot()
    )
    engine.dispose()


def _g(sql: str, question: str = "live probe") -> GeneratedSQL:
    return GeneratedSQL(
        question=question, sql=sql, model="live:test", grounded=True,
        retrieved_documents=[], retrieval_scores=[], issues=[], error=None,
    )


# ---------------------------------------------------------------------------
# Execution through the reader role
# ---------------------------------------------------------------------------
def test_executes_as_reader_role(live_service):
    result = live_service.execute(_g("SELECT COUNT(*) AS n FROM students"))
    assert result.execution_status is ExecutionStatus.SUCCESS
    assert result.executed_as == "schemarag_reader"
    assert result.rows == [[EXPECTED_COUNTS["students"]]]


def test_admin_credentials_are_not_the_execution_role(exec_settings):
    settings, user, _password = exec_settings
    assert user == "schemarag_reader"
    assert user != settings.db_user  # application role never used for execution


def test_empty_result_status(live_service):
    result = live_service.execute(_g("SELECT name FROM students WHERE false"))
    assert result.execution_status is ExecutionStatus.EMPTY
    assert result.row_count == 0


def test_row_limit_truncation_live(live_service):
    limited = SqlExecutionService(
        engine=live_service._engine_or_none(),
        max_rows=5,
        schema_metadata=load_schema_metadata_snapshot(),
    )
    result = limited.execute(_g("SELECT student_id FROM students"))
    assert result.execution_status is ExecutionStatus.ROW_LIMIT_EXCEEDED
    assert result.row_count == 5
    assert "truncated" in (result.error or "")


def test_connection_recovery_after_error(live_service):
    """A runtime DB error must not poison the pooled connection."""
    # Grounded SQL with a genuine execution-time failure (division by zero).
    bad = live_service.execute(_g("SELECT department_id / 0 FROM departments"))
    assert bad.execution_status is ExecutionStatus.EXECUTION_ERROR

    good = live_service.execute(_g("SELECT COUNT(*) AS n FROM departments"))
    assert good.execution_status is ExecutionStatus.SUCCESS
    assert good.rows == [[EXPECTED_COUNTS["departments"]]]


def test_statement_timeout_enforced_by_server(exec_settings):
    """Server-side statement_timeout cancels a runaway recursive CTE."""
    settings, user, password = exec_settings
    engine = create_execution_engine(
        user, password, statement_timeout_ms=800
    )
    try:
        svc = SqlExecutionService(engine=engine)
        runaway = (
            "WITH RECURSIVE t(n) AS (SELECT 1::int UNION ALL SELECT n+1 FROM t) "
            "SELECT count(*) FROM t"
        )
        status, _cols, rows, error, executed_as = svc._run(runaway)
        assert status is ExecutionStatus.STATEMENT_TIMEOUT
        assert rows == []
        assert "timeout" in (error or "").lower()
        assert executed_as is None
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# Database-level refusal of writes (harmless probes, always rolled back)
# ---------------------------------------------------------------------------
WRITE_PROBES = [
    ("UPDATE students SET student_id = student_id WHERE false", "update"),
    ("DELETE FROM students WHERE false", "delete"),
    ("INSERT INTO students DEFAULT VALUES", "insert"),
    ("TRUNCATE TABLE students", "truncate"),
    ("ALTER TABLE students ADD COLUMN __probe int", "alter"),
    ("DROP TABLE courses", "drop"),
    ("CREATE TABLE public.__sr_probe (id int)", "create"),
]


@pytest.mark.parametrize("sql,label", WRITE_PROBES, ids=[p[1] for p in WRITE_PROBES])
def test_writer_probes_refused_at_database_level(exec_settings, sql, label):
    import psycopg

    settings, user, password = exec_settings
    conn = psycopg.connect(
        host=settings.db_host, port=settings.db_port, dbname=settings.db_name,
        user=user, password=password, connect_timeout=5,
    )
    try:
        with conn.cursor() as cur:
            with pytest.raises(psycopg.Error) as excinfo:
                cur.execute(sql)
            conn.rollback()
            refused_properly = isinstance(
                excinfo.value,
                (psycopg.errors.InsufficientPrivilege, psycopg.errors.ReadOnlySqlTransaction),
            )
            msg = str(excinfo.value).lower()
            assert refused_properly or "read-only" in msg or "permission denied" in msg, (
                f"{label} refused for the wrong reason: {msg}"
            )
    finally:
        conn.close()


def test_row_counts_unchanged_after_all_probes(live_service):
    for table, expected in EXPECTED_COUNTS.items():
        result = live_service.execute(_g(f"SELECT COUNT(*) AS n FROM {table}"))
        assert result.execution_status is ExecutionStatus.SUCCESS, table
        assert result.rows == [[expected]], table
