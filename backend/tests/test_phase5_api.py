"""Phase 5 tests: POST /api/query HTTP contract and error mapping.

Both dependencies (generation + execution) are overridden with fakes so the
HTTP layer can be tested deterministically without an LLM or PostgreSQL.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.rag.text_to_sql import GeneratedSQL
from app.services.sql_execution import (
    ExecutionStatus,
    QueryResult,
    SqlExecutionService,
    get_sql_execution_service,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeGeneration:
    """Mimics TextToSQLService.generate for one canned outcome."""

    def __init__(self, generated=None, error=None):
        self._generated = generated
        self._error = error

    def generate(self, question: str) -> GeneratedSQL:
        if isinstance(self._error, Exception):
            raise self._error
        return (
            self._generated
            if self._generated is not None
            else _generated_ok(question)
        )


def _generated_ok(question="test question?"):
    return GeneratedSQL(
        question=question,
        sql="SELECT name FROM students WHERE semester = 5",
        model="fake:test",
        grounded=True,
        retrieved_documents=["schema_students"],
        retrieval_scores=[0.9],
        issues=[],
        error=None,
    )


class FakeExecutor:
    def __init__(self, result: QueryResult | None = None):
        self.calls: list[GeneratedSQL] = []
        self._result = result or QueryResult(
            question="q",
            sql="SELECT name FROM students WHERE semester = 5",
            model="fake:test",
            grounded=True,
            security_allowed=True,
            security_issues=[],
            execution_status=ExecutionStatus.SUCCESS,
            columns=["name"],
            rows=[["Alice"]],
            row_count=1,
            execution_time_ms=1.0,
            retrieved_documents=["schema_students"],
        )

    def execute(self, generated: GeneratedSQL) -> QueryResult:
        self.calls.append(generated)
        return self._result


def _status_result(status: ExecutionStatus, error: str) -> QueryResult:
    return QueryResult(
        question="q", sql="SELECT 1 FROM departments", model="fake:test",
        grounded=True, security_allowed=status not in (
            ExecutionStatus.SECURITY_REJECTED, ExecutionStatus.UNGROUNDED,
            ExecutionStatus.INVALID_SQL),
        security_issues=[],
        execution_status=status, error=error,
    )


@pytest.fixture()
def api_client(client: TestClient):
    """TestClient with both Phase 5 dependencies replaced."""
    gen, exe = FakeGeneration(), FakeExecutor()
    from app.api.routes.query import get_sql_generation_service as gen_getter
    app.dependency_overrides[gen_getter] = lambda: gen
    app.dependency_overrides[get_sql_execution_service] = lambda: exe
    try:
        yield client, gen, exe
    finally:
        app.dependency_overrides.pop(gen_getter, None)
        app.dependency_overrides.pop(get_sql_execution_service, None)


QUESTION = {"question": "List student names in semester 5"}


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
def test_query_success_returns_rows(api_client):
    c, gen, exe = api_client
    response = c.post("/api/query", json=QUESTION)
    assert response.status_code == 200
    body = response.json()
    assert body["execution_status"] == "success"
    assert body["columns"] == ["name"]
    assert body["rows"] == [["Alice"]]
    assert body["grounded"] is True
    assert body["security_allowed"] is True
    assert len(exe.calls) == 1
    assert exe.calls[0].question == QUESTION["question"]


def test_query_empty_result_is_200(api_client):
    c, _, exe = api_client
    exe._result.execution_status = ExecutionStatus.EMPTY
    exe._result.rows, exe._result.row_count = [], 0
    response = c.post("/api/query", json=QUESTION)
    assert response.status_code == 200
    assert response.json()["execution_status"] == "empty_result"


def test_query_row_limit_is_200_with_status_flag(api_client):
    c, _, exe = api_client
    exe._result.execution_status = ExecutionStatus.ROW_LIMIT_EXCEEDED
    exe._result.error = "result truncated at 500 rows"
    response = c.post("/api/query", json=QUESTION)
    assert response.status_code == 200
    assert response.json()["execution_status"] == "row_limit_exceeded"


# ---------------------------------------------------------------------------
# Generation-side failures
# ---------------------------------------------------------------------------
def test_query_llm_unavailable_503(api_client):
    c, gen, _ = api_client
    gen._error = RuntimeError("provider down")  # any exception type; route maps LLMUnavailableError below
    from app.rag.llm.base import LLMUnavailableError
    gen._error = LLMUnavailableError("no provider configured")
    response = c.post("/api/query", json=QUESTION)
    assert response.status_code == 503


def test_query_llm_error_502(api_client):
    c, gen, _ = api_client
    from app.rag.llm.base import LLMError
    gen._error = LLMError("bad JSON from provider")
    response = c.post("/api/query", json=QUESTION)
    assert response.status_code == 502


def test_query_no_sql_generated_400(api_client):
    c, gen, _ = api_client
    gen._generated = GeneratedSQL(
        question="q", sql=None, model="fake:test", grounded=False,
        retrieved_documents=[], retrieval_scores=[], issues=[],
        error="insufficient_context",
    )
    response = c.post("/api/query", json=QUESTION)
    assert response.status_code == 400
    assert "insufficient_context" in response.json()["detail"]["error"]


def test_query_ungrounded_sql_400(api_client):
    c, gen, exe = api_client
    gen._generated = GeneratedSQL(
        question="q", sql="SELECT potion FROM wizards", model="fake:test",
        grounded=False, retrieved_documents=[], retrieval_scores=[],
        issues=[], error=None,
    )
    response = c.post("/api/query", json=QUESTION)
    assert response.status_code == 400
    assert exe.calls == [], "executor ran despite ungrounded SQL"


# ---------------------------------------------------------------------------
# Execution-side failures -> mapped HTTP codes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (ExecutionStatus.SECURITY_REJECTED, 403),
        (ExecutionStatus.STATEMENT_TIMEOUT, 504),
        (ExecutionStatus.CONNECTION_ERROR, 503),
        (ExecutionStatus.DISABLED, 503),
        (ExecutionStatus.PERMISSION_DENIED, 503),
        (ExecutionStatus.EXECUTION_ERROR, 500),
    ],
)
def test_query_execution_failure_mapping(api_client, status, expected):
    c, _, exe = api_client
    exe._result = _status_result(status, f"synthetic {status.value}")
    if expected != 403:
        # Non-security failures must still have passed grounding upstream.
        pass
    response = c.post("/api/query", json=QUESTION)
    assert response.status_code == expected
    if expected == 403:
        detail = response.json()["detail"]
        assert detail["error"] == "security_rejected"


def test_query_request_too_short_422(api_client):
    c, *_ = api_client
    response = c.post("/api/query", json={"question": "hi"})
    assert response.status_code == 422
