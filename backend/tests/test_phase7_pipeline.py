"""Phase 7 integration tests: the two new pipeline stages, end to end.

Only the LLM is faked (same pattern as Phase 4 tests); retrieval,
grounding, validation and execution wiring are real application code.

* Query preprocessing must run BEFORE retrieval/prompting while the API
  contract keeps echoing the original question.
* Result processing must annotate QueryResult with column_kinds without
  changing any existing field's behaviour (Phase 5 contract preserved).
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.rag.llm.base import LLMProvider
from app.rag.retriever import KnowledgeRetriever
from app.rag.text_to_sql import TextToSQLService, load_schema_metadata_snapshot
from app.services.sql_execution import (
    SqlExecutionService,
    get_sql_execution_service,
)

GOOD_SQL = (
    "SELECT d.department_name FROM departments d "
    "WHERE d.department_id <= 2 ORDER BY d.department_name"
)

MESSY_QUESTION = "  Which\u00a0 departments\u2019 ids\u200b are\u2013small ? "
PROCESSED_QUESTION = "Which departments' ids are-small ?"


class FakeProvider(LLMProvider):
    name = "fake"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def is_available(self) -> bool:
        return True

    def generate(self, prompt: str, *, system=None, temperature=0.0, max_tokens=512):
        self.prompts.append(prompt)
        return f"```sql\n{GOOD_SQL};\n```"


class SpyRetriever(KnowledgeRetriever):
    """Records every query handed to retrieve() before delegating."""

    def __init__(self, inner: KnowledgeRetriever) -> None:
        super().__init__(
            embedding_service=inner.embedding_service,
            vector_store=inner.vector_store,
        )
        self.queries: list[str] = []

    def retrieve(self, query: str, top_k: int | None = None):
        self.queries.append(query)
        return super().retrieve(query, top_k=top_k)


@pytest.fixture()
def pipeline_parts(
    rag_retriever,
) -> tuple[TextToSQLService, FakeProvider, SpyRetriever]:
    provider = FakeProvider()
    spy = SpyRetriever(rag_retriever)
    service = TextToSQLService(
        provider=provider,
        retriever=spy,
        schema_metadata=load_schema_metadata_snapshot(),
    )
    return service, provider, spy


# ---------------------------------------------------------------------------
# 1. Service level: preprocessing runs before retrieval + prompting
# ---------------------------------------------------------------------------
def test_retrieval_receives_processed_query(pipeline_parts):
    service, _, spy = pipeline_parts
    result = service.generate(MESSY_QUESTION)

    assert result.sql == GOOD_SQL
    assert len(spy.queries) == 1
    assert spy.queries[0] == result.processed_question
    # The raw typographic noise never reaches retrieval.
    assert "\u00a0" not in spy.queries[0]
    assert "\u2019" not in spy.queries[0]
    assert not spy.queries[0].startswith(" ")
    assert "  " not in spy.queries[0]


def test_prompt_contains_processed_question_not_raw(pipeline_parts):
    service, provider, _ = pipeline_parts
    result = service.generate(MESSY_QUESTION)
    assert provider.prompts, "LLM must have been called"
    assert result.processed_question in provider.prompts[0]
    assert PROCESSED_QUESTION in provider.prompts[0]
    # Raw form (with NBSP / zero-width chars) is absent from the prompt.
    assert "\u200b" not in provider.prompts[0]


def test_original_question_preserved_in_result(pipeline_parts):
    service, _, _ = pipeline_parts
    result = service.generate(MESSY_QUESTION)
    # Contract: original text survives (only outer strip, as in every prior
    # phase); processed form recorded alongside it.
    assert result.question == MESSY_QUESTION.strip()
    assert result.processed_question == PROCESSED_QUESTION


# ---------------------------------------------------------------------------
# 2. Execution level: real service annotates column_kinds
# ---------------------------------------------------------------------------
class _FakeResult:
    def __init__(self, keys, rows):
        self._keys = list(keys)
        self._rows = list(rows)
        self.cursor_description = [
            type("D", (), {"name": k})() for k in self._keys
        ]

    @property
    def keys(self):
        return lambda: list(self._keys)

    def fetchmany(self, size):
        out, self._rows = self._rows[:size], self._rows[size:]
        return [tuple(r) for r in out]

    def close(self):
        pass

    def scalar(self):
        return self._rows[0][0] if self._rows else None


class _SpyConnection:
    def __init__(self):
        self.executed: list[str] = []
        self._result = _FakeResult(
            ["department_name", "budget", "is_active"],
            [("CSE", Decimal("120000"), True), ("ECE", None, False)],
        )

    def execution_options(self, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def execute(self, statement):
        sql_text = str(statement)
        self.executed.append(sql_text)
        if "current_user" in sql_text:
            return _FakeResult(["current_user"], [("schemarag_reader",)])
        return self._result


class _SpyEngine:
    def __init__(self, connection):
        self.connection = connection

    def connect(self):
        outer = self

        class _Ctx:
            def __enter__(self_inner):
                return outer.connection

            def __exit__(self_inner, *exc):
                return False

        return _Ctx()


@pytest.fixture()
def executor() -> tuple[SqlExecutionService, _SpyConnection]:
    from tests.test_phase5_execution import _schema_metadata_stub

    conn = _SpyConnection()
    svc = SqlExecutionService(
        engine=_SpyEngine(conn),  # type: ignore[arg-type]
        max_rows=500,
        statement_timeout_ms=5000,
        schema_metadata=_schema_metadata_stub(),
    )
    return svc, conn


def test_execute_annotates_column_kinds(executor):
    from app.rag.text_to_sql import GeneratedSQL

    svc, conn = executor
    generated = GeneratedSQL(
        question="departments",
        sql=GOOD_SQL,
        model="fake:test",
        retrieved_documents=["schema_departments"],
        retrieval_scores=[0.9],
    )
    result = svc.execute(generated)

    assert result.execution_status == "success"
    assert result.columns == ["department_name", "budget", "is_active"]
    # Phase 7 annotation computed by production execute() path.
    assert result.column_kinds == ["text", "number", "boolean"]
    # Decimal coerced; values themselves untouched otherwise.
    assert result.rows[0][1] == 120000.0
    assert result.rows[1][1] is None
    # Existing Phase 5 contract intact.
    assert result.executed_as == "schemarag_reader"
    assert result.row_count == 2


# ---------------------------------------------------------------------------
# 3. API: additive response fields, original contracts unchanged
# ---------------------------------------------------------------------------
def test_generate_sql_api_echoes_processed_question(
    client: TestClient, pipeline_parts
):
    service, _, _ = pipeline_parts
    from app.rag.text_to_sql import get_sql_generation_service

    app.dependency_overrides[get_sql_generation_service] = lambda: service
    try:
        resp = client.post("/api/generate-sql", json={"question": MESSY_QUESTION})
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    assert body["question"] == MESSY_QUESTION.strip()      # original kept
    assert body["processed_question"] == PROCESSED_QUESTION  # new field
    assert body["grounded"] is True


def test_query_api_returns_column_kinds_on_wire(
    client: TestClient, pipeline_parts, executor
):
    service, _, _ = pipeline_parts
    exec_svc, _ = executor
    from app.rag.text_to_sql import get_sql_generation_service

    app.dependency_overrides[get_sql_generation_service] = lambda: service
    app.dependency_overrides[get_sql_execution_service] = lambda: exec_svc
    try:
        resp = client.post("/api/query", json={"question": MESSY_QUESTION})
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()  # QueryResponse IS the QueryResult payload
    assert body["execution_status"] == "success"
    # Additive Phase 7 field present on the wire.
    assert body["column_kinds"] == ["text", "number", "boolean"]
