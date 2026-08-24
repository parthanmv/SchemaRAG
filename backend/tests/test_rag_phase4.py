"""Phase 4 tests: LLM providers, context assembly, prompting, SQL extraction,
grounding, orchestration, API, and evaluation plumbing.

Unit tests run against a FakeProvider (no LLM/API access required). The live
end-to-end test uses the real Google Gemini API and is *skipped* unless
LLM_PROVIDER=gemini and GEMINI_API_KEY are configured. Retrieval/embedding/
index components are the real Phase 3 stack.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.rag.context import ContextAssembler
from app.core.config import get_settings
from app.rag.grounding import ground_sql
from app.rag.llm.base import LLMProvider, LLMUnavailableError, create_provider
from app.rag.prompts import INSTRUCTION, build_system_message, build_text_to_sql_prompt
from app.rag.retriever import RetrievalResult
from app.rag.sql_evaluation import (
    EVALUATION_QUESTIONS,
    concept_coverage,
    relationship_coverage,
)
from app.rag.sql_parsing import (
    InvalidSQLResponseError,
    extract_sql,
    validate_single_select,
)
from app.rag.text_to_sql import (
    GeneratedSQL,
    TextToSQLService,
    get_sql_generation_service,
    load_schema_metadata_snapshot,
)

GOOD_JOIN_SQL = (
    "SELECT d.department_name, AVG(m.marks) AS avg_marks "
    "FROM departments d "
    "JOIN students s ON s.department_id = d.department_id "
    "JOIN marks m ON m.student_id = s.student_id "
    "GROUP BY d.department_name ORDER BY avg_marks DESC LIMIT 1"
)


# ---------------------------------------------------------------------------
# Helpers / fakes (only the LLM is faked; RAG + grounding are real)
# ---------------------------------------------------------------------------
class FakeProvider(LLMProvider):
    name = "fake"

    def __init__(self, response: str = "```sql\nSELECT 1\n```", available: bool = True):
        self.response = response
        self.available = available
        self.prompts: list[str] = []

    def is_available(self) -> bool:
        return self.available

    def generate(self, prompt: str, *, system=None, temperature=0.0, max_tokens=512) -> str:
        self.prompts.append(prompt)
        return self.response


def _result(doc_id: str, doc_type: str, score: float = 0.5, content: str = "c",
            tables: tuple[str, ...] = ("students",)) -> RetrievalResult:
    return RetrievalResult(
        document_id=doc_id, score=score, document_type=doc_type,
        source="postgresql_metadata", tables=list(tables), content=content,
    )


@pytest.fixture()
def fake_provider() -> FakeProvider:
    return FakeProvider(response=f"```sql\n{GOOD_JOIN_SQL};\n```")


@pytest.fixture()
def service(fake_provider, rag_retriever) -> TextToSQLService:
    """Real retriever/assembler/grounding; only the LLM call is faked."""
    return TextToSQLService(
        provider=fake_provider,
        retriever=rag_retriever,
        schema_metadata=load_schema_metadata_snapshot(),
    )


# ---------------------------------------------------------------------------
# 1. LLM configuration / provider factory
# ---------------------------------------------------------------------------
def test_create_ollama_provider_from_config():
    provider = create_provider("ollama", "test-model")
    assert provider.name == "ollama"
    assert provider.model == "test-model"


def test_create_gemini_provider_from_config():
    from app.rag.llm.gemini import GeminiProvider

    provider = create_provider("gemini", "gemini-2.5-flash")
    assert isinstance(provider, GeminiProvider)
    assert provider.name == "gemini"
    assert provider.model == "gemini-2.5-flash"


def test_unknown_provider_raises():
    with pytest.raises(LLMUnavailableError):
        create_provider("does-not-exist", "x")


def test_ollama_availability_false_for_dead_server():
    from app.rag.llm.ollama import OllamaProvider

    dead = OllamaProvider(model="whatever", base_url="http://127.0.0.1:9", timeout_seconds=0.2)
    assert dead.is_available() is False


# ---------------------------------------------------------------------------
# 2-3. Context assembly: grouping, dedupe, priority, budget
# ---------------------------------------------------------------------------
def test_context_groups_documents_by_type():
    assembler = ContextAssembler(max_chars=10_000)
    ctx = assembler.assemble([
        _result("schema_students", "schema"),
        _result("relationship_x", "relationship"),
        _result("constraint_x", "constraint"),
        _result("business_rule_x", "business_rule"),
        _result("query_example_x", "query_example"),
    ])
    assert set(ctx.sections) == {"schema", "relationship", "constraint",
                                 "business_rule", "query_example"}


def test_context_priority_order_schema_first_examples_last():
    assembler = ContextAssembler(max_chars=10_000)
    ctx = assembler.assemble([
        _result("query_example_a", "query_example"),
        _result("schema_a", "schema"),
        _result("constraint_a", "constraint"),
        _result("relationship_a", "relationship"),
        _result("business_rule_a", "business_rule"),
    ])
    types_in_order = [t for t in ("schema", "relationship", "constraint",
                                  "business_rule", "query_example")
                      if t in ctx.sections]
    assert types_in_order == ["schema", "relationship", "constraint",
                              "business_rule", "query_example"]


def test_context_removes_duplicates_keeping_best_score():
    assembler = ContextAssembler(max_chars=10_000)
    ctx = assembler.assemble([
        _result("schema_students", "schema", score=0.4),
        _result("schema_students", "schema", score=0.9),
    ])
    assert ctx.used_documents == ("schema_students",)


def test_context_respects_char_budget_and_reports_drops():
    assembler = ContextAssembler(max_chars=120)
    long_doc = _result("schema_students", "schema", content="x" * 100)
    filler = _result("query_example_q", "query_example", content="y" * 50)
    ctx = assembler.assemble([filler, long_doc])
    assert "schema_students" in ctx.used_documents  # higher priority wins
    assert "query_example_q" in ctx.dropped_documents


def test_context_never_invents_content():
    assembler = ContextAssembler(max_chars=5_000)
    original = "The marks table stores exam results."
    ctx = assembler.assemble([_result("schema_marks", "schema", content=original)])
    rendered = ctx.render()
    assert original in rendered
    assert len(ctx.used_documents) == 1


# ---------------------------------------------------------------------------
# 4. Prompt construction
# ---------------------------------------------------------------------------
REQUIRED_HEADERS = (
    "=== DATABASE SCHEMA ===", "=== RELATIONSHIPS ===", "=== CONSTRAINTS ===",
    "=== BUSINESS RULES ===", "=== QUERY EXAMPLES ===", "=== USER QUESTION ===",
)


def test_prompt_contains_all_required_sections():
    ctx = ContextAssembler(max_chars=10_000).assemble(
        [_result("schema_students", "schema")]
    )
    prompt = build_text_to_sql_prompt("Which students failed?", ctx)
    for header in REQUIRED_HEADERS:
        assert header in prompt
    assert INSTRUCTION in prompt
    assert "Which students failed?" in prompt.split("=== USER QUESTION ===")[1]


def test_prompt_with_empty_context_still_has_all_sections():
    ctx = ContextAssembler().assemble([])
    prompt = build_text_to_sql_prompt("q?", ctx)
    for header in REQUIRED_HEADERS:
        assert header in prompt
    assert "(no retrieved documents)" in prompt


def test_system_message_is_read_only():
    message = build_system_message().lower()
    assert "select" in message and "never execute" in message


def test_service_passes_grounded_prompt_to_llm(service, fake_provider):
    service.generate(GOOD_JOIN_SQL.replace("'", "") or "average marks per department")
    # The question actually sent through generate():
    prompt = fake_provider.prompts[0]
    assert "=== USER QUESTION ===" in prompt
    assert "DATABASE SCHEMA" in prompt
    assert "students" in prompt  # real retrieved schema reached the model


# ---------------------------------------------------------------------------
# 5-6. SQL extraction & invalid responses
# ---------------------------------------------------------------------------
def test_extract_fenced_sql():
    sql = extract_sql("Here you go:\n```sql\nSELECT 1 FROM students;\n```\nDone.")
    assert sql == "SELECT 1 FROM students"


def test_extract_bare_sql_strips_semicolon():
    assert extract_sql("SELECT * FROM students;") == "SELECT * FROM students"


def test_extract_accepts_cte_queries():
    sql = extract_sql("WITH x AS (SELECT 1 AS a) SELECT a FROM x;")
    assert sql.startswith("WITH")


def test_extract_rejects_empty_response():
    with pytest.raises(InvalidSQLResponseError):
        extract_sql("   ")


def test_extract_rejects_non_sql_text():
    with pytest.raises(InvalidSQLResponseError, match="does not look like"):
        extract_sql("I am sorry, I cannot answer that.")


def test_extract_rejects_multiple_statements():
    with pytest.raises(InvalidSQLResponseError, match="multiple"):
        extract_sql("SELECT 1; DELETE FROM students;")


def test_extract_rejects_write_statements():
    with pytest.raises(InvalidSQLResponseError):
        extract_sql("DELETE FROM students")


def test_insufficient_context_marker_detected():
    with pytest.raises(InvalidSQLResponseError, match="insufficient context: no schema"):
        extract_sql("INSUFFICIENT_CONTEXT: no schema provided")


def test_validate_single_select_rejects_update_ast():
    with pytest.raises(InvalidSQLResponseError, match="does not look like a SELECT"):
        validate_single_select("UPDATE students SET name = 'x'")


# ---------------------------------------------------------------------------
# 7. Grounding against Phase 2 metadata
# ---------------------------------------------------------------------------
def test_valid_join_sql_is_grounded():
    report = ground_sql(GOOD_JOIN_SQL, load_schema_metadata_snapshot())
    assert report.parsed is True
    assert report.grounded is True
    assert set(report.referenced_tables) == {"departments", "marks", "students"}


def test_hallucinated_table_detected():
    report = ground_sql("SELECT * FROM fake_table;", load_schema_metadata_snapshot())
    assert report.grounded is False
    assert any("unknown table: fake_table" in i for i in report.issues)


def test_hallucinated_column_detected():
    report = ground_sql(
        "SELECT gpa FROM students;", load_schema_metadata_snapshot()
    )
    assert report.grounded is False
    assert any("unknown column" in i and "gpa" in i for i in report.issues)


def test_cte_names_are_not_flagged_as_unknown_tables():
    sql = (
        "WITH dept_avg AS ("
        "SELECT s.department_id AS dept_id, AVG(m.marks) AS am "
        "FROM marks m JOIN students s ON m.student_id = s.student_id "
        "GROUP BY s.department_id) "
        "SELECT * FROM dept_avg WHERE am > 75;"
    )
    report = ground_sql(sql, load_schema_metadata_snapshot())
    assert report.grounded is True


def test_unparsable_sql_reported_not_repaired():
    report = ground_sql("SELECT FROM WHERE nonsense", load_schema_metadata_snapshot())
    assert report.parsed is False
    assert report.grounded is False
    assert any("does not parse" in i for i in report.issues)


# ---------------------------------------------------------------------------
# 8. Orchestration with faked LLM (everything else real)
# ---------------------------------------------------------------------------
def test_generate_happy_path_fields(service):
    result = service.generate("Which department has the highest average marks?")
    payload = result.model_dump()
    for key in ("question", "sql", "model", "grounded", "retrieved_documents",
                "retrieval_scores"):
        assert key in payload
    assert result.sql == GOOD_JOIN_SQL.rstrip(";")
    assert result.grounded is True
    assert result.retrieved_documents
    assert len(result.retrieved_documents) == len(result.retrieval_scores)


def test_generate_invalid_response_reported(fake_provider, rag_retriever):
    fake_provider.response = "I cannot help with that."
    result = TextToSQLService(provider=fake_provider, retriever=rag_retriever,
                              schema_metadata=load_schema_metadata_snapshot(),
                              ).generate("How many students are there?")
    assert result.sql is None
    assert result.error == "invalid_response"
    assert result.grounded is False


def test_generate_multiple_statements_reported(fake_provider, rag_retriever):
    fake_provider.response = "SELECT 1; SELECT 2;"
    result = TextToSQLService(provider=fake_provider, retriever=rag_retriever,
                              schema_metadata=load_schema_metadata_snapshot(),
                              ).generate("question?")
    assert result.error == "invalid_response"


def test_generate_hallucinated_sql_returned_ungrounded_not_repaired(
    fake_provider, rag_retriever
):
    fake_provider.response = "SELECT potion FROM wizards;"
    result = TextToSQLService(provider=fake_provider, retriever=rag_retriever,
                              schema_metadata=load_schema_metadata_snapshot(),
                              ).generate("magic data?")
    # Never silently repaired: the raw hallucinated SQL comes back flagged.
    assert result.sql == "SELECT potion FROM wizards"
    assert result.grounded is False
    assert result.error == "not_grounded"
    assert any("unknown table: wizards" in i for i in result.issues)


def test_generate_insufficient_context_response(fake_provider, rag_retriever):
    fake_provider.response = "INSUFFICIENT_CONTEXT: no attendance schema found"
    result = TextToSQLService(provider=fake_provider, retriever=rag_retriever,
                              schema_metadata=load_schema_metadata_snapshot(),
                              ).generate("attendance question?")
    assert result.sql is None
    assert result.error == "insufficient_context"


def test_generate_empty_question_raises(service):
    with pytest.raises(ValueError):
        service.generate("   ")


def test_llm_unavailable_propagates(rag_retriever):
    provider = FakeProvider(available=False)
    provider.generate = lambda *a, **k: (_ for _ in ()).throw(
        LLMUnavailableError("ollama down"))
    svc = TextToSQLService(provider=provider, retriever=rag_retriever,
                           schema_metadata=load_schema_metadata_snapshot())
    with pytest.raises(LLMUnavailableError):
        svc.generate("any question?")


# ---------------------------------------------------------------------------
# 9. API endpoint
# ---------------------------------------------------------------------------
@pytest.fixture()
def api_client(client: TestClient, service) -> TestClient:
    """TestClient with the generation dependency swapped to a fake-LLM service."""
    app.dependency_overrides[get_sql_generation_service] = lambda: service
    try:
        yield client
    finally:
        app.dependency_overrides.pop(get_sql_generation_service, None)


def test_api_generate_sql_success(api_client):
    response = api_client.post("/api/generate-sql",
                               json={"question": "Which department has the highest average marks?"})
    assert response.status_code == 200
    body = response.json()
    for key in ("question", "sql", "model", "grounded", "retrieved_documents"):
        assert key in body
    assert body["grounded"] is True
    assert body["sql"].upper().startswith("SELECT")


def test_api_generate_sql_ungrounded_passthrough(api_client, fake_provider):
    fake_provider.response = "SELECT * FROM nope_table"
    response = api_client.post("/api/generate-sql", json={"question": "weird stuff?"})
    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is False
    assert "nope_table" in body["sql"]  # not repaired


def test_api_generate_sql_llm_unavailable(client: TestClient, rag_retriever):
    provider = FakeProvider(available=False)
    def boom(*a, **k): raise LLMUnavailableError("down")
    provider.generate = boom
    broken = TextToSQLService(provider=provider, retriever=rag_retriever,
                              schema_metadata=load_schema_metadata_snapshot())
    app.dependency_overrides[get_sql_generation_service] = lambda: broken
    try:
        response = client.post("/api/generate-sql", json={"question": "anything at all"})
        assert response.status_code == 503
    finally:
        app.dependency_overrides.pop(get_sql_generation_service, None)


def test_api_generate_sql_validation_error(api_client):
    response = api_client.post("/api/generate-sql", json={"question": ""})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 10. Evaluation plumbing (metrics without live LLM)
# ---------------------------------------------------------------------------
def test_concept_coverage_scoring():
    sql = "SELECT d.department_name, AVG(m.marks) FROM departments d JOIN marks m ON 1=1 GROUP BY 1 ORDER BY 1 DESC LIMIT 1"
    assert concept_coverage(sql, ("AVG", "GROUP BY", "ORDER BY", "LIMIT")) == 1.0
    assert concept_coverage("SELECT 1", ("AVG", "HAVING")) == 0.0
    assert concept_coverage(None, ()) == 1.0


def test_relationship_coverage_scoring():
    from app.rag.sql_evaluation import ExpectedRelationship

    rels = (ExpectedRelationship("marks.student_id", "students.student_id"),)
    covered = relationship_coverage("SELECT ...", rels, grounded_tables=("marks", "students"))
    missing = relationship_coverage("SELECT ...", rels, grounded_tables=("marks",))
    assert covered == 1.0 and missing == 0.0


def test_evaluation_set_covers_required_topics():
    assert len(EVALUATION_QUESTIONS) >= 15
    text_blob = " ".join(q.question.lower() for q in EVALUATION_QUESTIONS)
    for topic in ("attendance", "course", "department", "top", "above their department average"):
        assert topic in text_blob
    concepts = {c for q in EVALUATION_QUESTIONS for c in q.concepts}
    assert {"JOIN", "GROUP BY", "HAVING", "LIMIT", "AVG", "COUNT"} <= concepts


# ---------------------------------------------------------------------------
# 11. Live Gemini test - SKIPPED unless LLM_PROVIDER=gemini + GEMINI_API_KEY
# ---------------------------------------------------------------------------
def _live_gemini_ready() -> str | None:
    """Return a skip reason, or None when live credentials are configured."""
    settings = get_settings()
    if settings.llm_provider != "gemini":
        return f"LLM_PROVIDER is {settings.llm_provider!r}, not 'gemini'"
    if not settings.gemini_api_key.get_secret_value().strip():
        return "GEMINI_API_KEY is not set"
    return None


_LIVE_SKIP_REASON = _live_gemini_ready()


@pytest.fixture(scope="module")
def live_service(rag_retriever) -> TextToSQLService:
    """Real retriever/assembler/grounding with the real Gemini API."""
    if _LIVE_SKIP_REASON is not None:
        pytest.skip(f"live Gemini test skipped: {_LIVE_SKIP_REASON}")
    settings = get_settings()
    provider = create_provider(settings.llm_provider, settings.active_llm_model)
    if not provider.is_available():
        pytest.skip(
            f"Gemini model {provider.model!r} is not reachable with the "
            "configured credentials"
        )
    return TextToSQLService(provider=provider, retriever=rag_retriever,
                            schema_metadata=load_schema_metadata_snapshot())


def test_live_gemini_pipeline_end_to_end(live_service):
    """Full RAG pipeline against the real Gemini API (generation only - the
    generated SQL is never executed)."""
    result = live_service.generate("Which department has the highest average marks?")
    # 1. RAG retrieval occurred.
    assert result.retrieved_documents, "no documents were retrieved"
    assert len(result.retrieved_documents) == len(result.retrieval_scores)
    # 2. Context reached Gemini and it produced SQL (parser step 4).
    if result.error == "insufficient_context":
        pytest.skip("model reported insufficient context")
    assert result.sql is not None
    parsed = validate_single_select(result.sql)
    assert parsed.ast is not None
    assert result.model.startswith("gemini:")
    # 3. Grounding check ran and its verdict is reported consistently.
    assert result.grounded == (not result.issues)
