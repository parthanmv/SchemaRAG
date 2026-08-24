"""Lightweight retrieval evaluation.

Defines the curated evaluation set and computes:

* document-level Recall@K  – a question "hits" at K when any top-K document
  matches one of its expected ``document_id``s **or** covers all of its
  expected tables (relevant information found, even if the exact id differs)
* table-level Recall@K     – the union of tables across the top-K documents
  covers all expected tables (and, when specified, at least one retrieved
  document has the expected type)

The evaluation is intentionally simple and deterministic; it exists to give
quick feedback on retrieval quality, not to be a benchmark suite.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.rag.retriever import KnowledgeRetriever, RetrievalResult


@dataclass(frozen=True)
class EvaluationCase:
    """One test question with its relevance expectations."""

    question: str
    expected_document_ids: tuple[str, ...] = ()
    expected_tables: frozenset[str] = field(default_factory=frozenset)
    # Optional extra requirement for the table-level metric.
    expected_document_type: str | None = None


EVALUATION_SET: tuple[EvaluationCase, ...] = (
    EvaluationCase(
        question="Which department has the highest average marks?",
        expected_tables=frozenset({"departments", "students", "marks"}),
    ),
    EvaluationCase(
        question="Which students have attendance below 75%?",
        expected_tables=frozenset({"students", "attendance"}),
    ),
    EvaluationCase(
        question="Show the top 5 students by marks.",
        expected_tables=frozenset({"students", "marks"}),
    ),
    EvaluationCase(
        question="Which departments have an average mark above 75?",
        expected_tables=frozenset({"departments", "students", "marks"}),
    ),
    EvaluationCase(
        question="Which students scored above their department average?",
        expected_tables=frozenset({"departments", "students", "marks"}),
    ),
    EvaluationCase(
        question="Which courses belong to each department?",
        expected_tables=frozenset({"courses", "departments"}),
    ),
    EvaluationCase(
        question="What is the relationship between students and departments?",
        expected_tables=frozenset({"students", "departments"}),
        expected_document_type="relationship",
    ),
    EvaluationCase(
        question="What are the allowed mark values?",
        expected_tables=frozenset({"marks"}),
        expected_document_type="constraint",
    ),
    EvaluationCase(
        question="What does attendance below 75% mean?",
        expected_tables=frozenset({"attendance"}),
        expected_document_type="business_rule",
    ),
    EvaluationCase(
        question="Show student email information.",
        expected_tables=frozenset({"students"}),
        expected_document_type="schema",
        expected_document_ids=("schema_students",),
    ),
)


@dataclass
class CaseEvaluation:
    """Per-question outcome for one K value."""

    question: str
    k: int
    retrieved_ids: tuple[str, ...]
    document_hit: bool
    table_hit: bool


@dataclass
class EvaluationReport:
    """Aggregate metrics over the whole evaluation set."""

    total_questions: int
    cases: dict[int, list[CaseEvaluation]]

    def hit_rate(self, k: int, *, table_level: bool) -> float:
        rows = self.cases.get(k, [])
        if not rows:
            return 0.0
        hits = sum(1 for c in rows if (c.table_hit if table_level else c.document_hit))
        return hits / len(rows)

    def summary(self) -> dict[str, float]:
        out: dict[str, float] = {"questions": float(self.total_questions)}
        for k in sorted(self.cases):
            out[f"recall@{k}"] = self.hit_rate(k, table_level=False)
            out[f"table_recall@{k}"] = self.hit_rate(k, table_level=True)
        return out


def _document_hit(result: RetrievalResult, case: EvaluationCase) -> bool:
    ids_ok = not case.expected_document_ids or result.document_id in case.expected_document_ids
    tables_ok = not case.expected_tables or set(case.expected_tables) <= set(result.tables)
    return ids_ok or tables_ok


def _table_hit(results: list[RetrievalResult], case: EvaluationCase) -> bool:
    covered: set[str] = set()
    for result in results:
        covered |= set(result.tables)
    tables_ok = set(case.expected_tables) <= covered
    if case.expected_document_type is None:
        return tables_ok
    type_found = any(r.document_type == case.expected_document_type for r in results)
    return tables_ok and type_found


def evaluate_retrieval(
    retriever: KnowledgeRetriever,
    cases: tuple[EvaluationCase, ...] | list[EvaluationCase] = EVALUATION_SET,
    ks: tuple[int, ...] = (1, 3, 5),
) -> EvaluationReport:
    """Run every case through the retriever once and score it at each K."""
    per_k: dict[int, list[CaseEvaluation]] = {k: [] for k in ks}
    max_k = max(ks)

    for case in cases:
        results = retriever.retrieve(case.question, top_k=max_k)
        for k in ks:
            top = results[:k]
            per_k[k].append(
                CaseEvaluation(
                    question=case.question,
                    k=k,
                    retrieved_ids=tuple(r.document_id for r in top),
                    document_hit=any(_document_hit(r, case) for r in top),
                    table_hit=_table_hit(top, case),
                )
            )

    return EvaluationReport(total_questions=len(cases), cases=per_k)


def format_report(report: EvaluationReport) -> str:
    """Human-readable multi-line summary of an :class:`EvaluationReport`."""
    lines = [f"Questions evaluated: {report.total_questions}"]
    for k in sorted(report.cases):
        lines.append(
            f"Recall@{k}: {report.hit_rate(k, table_level=False):.0%}   "
            f"(table-level: {report.hit_rate(k, table_level=True):.0%})"
        )
    return "\n".join(lines)
