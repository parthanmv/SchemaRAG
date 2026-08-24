"""Text-to-SQL evaluation set + metrics (no SQL is ever executed).

For each question we define the expected tables, SQL concepts and join
relationships, run the real pipeline (retrieval -> context -> LLM ->
extraction -> grounding), and measure:

* parse success      - extracted SQL parses with sqlglot (postgres dialect)
* table grounding    - expected tables appear among grounded tables AND
                       grounding found no unknown tables
* column grounding   - grounding found no unknown columns
* concept coverage   - share of expected SQL concepts present in the SQL text

The runner requires a live LLM; callers skip it when none is available.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ExpectedRelationship:
    """A join path the generated SQL is expected to rely on."""

    source: str  # e.g. "students.department_id"
    target: str  # e.g. "departments.department_id"


@dataclass(frozen=True)
class EvaluationQuestion:
    """One evaluation question with its expectations."""

    question: str
    tables: tuple[str, ...]
    concepts: tuple[str, ...]
    relationships: tuple[ExpectedRelationship, ...] = ()


def _rel(source: str, target: str) -> ExpectedRelationship:
    return ExpectedRelationship(source=source, target=target)


EVALUATION_QUESTIONS: tuple[EvaluationQuestion, ...] = (
    EvaluationQuestion(
        question="Which department has the highest average marks?",
        tables=("departments", "students", "marks"),
        concepts=("AVG", "GROUP BY", "ORDER BY", "LIMIT"),
        relationships=(
            _rel("marks.student_id", "students.student_id"),
            _rel("students.department_id", "departments.department_id"),
        ),
    ),
    EvaluationQuestion(
        question="List the names of all students in the Computer Science and Engineering department.",
        tables=("students", "departments"),
        concepts=("JOIN",),
        relationships=(
            _rel("students.department_id", "departments.department_id"),
        ),
    ),
    EvaluationQuestion(
        question="How many students are enrolled in each course?",
        tables=("courses", "enrollments"),
        concepts=("JOIN", "COUNT", "GROUP BY"),
        relationships=(
            _rel("enrollments.course_id", "courses.course_id"),
        ),
    ),
    EvaluationQuestion(
        question="What is the average mark for each exam type?",
        tables=("marks",),
        concepts=("AVG", "GROUP BY"),
        relationships=(),
    ),
    EvaluationQuestion(
        question="Show the top 5 students by average marks.",
        tables=("students", "marks"),
        concepts=("JOIN", "AVG", "ORDER BY", "LIMIT"),
        relationships=(
            _rel("marks.student_id", "students.student_id"),
        ),
    ),
    EvaluationQuestion(
        question="Which courses have more than 100 enrollments?",
        tables=("courses", "enrollments"),
        concepts=("JOIN", "COUNT", "GROUP BY", "HAVING"),
        relationships=(
            _rel("enrollments.course_id", "courses.course_id"),
        ),
    ),
    EvaluationQuestion(
        question="Which departments have an average mark above 75?",
        tables=("departments", "students", "marks"),
        concepts=("JOIN", "AVG", "GROUP BY", "HAVING"),
        relationships=(
            _rel("marks.student_id", "students.student_id"),
            _rel("students.department_id", "departments.department_id"),
        ),
    ),
    EvaluationQuestion(
        question="Which students scored above their department average?",
        tables=("departments", "students", "marks"),
        concepts=("JOIN", "AVG", "subquery"),
        relationships=(
            _rel("marks.student_id", "students.student_id"),
            _rel("students.department_id", "departments.department_id"),
        ),
    ),
    EvaluationQuestion(
        question="Find students who have never been enrolled in any course.",
        tables=("enrollments", "students"),
        concepts=("NOT EXISTS", "subquery"),
        relationships=(
            _rel("enrollments.student_id", "students.student_id"),
        ),
    ),
    EvaluationQuestion(
        question="What is the attendance percentage of each student for each course?",
        tables=("attendance", "students", "courses"),
        concepts=("JOIN",),
        relationships=(
            _rel("attendance.student_id", "students.student_id"),
            _rel("attendance.course_id", "courses.course_id"),
        ),
    ),
    EvaluationQuestion(
        question="How many classes were held in total for the Database Management Systems course?",
        tables=("attendance", "courses"),
        concepts=("JOIN", "SUM", "WHERE"),
        relationships=(
            _rel("attendance.course_id", "courses.course_id"),
        ),
    ),
    EvaluationQuestion(
        question="List all courses offered by the Mathematics department.",
        tables=("courses", "departments"),
        concepts=("JOIN", "WHERE"),
        relationships=(
            _rel("courses.department_id", "departments.department_id"),
        ),
    ),
    EvaluationQuestion(
        question="Show the count of students per department.",
        tables=("students", "departments"),
        concepts=("JOIN", "COUNT", "GROUP BY"),
        relationships=(
            _rel("students.department_id", "departments.department_id"),
        ),
    ),
    EvaluationQuestion(
        question="Which student has the lowest attendance percentage?",
        tables=("attendance", "students"),
        concepts=("JOIN", "ORDER BY", "LIMIT"),
        relationships=(
            _rel("attendance.student_id", "students.student_id"),
        ),
    ),
    EvaluationQuestion(
        question="What is the average mark of students admitted in 2025?",
        tables=("students", "marks"),
        concepts=("JOIN", "AVG", "WHERE"),
        relationships=(
            _rel("marks.student_id", "students.student_id"),
        ),
    ),
    EvaluationQuestion(
        question="For each department, how many courses does it offer?",
        tables=("departments", "courses"),
        concepts=("JOIN", "COUNT", "GROUP BY"),
        relationships=(
            _rel("courses.department_id", "departments.department_id"),
        ),
    ),
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
_CONCEPT_PATTERNS = {
    # concept -> regex over the normalised SQL text
    "AVG": r"\bAVG\s*\(",
    "COUNT": r"\bCOUNT\s*\(",
    "SUM": r"\bSUM\s*\(",
    "MIN": r"\bMIN\s*\(",
    "MAX": r"\bMAX\s*\(",
    "JOIN": r"\bJOIN\b",
    "WHERE": r"\bWHERE\b",
    "GROUP BY": r"\bGROUP\s+BY\b",
    "HAVING": r"\bHAVING\b",
    "ORDER BY": r"\bORDER\s+BY\b",
    "LIMIT": r"\bLIMIT\b",
    "DISTINCT": r"\bDISTINCT\b",
    "subquery": r"\(\s*SELECT\b",
    "NOT EXISTS": r"\bNOT\s+EXISTS\b",
}


def concept_coverage(sql: str | None, concepts: tuple[str, ...]) -> float:
    """Fraction of *concepts* detectable in the SQL text (1.0 when none)."""
    if not concepts:
        return 1.0
    if not sql:
        return 0.0
    normalized = " ".join(sql.upper().split())
    hits = sum(
        1
        for concept in concepts
        if re.search(_CONCEPT_PATTERNS.get(concept, re.escape(concept.upper())),
                     normalized)
    )
    return hits / len(concepts)


def relationship_coverage(
    sql: str | None,
    relationships: tuple[ExpectedRelationship, ...],
    grounded_tables: tuple[str, ...],
) -> float:
    """Share of expected relationships whose endpoint pairs co-occur.

    A relationship counts as covered when both of its tables are among the
    grounded referenced tables (the SQL joins them somewhere) - a robust,
    string-independent proxy that still catches missing join paths.
    """
    if not relationships:
        return 1.0
    if not sql:
        return 0.0
    table_set = set(grounded_tables) | set(re.findall(r"\bFROM\s+([a-z_]+)", sql.upper().lower()))
    hits = sum(
        1
        for rel in relationships
        if rel.source.split(".")[0] in table_set
        and rel.target.split(".")[0] in table_set
    )
    return hits / len(relationships)


@dataclass
class QuestionResult:
    """Per-question outcome of one evaluation run."""

    question: str
    sql: str | None
    model: str
    parse_success: bool
    grounded: bool
    referenced_tables: tuple[str, ...]
    table_grounding_ok: bool
    column_grounding_ok: bool
    concept_score: float
    relationship_score: float
    error: str | None = None


@dataclass
class EvaluationReport:
    """Aggregate Text-to-SQL quality metrics."""

    results: list[QuestionResult]
    total_questions: int

    @property
    def parse_success_rate(self) -> float:
        return sum(r.parse_success for r in self.results) / self.total_questions

    @property
    def table_grounding_rate(self) -> float:
        return sum(r.table_grounding_ok for r in self.results) / self.total_questions

    @property
    def column_grounding_rate(self) -> float:
        return sum(r.column_grounding_ok for r in self.results) / self.total_questions

    @property
    def mean_concept_coverage(self) -> float:
        return sum(r.concept_score for r in self.results) / self.total_questions

    @property
    def mean_relationship_coverage(self) -> float:
        return sum(r.relationship_score for r in self.results) / self.total_questions

    def summary(self) -> dict[str, float]:
        return {
            "questions": float(self.total_questions),
            "parse_success": self.parse_success_rate,
            "table_grounding": self.table_grounding_rate,
            "column_grounding": self.column_grounding_rate,
            "concept_coverage": self.mean_concept_coverage,
            "relationship_coverage": self.mean_relationship_coverage,
        }


def evaluate_question(service, item: EvaluationQuestion) -> QuestionResult:  # noqa: ANN001
    """Run one question through the live service and score the outcome."""
    result = service.generate(item.question)

    from app.rag.grounding import ground_sql

    report = ground_sql(result.sql, service.schema_metadata) if result.sql else None
    parse_success = result.sql is not None and report is not None and report.parsed

    referenced = report.referenced_tables if report else ()
    unknown_tables = any(i.startswith("unknown table:") for i in (report.issues if report else ()))
    unknown_columns = any(i.startswith("unknown column:") for i in (result.issues))

    table_ok = (
        bool(result.sql)
        and set(item.tables) <= set(referenced)
        and not unknown_tables
    )
    column_ok = bool(result.sql) and not unknown_columns

    return QuestionResult(
        question=item.question,
        sql=result.sql,
        model=result.model,
        parse_success=parse_success,
        grounded=result.grounded,
        referenced_tables=tuple(sorted(set(item.tables) & set(referenced))),
        table_grounding_ok=bool(table_ok),
        column_grounding_ok=bool(column_ok),
        concept_score=concept_coverage(result.sql, item.concepts),
        relationship_score=relationship_coverage(
            result.sql, item.relationships, referenced
        ),
        error=result.error,
    )


def run_evaluation(service) -> EvaluationReport:  # noqa: ANN001 - TextToSQLService
    """Evaluate every configured question against the live pipeline."""
    results = [evaluate_question(service, q) for q in EVALUATION_QUESTIONS]
    return EvaluationReport(results=results, total_questions=len(results))
