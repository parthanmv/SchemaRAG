"""Curated natural-language query examples for future Text-to-SQL retrieval.

These are knowledge documents only: they pair a natural-language question
with the tables and SQL concepts needed to answer it. No SQL is generated or
executed in this phase.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class QueryExample(BaseModel):
    """One curated question with its relevant tables and SQL concepts."""

    model_config = ConfigDict(frozen=True)

    example_id: str
    question: str
    relevant_tables: tuple[str, ...]
    concepts: tuple[str, ...]


QUERY_EXAMPLES: tuple[QueryExample, ...] = (
    QueryExample(
        example_id="dept_highest_avg_marks",
        question="Which department has the highest average marks?",
        relevant_tables=("departments", "students", "marks"),
        concepts=("JOIN", "GROUP BY", "AVG", "ORDER BY", "LIMIT"),
    ),
    QueryExample(
        example_id="students_low_attendance",
        question="Which students have attendance below 75%?",
        relevant_tables=("students", "attendance"),
        concepts=("JOIN", "filtering"),
    ),
    QueryExample(
        example_id="top_students_by_marks",
        question="Show the top 5 students by marks.",
        relevant_tables=("students", "marks"),
        concepts=("JOIN", "ORDER BY", "LIMIT"),
    ),
    QueryExample(
        example_id="depts_avg_above_75",
        question="Which departments have an average mark above 75?",
        relevant_tables=("departments", "students", "marks"),
        concepts=("JOIN", "GROUP BY", "AVG", "HAVING"),
    ),
    QueryExample(
        example_id="above_department_average",
        question="Which students scored above their department average?",
        relevant_tables=("departments", "students", "marks"),
        concepts=("JOIN", "aggregation", "subquery/correlated logic"),
    ),
)
