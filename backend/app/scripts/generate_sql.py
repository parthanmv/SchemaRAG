"""Generate grounded PostgreSQL SELECT SQL from a natural-language question.

Run from the ``backend`` directory:

    python -m app.scripts.generate_sql "Which department has the highest average marks?"
    python -m app.scripts.generate_sql --eval          # run the evaluation set
    python -m app.scripts.generate_sql --top-k 12 "..."

Displays the question, retrieved documents, generated SQL, and grounding
status. SQL is NEVER executed.
"""

from __future__ import annotations

import argparse
import sys

from app.rag.llm.base import LLMError, LLMUnavailableError
from app.rag.sql_evaluation import EVALUATION_QUESTIONS, run_evaluation
from app.rag.text_to_sql import TextToSQLService, format_generated


def _build_service(top_k: int | None) -> TextToSQLService:
    if top_k is None:
        return TextToSQLService()
    return TextToSQLService(top_k=top_k)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate (never execute) grounded SELECT SQL from a question."
    )
    parser.add_argument("questions", nargs="*", help="Questions to convert.")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Override RAG_TOP_K for this run.")
    parser.add_argument("--eval", action="store_true",
                        help="Run the full Text-to-SQL evaluation set.")
    args = parser.parse_args(argv)

    try:
        service = _build_service(args.top_k)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not args.eval and not args.questions:
        args.questions = [EVALUATION_QUESTIONS[0].question]

    if args.eval:
        try:
            report = run_evaluation(service)
        except LLMUnavailableError as exc:
            print(f"LLM unavailable - evaluation skipped: {exc}", file=sys.stderr)
            return 3
        summary = report.summary()
        print("=== Text-to-SQL evaluation ===")
        print(f"Questions              : {summary['questions']:.0f}")
        print(f"Parse success          : {summary['parse_success']:.0%}")
        print(f"Table grounding        : {summary['table_grounding']:.0%}")
        print(f"Column grounding       : {summary['column_grounding']:.0%}")
        print(f"Concept coverage (avg) : {summary['concept_coverage']:.0%}")
        print(f"Relationship coverage  : {summary['relationship_coverage']:.0%}")
        print()
        for qr in report.results:
            status = (
                f"grounded={qr.grounded} tables={','.join(qr.referenced_tables)}"
                if qr.sql else f"no-sql ({qr.error})"
            )
            print(f"- {qr.question}\n    -> {status}")
        return 0

    exit_code = 0
    for question in args.questions:
        print("=" * 70)
        try:
            result = service.generate(question)
        except LLMUnavailableError as exc:
            print(f"Question: {question}\nLLM unavailable: {exc}", file=sys.stderr)
            exit_code = 3
            continue
        except LLMError as exc:
            print(f"Question: {question}\nLLM error: {exc}", file=sys.stderr)
            exit_code = 4
            continue
        print(format_generated(result))
        if not result.grounded or not result.sql:
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
