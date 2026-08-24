"""Manual retrieval demo / evaluation runner.

Run from the ``backend`` directory:

    python -m app.scripts.test_retrieval                 # run the evaluation set
    python -m app.scripts.test_retrieval --top-k 3 "question one" "question two"
    python -m app.scripts.test_retrieval --eval          # evaluation set with metrics

Prints per-question ranked results (document_id, score, type, tables) without
dumping full document contents.
"""

from __future__ import annotations

import argparse
import sys

from app.rag.evaluation import (
    EVALUATION_SET,
    evaluate_retrieval,
    format_report,
)
from app.rag.retriever import KnowledgeRetriever


def print_results(question: str, results, top_k: int) -> None:  # noqa: ANN001
    print(f"\nQuestion:\n{question}\n")
    if not results:
        print("No results (empty query or invalid top_k).")
        return
    print(f"Top results (top_k={min(top_k, len(results))}):")
    for rank, result in enumerate(results[:top_k], start=1):
        tables = ", ".join(result.tables) if result.tables else "-"
        print(f"{rank}. {result.document_id}")
        print(f"   score        : {result.score:.4f}")
        print(f"   document_type: {result.document_type}")
        print(f"   tables       : {tables}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Try semantic retrieval over the SchemaRAG knowledge base."
    )
    parser.add_argument(
        "questions", nargs="*", help="Questions to retrieve for (default: demo set)."
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--eval", action="store_true",
        help="Run the full retrieval evaluation (Recall@1/3/5 + table-level).",
    )
    parser.add_argument(
        "--type", dest="document_type", default=None,
        choices=["schema", "relationship", "constraint", "business_rule", "query_example"],
        help="Optional metadata filter.",
    )
    args = parser.parse_args(argv)

    retriever = KnowledgeRetriever()

    if args.eval:
        report = evaluate_retrieval(retriever, ks=(1, 3, 5))
        for case in EVALUATION_SET:
            results = retriever.retrieve(case.question, top_k=args.top_k,
                                         document_type=args.document_type)
            print_results(case.question, results, args.top_k)
        print("\n=== Evaluation ===")
        print(format_report(report))
        return 0

    questions = args.questions or [case.question for case in EVALUATION_SET]
    for question in questions:
        results = retriever.retrieve(question, top_k=args.top_k,
                                     document_type=args.document_type)
        print_results(question, results, args.top_k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
