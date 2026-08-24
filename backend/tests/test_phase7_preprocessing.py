"""Phase 7 tests: query preprocessing (pipeline stage 1).

The preprocessor must be deterministic, idempotent and purely textual:
normalise unicode punctuation / whitespace without ever changing the
meaning of the question.
"""

from __future__ import annotations

import pytest

from app.rag.preprocessing import (
    MAX_QUESTION_CHARS,
    ProcessedQuestion,
    QueryPreprocessor,
)


@pytest.fixture()
def preprocessor() -> QueryPreprocessor:
    return QueryPreprocessor()


# ---------------------------------------------------------------------------
# 1. Clean input passes through untouched
# ---------------------------------------------------------------------------
def test_clean_question_is_unchanged(preprocessor):
    question = "Which department has the highest average marks?"
    result = preprocessor.preprocess(question)
    assert isinstance(result, ProcessedQuestion)
    assert result.original == question
    assert result.processed == question
    assert result.transformations == []


def test_original_is_always_preserved(preprocessor):
    messy = "   Which\tdepartment\u2019s  students\u200b scored\u2013high? "
    result = preprocessor.preprocess(messy)
    assert result.original == messy
    assert result.processed != messy


def test_case_is_never_changed(preprocessor):
    assert preprocessor.preprocess("List ALL Departments").processed == (
        "List ALL Departments"
    )


# ---------------------------------------------------------------------------
# 2. Whitespace normalisation
# ---------------------------------------------------------------------------
def test_leading_and_trailing_whitespace_stripped(preprocessor):
    assert preprocessor.preprocess("  hello world  ").processed == "hello world"
    assert "strip" in preprocessor.preprocess("  hello  ").transformations


def test_internal_whitespace_collapsed(preprocessor):
    result = preprocessor.preprocess("Which   departments\n\thave  marks?")
    assert result.processed == "Which departments have marks?"
    assert "collapse_whitespace" in result.transformations


# ---------------------------------------------------------------------------
# 3. Unicode punctuation normalisation
# ---------------------------------------------------------------------------
def test_typographic_quotes_become_ascii(preprocessor):
    result = preprocessor.preprocess("Which \u201cCSE\u201d students passed?")
    assert '"CSE"' in result.processed

def test_curly_apostrophe_normalised(preprocessor):
    result = preprocessor.preprocess("department\u2019s average")
    assert "department's average" == result.processed


def test_dashes_and_special_spaces_normalised(preprocessor):
    result = preprocessor.preprocess(
        "high\u2013performing\u2014students\u00a0with\u200bzeros"
    )
    assert result.processed == "high-performing-students withzeros"


# ---------------------------------------------------------------------------
# 4. Truncation guard
# ---------------------------------------------------------------------------
def test_long_question_truncated_to_limit(preprocessor):
    long_question = "marks " * 300  # 1800 chars
    result = preprocessor.preprocess(long_question)
    assert len(result.processed) <= MAX_QUESTION_CHARS
    assert "truncate_to_limit" in result.transformations


# ---------------------------------------------------------------------------
# 5. Failure + idempotency contracts
# ---------------------------------------------------------------------------
def test_empty_question_raises(preprocessor):
    with pytest.raises(ValueError):
        preprocessor.preprocess("")


def test_whitespace_only_question_raises(preprocessor):
    with pytest.raises(ValueError):
        preprocessor.preprocess("   \t\n  ")


def test_preprocessing_is_idempotent(preprocessor):
    first = preprocessor.preprocess("  Which\u00a0 dept\u2019s  marks\u2013avg? ")
    second = preprocessor.preprocess(first.processed)
    assert second.processed == first.processed
    assert second.transformations == []
