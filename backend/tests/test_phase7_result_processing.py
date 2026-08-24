"""Phase 7 tests: result processing (pipeline stage: PostgreSQL -> UI).

Covers JSON-safe coercion and column-kind inference, plus the
``sql_execution`` compatibility alias for the historical private helper.
"""

from __future__ import annotations

import datetime
from decimal import Decimal

from app.services.result_processing import (
    KIND_BOOLEAN,
    KIND_NULL,
    KIND_NUMBER,
    KIND_TEXT,
    KIND_UNKNOWN,
    infer_column_kinds,
    jsonable,
    process_row,
)
from app.services.sql_execution import _jsonable


# ---------------------------------------------------------------------------
# 1. jsonable coercion
# ---------------------------------------------------------------------------
def test_jsonable_passthrough_primitives():
    assert jsonable(None) is None
    assert jsonable(True) is True
    assert jsonable(7) == 7
    assert jsonable(2.5) == 2.5
    assert jsonable("cse") == "cse"


def test_jsonable_decimal_becomes_float():
    assert isinstance(jsonable(Decimal("71.99")), float)
    assert jsonable(Decimal("71.99")) == 71.99


def test_jsonable_temporal_types_isoformat():
    d = datetime.date(2025, 8, 24)
    dt = datetime.datetime(2025, 8, 24, 10, 30, 0)
    t = datetime.time(9, 15, 0)
    assert jsonable(d) == "2025-08-24"
    assert jsonable(dt) == "2025-08-24T10:30:00"
    assert jsonable(t) == "09:15:00"


def test_jsonable_unknown_object_falls_back_to_str():
    class Widget:
        def __str__(self):
            return "widget!"

    assert jsonable(Widget()) == "widget!"


def test_process_row_coerces_every_cell():
    row = [1, None, Decimal("3.5"), datetime.date(2025, 1, 1)]
    assert process_row(row) == [1, None, 3.5, "2025-01-01"]


def test_sql_execution_alias_matches_new_implementation():
    # The Phase 5 module keeps its historical private name; behaviour must be
    # identical to the shared result-processing implementation.
    assert _jsonable(Decimal("1.25")) == 1.25
    assert _jsonable(datetime.date(2024, 12, 31)) == "2024-12-31"
    assert _jsonable(None) is None


# ---------------------------------------------------------------------------
# 2. Column-kind inference
# ---------------------------------------------------------------------------
def test_empty_columns_give_no_kinds():
    assert infer_column_kinds([], []) == []


def test_no_rows_means_unknown():
    assert infer_column_kinds(["a", "b"], []) == [KIND_UNKNOWN, KIND_UNKNOWN]


def test_all_null_column():
    rows = [[None], [None], [None]]
    assert infer_column_kinds(["a"], rows) == [KIND_NULL]


def test_integer_and_float_columns_are_number():
    rows = [[1, 0.5], [2, 1.25], [3, 9]]
    kinds = infer_column_kinds(["ints", "floats"], rows)
    assert kinds == [KIND_NUMBER, KIND_NUMBER]


def test_boolean_checked_before_int_subclass_trap():
    # bool IS an int subclass - must still be reported as boolean.
    rows = [[True], [False], [None]]
    assert infer_column_kinds(["flags"], rows) == [KIND_BOOLEAN]


def test_text_column_and_mixed_column_are_text():
    rows = [["a", "x"], ["b", 5]]
    assert infer_column_kinds(["t", "mixed"], rows) == [KIND_TEXT, KIND_TEXT]


def test_null_cells_do_not_change_numeric_kind():
    rows = [[1], [None], [3]]
    assert infer_column_kinds(["n"], rows) == [KIND_NUMBER]


def test_realistic_result_shape():
    columns = ["department_name", "student_count", "avg_marks", "is_active"]
    rows = [
        ["Computer Science", 120, Decimal("71.99"), True],
        ["Electronics", 98, Decimal("66.40"), False],
        [None, 75, Decimal("70.00"), None],
    ]
    kinds = infer_column_kinds(
        columns, [process_row(r) for r in rows]
    )
    assert kinds == [KIND_TEXT, KIND_NUMBER, KIND_NUMBER, KIND_BOOLEAN]
