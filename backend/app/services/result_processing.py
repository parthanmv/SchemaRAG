"""Phase 7: result processing - the final stage of the product pipeline.

Turns raw PostgreSQL rows into the presentation-ready payload consumed by
the API/React UI:

* JSON-safe coercion of driver types (Decimal, date/datetime, ...)
* deterministic column-kind inference (``number`` / ``boolean`` / ``text`` /
  ``null`` / ``unknown``) so clients can format values without guessing

Values are NEVER altered semantically: coercion is lossless for the JSON
payload, and kinds are annotations only. This stage runs after the Phase 5
security validator and row limiting; it has no SQL or database access.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

#: Column-kind labels exposed to clients.
KIND_NUMBER = "number"
KIND_BOOLEAN = "boolean"
KIND_TEXT = "text"
KIND_NULL = "null"
KIND_UNKNOWN = "unknown"

_JSON_SAFE = (bool, int, float, str)


def jsonable(value: Any) -> Any:
    """Convert DB driver values into JSON-safe primitives."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)):
        return value
    # Decimal / numeric
    if isinstance(value, Decimal):
        return float(value)
    # date / datetime / time
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def process_row(row: tuple | list) -> list[Any]:
    """Coerce one raw database row into a JSON-safe list of cells."""
    return [jsonable(v) for v in row]


def infer_column_kinds(
    columns: list[str], rows: list[list[Any]]
) -> list[str]:
    """Infer a display kind per column from already-coerced values.

    Rules (first match wins, evaluated over non-NULL cells):
    * no rows                       -> ``unknown``
    * every cell NULL               -> ``null``
    * all non-NULL cells are bools  -> ``boolean``  (checked before number:
      ``bool`` is an ``int`` subclass)
    * all non-NULL cells are ints/floats -> ``number``
    * anything else                 -> ``text``

    NULL cells never change a column's kind; they only matter when a column
    is entirely NULL.
    """
    if not columns:
        return []
    if not rows:
        return [KIND_UNKNOWN] * len(columns)

    kinds: list[str] = []
    for idx in range(len(columns)):
        non_null = [row[idx] for row in rows if row[idx] is not None]
        if not non_null:
            kinds.append(KIND_NULL)
        elif all(isinstance(v, bool) for v in non_null):
            kinds.append(KIND_BOOLEAN)
        elif all(
            isinstance(v, (int, float)) and not isinstance(v, bool)
            for v in non_null
        ):
            kinds.append(KIND_NUMBER)
        else:
            kinds.append(KIND_TEXT)
    return kinds
