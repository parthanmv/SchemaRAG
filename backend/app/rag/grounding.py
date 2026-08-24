"""Grounding check: verify generated SQL against real PostgreSQL metadata.

Every table and column reference in the generated statement is resolved
against the Phase 2 :class:`SchemaMetadata` snapshot. Unknown objects are
reported; SQL is never repaired or rewritten. This is a grounding report,
not the full Phase 5 security validator.
"""

from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import expressions as exp
from sqlglot.errors import ParseError

from app.rag.models import SchemaMetadata


@dataclass(frozen=True)
class GroundingReport:
    """Outcome of checking one candidate statement."""

    parsed: bool
    grounded: bool
    referenced_tables: tuple[str, ...]
    issues: tuple[str, ...]


def _collect_cte_names(ast) -> set[str]:  # noqa: ANN001 - sqlglot Expression
    names: set[str] = set()
    for with_node in ast.find_all(exp.With):
        for cte in with_node.find_all(exp.CTE):
            alias = cte.alias_or_name
            if alias:
                names.add(alias.lower())
    return names


def ground_sql(sql: str, metadata: SchemaMetadata) -> GroundingReport:
    """Check all table/column references in *sql* against *metadata*.

    Rules:
    * CTE aliases count as valid "tables" (they exist within the query).
    * A column is accepted if it exists on any base table referenced by the
      query or matches a SELECT output alias. This union approach keeps the
      check strict about hallucinated names while avoiding false positives
      from incomplete scope binding (full validation is Phase 5's job).
    """
    try:
        ast = sqlglot.parse_one(sql, dialect="postgres")
    except ParseError as exc:
        return GroundingReport(
            parsed=False, grounded=False, referenced_tables=(),
            issues=(f"SQL does not parse: {exc}",),
        )
    if ast is None:
        return GroundingReport(parsed=False, grounded=False, referenced_tables=(),
                               issues=("empty expression",))

    known_tables = {t.name.lower(): t for t in metadata.tables}
    cte_names = _collect_cte_names(ast)

    referenced_tables: list[str] = []
    unknown_tables: list[str] = []
    base_column_pool: set[str] = set()

    for table_node in ast.find_all(exp.Table):
        name = table_node.name
        if not name:
            continue
        lowered = name.lower()
        if lowered in cte_names:
            continue  # internal query-scope relation, not a database object
        if lowered in known_tables:
            if lowered not in referenced_tables:
                referenced_tables.append(lowered)
                base_column_pool |= {
                    c.name.lower() for c in known_tables[lowered].columns
                }
        else:
            if lowered not in unknown_tables:
                unknown_tables.append(lowered)
                # Still allow columns to be checked leniently? No: unknown
                # tables contribute nothing to the column pool.

    select_aliases: set[str] = set()
    for alias_node in ast.find_all(exp.Alias):
        alias_name = alias_node.alias_or_name
        if alias_name:
            select_aliases.add(alias_name.lower().strip('"'))

    unknown_columns: list[str] = []
    for column_node in ast.find_all(exp.Column):
        col_name = column_node.name
        if not col_name or isinstance(column_node.parent, exp.Star):
            continue
        lowered = col_name.lower().strip('"')
        if lowered not in base_column_pool and lowered not in select_aliases:
            display = f"{column_node.table}.{col_name}" if column_node.table else col_name
            if display.lower() not in unknown_columns:
                unknown_columns.append(display.lower())

    issues: list[str] = [f"unknown table: {t}" for t in unknown_tables]
    issues.extend(f"unknown column: {c}" for c in unknown_columns)

    return GroundingReport(
        parsed=True,
        grounded=not issues,
        referenced_tables=tuple(sorted(set(referenced_tables))),
        issues=tuple(issues),
    )
