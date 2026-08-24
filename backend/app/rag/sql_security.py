"""Phase 5 AST-based SQL security validation (read-only policy).

The validator inspects the sqlglot parse tree of a candidate statement and
enforces a strict allowlist policy:

* exactly one statement
* statement root must be ``SELECT``, ``WITH ... SELECT`` or a set operation
  over selects (``UNION``/``EXCEPT``/``INTERSECT``)
* every node in the tree must belong to an allowlist of read-only expression
  and clause types - anything unknown is rejected (fail closed)
* no comments anywhere in the parsed statement (they can hide content from
  reviewers and are never needed in generated SQL)
* no system/catalog access: ``pg_catalog``, ``information_schema``,
  ``pg_toast``/``pg_temp`` namespaces, and unqualified ``pg_*`` tables
* no dangerous functions (``pg_sleep``, server file access, large objects,
  ``dblink``, advisory locks, sequence mutation, ...) and no schema-qualified
  function calls at all

This module deliberately does NOT check table/column existence against the
database schema; that is grounding's job (:mod:`app.rag.grounding`). The two
checks are composed by the execution service. Nothing here repairs or rewrites
SQL: the output is a verdict plus an optional normalized rendering.

Schema-qualified function calls are rejected wholesale (fail closed): the
college-domain queries never need them, and qualification is the primary way
to reach privileged routines such as ``pg_catalog.pg_read_file``.
"""

from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import expressions as exp
from sqlglot.errors import ParseError

#: Node types that make up legitimate read-only queries. Anything whose type
#: does not ``isinstance``-match this tuple is rejected. ``Func`` covers all
#: scalar/aggregate/window function nodes (function *names* are screened
#: separately); ``With``/``CTE`` cover CTEs; set operations recurse into their
#: SELECT operands which are validated by the same walk.
_ALLOWED_NODE_TYPES: tuple[type, ...] = (
    # statements / structure
    exp.Select,
    exp.With,
    exp.CTE,
    exp.Union,
    exp.Except,
    exp.Intersect,
    # clauses
    exp.From,
    exp.Join,
    exp.Where,
    exp.Group,
    exp.Having,
    exp.Order,
    exp.Ordered,
    exp.Limit,
    exp.Offset,
    exp.Distinct,
    # relations / references
    exp.Table,
    exp.TableAlias,
    exp.Alias,
    exp.Column,
    exp.Identifier,
    exp.Star,
    exp.Subquery,
    # plain value holders (EXTRACT field names, IN-lists, ...)
    exp.Var,
    exp.Tuple,
    # expressions / predicates
    exp.Paren,
    exp.Literal,
    exp.Boolean,
    exp.Null,
    exp.Cast,
    exp.DataType,
    exp.Case,
    exp.If,
    exp.When,
    exp.And,
    exp.Or,
    exp.Not,
    exp.Is,
    exp.In,
    exp.Between,
    exp.Like,
    exp.ILike,
    exp.EQ,
    exp.NEQ,
    exp.GT,
    exp.GTE,
    exp.LT,
    exp.LTE,
    # arithmetic / string / bitwise operators
    exp.Add,
    exp.Sub,
    exp.Mul,
    exp.Div,
    exp.Mod,
    exp.Pow,
    exp.Neg,
    exp.BitwiseAnd,
    exp.BitwiseOr,
    exp.BitwiseXor,
    exp.DPipe,
    # functions (name screening happens separately)
    exp.Func,
    exp.Window,
    exp.WindowSpec,
    exp.WithinGroup,
)

#: PostgreSQL namespaces that must never be queried.
_SYSTEM_SCHEMAS = frozenset({"pg_catalog", "information_schema", "pg_toast"})

#: Functions with side effects, server access, or blocking behaviour.
_DANGEROUS_FUNCTIONS = frozenset(
    {
        # sleep / backend manipulation
        "pg_sleep",
        "pg_sleep_for",
        "pg_sleep_until",
        "pg_terminate_backend",
        "pg_cancel_backend",
        "pg_reload_conf",
        "pg_rotate_logfile",
        # server filesystem access
        "pg_read_file",
        "pg_read_binary_file",
        "pg_ls_dir",
        "pg_stat_file",
        # large objects
        "lo_import",
        "lo_import_file",
        "lo_export",
        "lo_get",
        "lo_put",
        "lo_unlink",
        # remote execution / arbitrary query helpers
        "dblink",
        "dblink_connect",
        "dblink_connect_u",
        "dblink_exec",
        "dblink_send_query",
        "query_to_xml",
        "cursor_to_xml",
        "table_to_xml",
        # sequence mutation (side effects)
        "nextval",
        "setval",
        # session/config mutation
        "set_config",
        "set_role",
        "reset_role",
        # advisory locks (can block other sessions indefinitely)
        "pg_advisory_lock",
        "pg_advisory_xact_lock",
        "pg_advisory_shared_lock",
        "pg_advisory_unlock",
        "pg_advisory_unlock_all",
        # backup / WAL control
        "pg_backup_start",
        "pg_backup_stop",
        "pg_switch_wal",
        "pg_create_physical_replication_slot",
        "pg_create_logical_replication_slot",
        "pg_drop_replication_slot",
    }
)


@dataclass(frozen=True)
class SecurityReport:
    """Outcome of validating one candidate read-only statement."""

    allowed: bool
    issues: tuple[str, ...]
    normalized_sql: str | None


class SQLSecurityValidator:
    """Fail-closed AST validator enforcing the Phase 5 read-only policy."""

    def validate(self, sql: str) -> SecurityReport:
        """Validate one statement string; never raises for bad input."""
        if not sql or not sql.strip():
            return SecurityReport(False, ("empty SQL",), None)

        try:
            statements = sqlglot.parse(sql.strip(), dialect="postgres")
        except ParseError as exc:
            return SecurityReport(
                False, (f"SQL does not parse: {_short(str(exc))}",), None
            )
        except Exception as exc:  # fail closed on any unexpected parser error
            return SecurityReport(
                False,
                (f"SQL could not be validated ({type(exc).__name__})",),
                None,
            )

        statements = [s for s in statements if s is not None]
        if len(statements) != 1:
            return SecurityReport(
                False,
                (f"exactly one statement required (found {len(statements)})",),
                None,
            )

        ast = statements[0]
        issues: list[str] = []

        root = ast
        if isinstance(root, exp.With):
            root = root.this  # CTE wrapper: policy applies to what follows
        if not isinstance(root, (exp.Select, exp.Union, exp.Except, exp.Intersect)):
            issues.append(
                f"only SELECT/WITH read queries are allowed "
                f"(got {type(root).__name__})"
            )
            return SecurityReport(False, tuple(issues), None)

        # Comments can hide text from reviewers; the AST drops them, so any
        # surviving comment means the source contained one -> reject.
        if _has_comments(ast):
            issues.append("comments are not allowed in generated SQL")

        # Structural allowlist walk (fail closed on unknown node types).
        unknown_types = _walk_unknown_nodes(ast)
        for name in unknown_types[:5]:
            issues.append(f"disallowed construct: {name}")

        # System/catalog access.
        for issue in _system_table_issues(ast):
            issues.append(issue)

        # Dangerous functions & qualified function calls.
        for issue in _function_issues(ast):
            issues.append(issue)

        return SecurityReport(
            allowed=not issues,
            issues=tuple(issues),
            normalized_sql=ast.sql(dialect="postgres") if not issues else None,
        )


def validate_sql_security(sql: str) -> SecurityReport:
    """Convenience wrapper around a shared validator instance."""
    return SQLSecurityValidator().validate(sql)


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------
def _short(message: str, limit: int = 160) -> str:
    message = " ".join(message.split())
    return message if len(message) <= limit else message[: limit - 3] + "..."


def _has_comments(ast) -> bool:  # noqa: ANN001 - sqlglot Expression
    return any(getattr(node, "comments", None) for node in ast.walk())


def _walk_unknown_nodes(ast) -> list[str]:  # noqa: ANN001
    """Return disallowed/unknown node type names found in the tree."""
    unknown: list[str] = []
    seen: set[str] = set()
    for node in ast.walk():
        if isinstance(node, _ALLOWED_NODE_TYPES):
            continue
        name = type(node).__name__
        if name not in seen:
            seen.add(name)
            unknown.append(name)
    return unknown


def _table_parts(table_node: exp.Table) -> tuple[str, str, str]:
    """Return (catalog, db, name) of a table reference, lowercased."""
    catalog = table_node.catalog or ""
    db = table_node.db or ""
    name = table_node.name or ""
    return catalog.lower(), db.lower(), name.lower()


def _system_table_issues(ast) -> list[str]:  # noqa: ANN001
    issues: list[str] = []
    for node in ast.find_all(exp.Table):
        catalog, db, name = _table_parts(node)
        qualified = {part for part in (catalog, db) if part}
        if qualified & _SYSTEM_SCHEMAS:
            issues.append(f"system/catalog access is not allowed: {node.sql()}")
        elif name.startswith("pg_"):
            issues.append(f"access to pg_* relations is not allowed: {name}")
        elif catalog:
            # Three-part references (catalog.db.table) reach outside the
            # project database entirely.
            issues.append(f"cross-database access is not allowed: {node.sql()}")
        elif db and db != "public":
            # Only the public schema of THIS database is approved; anything
            # else (other schemas) is rejected even if grounding would also
            # flag it - defence in depth.
            issues.append(f"access to non-public schema is not allowed: {node.sql()}")
        elif db == "public":
            # Explicitly schema-qualified project tables stay allowed.
            continue
    return issues


def _function_name(func_node: exp.Func) -> str | None:
    """Best-effort extraction of a bare, lowercased function name."""
    if isinstance(func_node, exp.Anonymous):
        name = func_node.name
        return name.lower().strip('"') if name else None
    return None


def _function_issues(ast) -> list[str]:  # noqa: ANN001
    issues: list[str] = []
    for func in ast.find_all(exp.Func):
        if isinstance(func, (exp.Window, exp.WindowSpec, exp.WithinGroup)):
            continue  # structural wrappers around real function nodes
        # Schema-qualified calls (schema.func(...)) parse to Anonymous over a
        # Dot chain; they are rejected outright (fail closed).
        inner = func.this
        if isinstance(inner, exp.Dot):
            issues.append("schema-qualified function calls are not allowed")
            continue
        if isinstance(inner, exp.Column) and isinstance(inner.this, exp.Dot):
            issues.append("schema-qualified function calls are not allowed")
            continue
        name = _function_name(func)
        if name and name in _DANGEROUS_FUNCTIONS:
            issues.append(f"dangerous function is not allowed: {name}")
    return issues
