"""Phase 5 read-only SQL execution service.

Takes a Phase 4 :class:`GeneratedSQL` and runs it against PostgreSQL *only*
after every gate passes:

1. generation succeeded (SQL present, no LLM error)
2. grounding passed (all tables/columns exist in Phase 2 metadata)
3. AST security validation passed (read-only policy, fail closed)

The connection itself uses a dedicated low-privilege database role configured
via ``EXEC_DB_USER`` / ``EXEC_DB_PASSWORD`` (SELECT-only on the six project
tables) plus session-level ``default_transaction_read_only`` and a
server-side ``statement_timeout`` - defence in depth behind the application
checks. When those credentials are not configured, execution is disabled and
every request reports :attr:`ExecutionStatus.DISABLED` without touching the
database.

Errors are mapped to typed statuses with sanitised messages; raw exception
text, DSNs, credentials and stack traces never leave this module.
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import (
    DBAPIError,
    SQLAlchemyError,
)
from urllib.parse import quote_plus

from app.core.config import get_settings
from app.rag.grounding import ground_sql
from app.rag.sql_security import SQLSecurityValidator
from app.rag.text_to_sql import GeneratedSQL, load_schema_metadata_snapshot
from app.rag.models import SchemaMetadata
# Phase 7 result processing (JSON-safe coercion + column-kind annotations).
from app.services.result_processing import infer_column_kinds, jsonable

logger = logging.getLogger(__name__)

#: PostgreSQL SQLSTATE cancelled by statement_timeout.
_QUERY_CANCELED_SQLSTATE = "57014"


class ExecutionStatus(str, Enum):
    """Terminal states of one execution attempt."""

    SUCCESS = "success"
    EMPTY = "empty_result"
    ROW_LIMIT_EXCEEDED = "row_limit_exceeded"

    # pre-execution gates (nothing was sent to PostgreSQL)
    INVALID_SQL = "invalid_sql"
    UNGROUNDED = "ungrounded"
    SECURITY_REJECTED = "security_rejected"
    DISABLED = "execution_disabled"

    # database-side failures
    STATEMENT_TIMEOUT = "statement_timeout"
    CONNECTION_ERROR = "connection_error"
    PERMISSION_DENIED = "permission_denied"
    EXECUTION_ERROR = "execution_error"


#: Statuses that mean "PostgreSQL never saw this statement".
NOT_EXECUTED_STATUSES = frozenset(
    {
        ExecutionStatus.INVALID_SQL,
        ExecutionStatus.UNGROUNDED,
        ExecutionStatus.SECURITY_REJECTED,
        ExecutionStatus.DISABLED,
    }
)


class QueryResult(BaseModel):
    """Typed end-to-end outcome of question -> SQL -> validation -> execution."""

    question: str
    sql: str | None = None
    model: str = ""
    grounded: bool = False
    security_allowed: bool = False
    security_issues: list[str] = Field(default_factory=list)
    execution_status: ExecutionStatus
    columns: list[str] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list)
    row_count: int = 0
    execution_time_ms: float | None = None
    retrieved_documents: list[str] = Field(default_factory=list)
    error: str | None = None
    #: PostgreSQL role that actually ran the statement (e.g. schemarag_reader).
    executed_as: str | None = None
    #: Phase 7 result processing: per-column display kinds
    #: (number/boolean/text/null/unknown); annotation only, values untouched.
    column_kinds: list[str] | None = None


#: Phase 7: JSON-safe coercion now lives in result_processing (single
#: implementation); the historical private name is kept as an alias.
_jsonable = jsonable


class SqlExecutionService:
    """Validates and executes generated SELECT statements safely."""

    def __init__(
        self,
        engine: Engine | None = None,
        *,
        max_rows: int | None = None,
        statement_timeout_ms: int | None = None,
        validator: SQLSecurityValidator | None = None,
        schema_metadata: SchemaMetadata | None = None,
    ) -> None:
        settings = get_settings()
        self.max_rows = max_rows if max_rows is not None else settings.sql_max_rows
        self.statement_timeout_ms = (
            statement_timeout_ms
            if statement_timeout_ms is not None
            else settings.sql_statement_timeout_ms
        )
        self.validator = validator or SQLSecurityValidator()
        self._metadata_override = schema_metadata
        self._engine = engine

    # ------------------------------------------------------------------
    def execute(self, generated: GeneratedSQL) -> QueryResult:
        """Run the full gate chain; returns a typed result, never raises."""
        base = {
            "question": generated.question,
            "model": generated.model,
            "retrieved_documents": list(generated.retrieved_documents),
        }

        # Gate 1: usable SQL from generation.
        if generated.error or not generated.sql or not generated.sql.strip():
            reason = generated.error or "no SQL was produced"
            return QueryResult(
                **base,
                grounded=False,
                security_allowed=False,
                execution_status=ExecutionStatus.INVALID_SQL,
                error=f"generation did not produce executable SQL ({reason})",
            )

        sql = generated.sql.strip()

        # Gate 2: grounding against real schema metadata.
        report = ground_sql(sql, self._schema_metadata())
        if not report.parsed or not report.grounded:
            issues = list(report.issues) or ["statement is not grounded in the schema"]
            return QueryResult(
                **base,
                sql=sql,
                grounded=False,
                security_allowed=False,
                security_issues=issues,
                execution_status=ExecutionStatus.UNGROUNDED,
                error="generated SQL references unknown tables/columns",
            )

        # Gate 3: AST security policy.
        sec = self.validator.validate(sql)
        if not sec.allowed:
            logger.info("Execution blocked by security validator: %s", sec.issues)
            return QueryResult(
                **base,
                sql=sql,
                grounded=True,
                security_allowed=False,
                security_issues=list(sec.issues),
                execution_status=ExecutionStatus.SECURITY_REJECTED,
                error="statement rejected by the read-only security policy",
            )

        # Gates passed - execute.
        started = time.perf_counter()
        status, columns, rows, error, executed_as = self._run(sql)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        return QueryResult(
            **base,
            sql=sec.normalized_sql or sql,
            grounded=True,
            security_allowed=True,
            security_issues=[],
            execution_status=status,
            columns=columns,
            rows=rows,
            row_count=len(rows),
            execution_time_ms=elapsed_ms,
            error=error,
            executed_as=executed_as,
            column_kinds=infer_column_kinds(columns, rows),
        )

    # ------------------------------------------------------------------
    def _schema_metadata(self) -> SchemaMetadata:
        if self._metadata_override is None:
            self._metadata_override = load_schema_metadata_snapshot()
        assert self._metadata_override is not None
        return self._metadata_override

    def _engine_or_none(self) -> Engine | None:
        if self._engine is not None:
            return self._engine
        settings = get_settings()
        if not settings.execution_enabled:
            return None
        return create_execution_engine(settings.exec_db_user, settings.exec_db_password)

    def _run(
        self, sql: str
    ) -> tuple[ExecutionStatus, list[str], list[list[Any]], str | None, str | None]:
        """Execute on the read-only engine.

        Returns ``(status, columns, rows, error, executed_as)``. The
        ``executed_as`` probe runs on the SAME connection so the result can
        prove which PostgreSQL role performed the execution.
        """
        engine = self._engine_or_none()
        if engine is None:
            return (
                ExecutionStatus.DISABLED,
                [],
                [],
                "execution is disabled; configure EXEC_DB_USER/EXEC_DB_PASSWORD "
                "with the dedicated read-only role",
                None,
            )
        try:
            with engine.connect() as connection:
                result = (
                    connection.execution_options(stream_results=True)
                    .execute(text(sql))
                )
                columns = [str(c) for c in result.keys()]
                fetched = [
                    [_jsonable(v) for v in row]
                    for row in result.fetchmany(self.max_rows + 1)
                ]
                result.close()

                # Identity probe on the same connection - evidence of the
                # executing role (never a credential).
                try:
                    executed_as = connection.execute(
                        text("SELECT current_user")
                    ).scalar()
                except SQLAlchemyError:
                    executed_as = None

            if len(fetched) > self.max_rows:
                return (
                    ExecutionStatus.ROW_LIMIT_EXCEEDED,
                    columns,
                    fetched[: self.max_rows],
                    f"result truncated at {self.max_rows} rows "
                    "(SQL_MAX_ROWS); refine the query for full results",
                    str(executed_as) if executed_as else None,
                )
            if not fetched:
                return ExecutionStatus.EMPTY, columns, [], None, str(executed_as) if executed_as else None
            return (
                ExecutionStatus.SUCCESS,
                columns,
                fetched,
                None,
                str(executed_as) if executed_as else None,
            )

        except DBAPIError as exc:
            status, _c, _r, err = self._map_dbapi_error(exc)
            return status, [], [], err, None
        except SQLAlchemyError as exc:
            logger.warning("Execution failed (%s)", type(exc).__name__)
            return (
                ExecutionStatus.CONNECTION_ERROR,
                [], [], "could not execute the query on the database", None,
            )

    def _map_dbapi_error(
        self, exc: DBAPIError
    ) -> tuple[ExecutionStatus, list[str], list[list[Any]], str | None]:
        sqlstate = getattr(exc.orig, "sqlstate", None) if exc.orig is not None else None
        if sqlstate == _QUERY_CANCELED_SQLSTATE:  # statement_timeout cancellation
            return (
                ExecutionStatus.STATEMENT_TIMEOUT, [], [],
                f"statement exceeded the {self.statement_timeout_ms}ms timeout",
            )
        if sqlstate == "42501":  # insufficient_privilege
            return (
                ExecutionStatus.PERMISSION_DENIED, [], [],
                "the execution role lacks permission for this statement",
            )
        if sqlstate == "25006":  # read_only_sql_transaction (defence in depth)
            return (
                ExecutionStatus.PERMISSION_DENIED, [], [],
                "the execution role cannot modify data (read-only)",
            )
        # Connection-class failures: no driver sqlstate at all (connect/DNS),
        # connection_exception family (08xxx), or fatal server shutdowns.
        if (
            sqlstate is None
            or sqlstate.startswith("08")
            or sqlstate in ("57P01", "57P02", "57P03")
        ):
            logger.warning("Connection-level failure (%s)", type(exc).__name__)
            return (
                ExecutionStatus.CONNECTION_ERROR, [], [],
                "could not reach the database with the execution role",
            )
        logger.warning("Statement failed (%s, sqlstate=%s)", type(exc).__name__, sqlstate)
        return (
            ExecutionStatus.EXECUTION_ERROR, [], [],
            "PostgreSQL rejected the statement",
        )


def create_execution_engine(
    user: str,
    password_secret,
    *,
    statement_timeout_ms: int | None = None,
) -> Engine:
    """Engine bound to the dedicated read-only execution role.

    Session hardening applied by the server for every connection:
      * ``statement_timeout``   - server-enforced per-statement deadline
      * ``default_transaction_read_only`` - writes rejected even if every
        other layer somehow allowed them
    """
    settings = get_settings()
    timeout = (
        int(statement_timeout_ms)
        if statement_timeout_ms is not None
        else int(settings.sql_statement_timeout_ms)
    )
    password = (
        password_secret.get_secret_value()
        if hasattr(password_secret, "get_secret_value")
        else str(password_secret)
    )
    url = (
        f"postgresql+psycopg://{quote_plus(user)}:{quote_plus(password)}"
        f"@{settings.db_host}:{settings.db_port}/{settings.db_name}"
    )
    options = (
        f"-c statement_timeout={timeout} "
        f"-c default_transaction_read_only=on"
    )
    return create_engine(
        url,
        echo=False,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 3, "options": options},
    )


@lru_cache(maxsize=1)
def _cached_engine() -> Engine:
    settings = get_settings()
    return create_execution_engine(settings.exec_db_user, settings.exec_db_password)


@lru_cache(maxsize=1)
def get_sql_execution_service() -> SqlExecutionService:
    """FastAPI dependency returning a cached execution service."""
    return SqlExecutionService(engine=None)
