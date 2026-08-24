"""POST /api/query - natural-language question to executed, read-only result.

Full Phase 5 pipeline: RAG generation (Phase 4) -> grounding -> AST security
validation -> read-only PostgreSQL execution -> typed result.

HTTP mapping:
* LLM unavailable            -> 503
* other LLM failure          -> 502
* no/ungrounded SQL          -> 400
* security rejection         -> 403
* statement timeout          -> 504
* DB unavailable/disabled    -> 503
* unexpected execution error -> 500

Internal exception details are never returned; only short sanitised messages.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.rag.llm.base import LLMError, LLMUnavailableError
from app.schemas.query import QueryRequest, QueryResponse
from app.services.sql_execution import (
    ExecutionStatus,
    NOT_EXECUTED_STATUSES,
    QueryResult,
    SqlExecutionService,
    get_sql_execution_service,
)
from app.rag.text_to_sql import TextToSQLService, get_sql_generation_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["query"])


_STATUS_TO_HTTP = {
    ExecutionStatus.SUCCESS: status.HTTP_200_OK,
    ExecutionStatus.EMPTY: status.HTTP_200_OK,
    # Truncation is a usable partial answer; the payload carries the status.
    ExecutionStatus.ROW_LIMIT_EXCEEDED: status.HTTP_200_OK,
    ExecutionStatus.STATEMENT_TIMEOUT: 504,  # Gateway Timeout
    ExecutionStatus.CONNECTION_ERROR: status.HTTP_503_SERVICE_UNAVAILABLE,
    ExecutionStatus.DISABLED: status.HTTP_503_SERVICE_UNAVAILABLE,
    ExecutionStatus.PERMISSION_DENIED: status.HTTP_503_SERVICE_UNAVAILABLE,
    ExecutionStatus.EXECUTION_ERROR: status.HTTP_500_INTERNAL_SERVER_ERROR,
}


@router.post("/query", response_model=QueryResponse)
def run_query(
    request: QueryRequest,
    generation: TextToSQLService = Depends(get_sql_generation_service),
    executor: SqlExecutionService = Depends(get_sql_execution_service),
) -> QueryResult:
    """Answer a question with data: generate SQL, validate it, execute read-only."""
    try:
        generated = generation.generate(request.question)
    except LLMUnavailableError as exc:
        logger.warning("LLM unavailable during query: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM backend is unavailable; check LLM_PROVIDER, "
            "GEMINI_API_KEY and GEMINI_MODEL settings.",
        ) from exc
    except LLMError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM backend error while generating SQL.",
        ) from exc

    if generated.sql is None or not generated.grounded:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": generated.error or "no executable SQL was generated",
                "issues": list(generated.issues)[:10],
            },
        )

    result = executor.execute(generated)

    if result.execution_status in NOT_EXECUTED_STATUSES:
        # Security rejections are surfaced explicitly (the pre-check above only
        # covers grounding; the validator runs inside the executor).
        if result.execution_status is ExecutionStatus.SECURITY_REJECTED:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "security_rejected",
                    "issues": result.security_issues[:10],
                },
            )
        raise HTTPException(
            status_code=_STATUS_TO_HTTP[result.execution_status],
            detail=result.error or result.execution_status.value,
        )

    http_code = _STATUS_TO_HTTP.get(result.execution_status)
    if http_code != status.HTTP_200_OK:
        raise HTTPException(
            status_code=http_code,
            detail=result.error or "query could not be completed",
        )

    return result
