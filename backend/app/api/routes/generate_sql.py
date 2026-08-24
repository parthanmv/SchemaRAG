"""POST /api/generate-sql - natural-language question to grounded SELECT SQL.

The endpoint returns the generated SQL for inspection. It never executes
SQL and never touches database rows; that is Phase 5 scope.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.rag.llm.base import LLMError, LLMUnavailableError
from app.rag.text_to_sql import GeneratedSQL, TextToSQLService, get_sql_generation_service
from app.schemas.generate_sql import GenerateSQLRequest, GenerateSQLResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["text-to-sql"])


@router.post("/generate-sql", response_model=GenerateSQLResponse)
def generate_sql(
    request: GenerateSQLRequest,
    service: TextToSQLService = Depends(get_sql_generation_service),
) -> GenerateSQLResponse:
    """Convert a natural-language question into one grounded PostgreSQL SELECT."""
    try:
        result: GeneratedSQL = service.generate(request.question)
    except LLMUnavailableError as exc:
        logger.warning("LLM unavailable while generating SQL: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM backend is unavailable; check LLM_PROVIDER, "
            "GEMINI_API_KEY and GEMINI_MODEL settings.",
        ) from exc
    except LLMError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM backend error: {exc}",
        ) from exc

    return GenerateSQLResponse(**result.model_dump())
