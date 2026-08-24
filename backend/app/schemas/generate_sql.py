"""Pydantic schemas for the Text-to-SQL endpoint."""

from pydantic import BaseModel, Field


class GenerateSQLRequest(BaseModel):
    """Request body for POST /api/generate-sql."""

    question: str = Field(min_length=3, max_length=500)


class GenerateSQLResponse(BaseModel):
    """Response body for POST /api/generate-sql.

    The generated SQL is returned for inspection only - it is never executed
    by this service (execution arrives with Phase 5).
    """

    question: str
    #: Phase 7: normalised question used for retrieval/prompting (may equal
    #: ``question`` when no transformation was needed).
    processed_question: str | None = None
    sql: str | None
    model: str
    grounded: bool
    retrieved_documents: list[str]
    retrieval_scores: list[float] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    error: str | None = None
