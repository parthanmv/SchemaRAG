"""Pydantic schemas for the Phase 5 POST /api/query endpoint."""

from pydantic import BaseModel, Field

from app.services.sql_execution import ExecutionStatus, QueryResult


class QueryRequest(BaseModel):
    """Request body for POST /api/query."""

    question: str = Field(min_length=3, max_length=500)


# The full typed execution outcome doubles as the HTTP response body.
QueryResponse = QueryResult

__all__ = ["ExecutionStatus", "QueryRequest", "QueryResult", "QueryResponse"]
