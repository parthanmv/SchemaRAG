"""Pydantic schemas for health endpoints."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str
    database: str
    detail: str | None = None
