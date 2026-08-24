"""Health API routes."""

from fastapi import APIRouter, Response

from app.schemas.health import HealthResponse
from app.services.health_service import DATABASE_CONNECTED, DATABASE_UNAVAILABLE, check_database

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(response: Response) -> HealthResponse:
    """Report API liveness and PostgreSQL connectivity.

    Returns HTTP 200 with ``status=healthy`` when the database responds,
    and HTTP 503 with ``status=unhealthy`` when it does not - the endpoint
    fails gracefully instead of raising.
    """
    database_status = check_database()
    if database_status == DATABASE_CONNECTED:
        return HealthResponse(status="healthy", database=DATABASE_CONNECTED)
    response.status_code = 503
    return HealthResponse(
        status="unhealthy",
        database=DATABASE_UNAVAILABLE,
        detail="PostgreSQL is unreachable; check DB_* environment settings.",
    )
