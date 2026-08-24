"""Health service: verifies PostgreSQL connectivity for the /health endpoint."""

import logging

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from app.db.session import engine

logger = logging.getLogger(__name__)

DATABASE_CONNECTED = "connected"
DATABASE_UNAVAILABLE = "unavailable"


def check_database(engine: Engine = engine) -> str:
    """Return :data:`DATABASE_CONNECTED` if a trivial query succeeds.

    Never raises: any SQLAlchemy error (including connection failures and
    timeouts) is caught, logged, and reported as
    :data:`DATABASE_UNAVAILABLE` so callers can degrade gracefully.
    """
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        logger.warning("Database health check failed: %s", exc)
        return DATABASE_UNAVAILABLE
    return DATABASE_CONNECTED
