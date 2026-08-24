"""FastAPI application entry point."""

import logging
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import generate_sql as generate_sql_routes
from app.api.routes import health as health_routes
from app.api.routes import query as query_routes
from app.core.config import get_settings
from app.db.session import engine
from app.services.health_service import DATABASE_CONNECTED, check_database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Log database reachability at startup without blocking startup on failure."""
    settings = get_settings()
    target = f"{settings.db_user}@{settings.db_host}:{settings.db_port}/{settings.db_name}"

    def _log_db_status() -> None:
        if check_database(engine) == DATABASE_CONNECTED:
            logger.info("Database connected: %s", target)
        else:
            logger.warning("Database unreachable; /health will report unhealthy.")

    # Non-blocking: a slow/unreachable DB must not delay server startup.
    threading.Thread(target=_log_db_status, name="startup-db-check", daemon=True).start()
    yield


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(
        title="SchemaRAG API",
        description=(
            "RAG-powered natural-language interface for a PostgreSQL college "
            "database. Phase 4 adds grounded Text-to-SQL generation via "
            "POST /api/generate-sql (SQL is generated, never executed). "
            "Phase 5 adds POST /api/query: security-validated, read-only "
            "execution through a dedicated low-privilege database role."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    # CORS for the local Vite dev server (Phase 6 frontend). The browser
    # talks ONLY to this API; PostgreSQL is never exposed to the frontend.
    # The regex additionally covers Vite's port auto-increment (5174, 5175,
    # ...) when 5173 is already taken - still strictly loopback-only, never
    # a wildcard origin.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.include_router(health_routes.router)
    app.include_router(generate_sql_routes.router)
    app.include_router(query_routes.router)

    @app.get("/", tags=["meta"])
    def root() -> dict[str, str]:
        """Service metadata pointer."""
        return {"service": "SchemaRAG API", "docs": "/docs", "health": "/health"}

    return app


app = create_app()
