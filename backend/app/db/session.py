"""SQLAlchemy engine and session factory.

The engine is created lazily from :mod:`app.core.config` settings so that the
module can be imported in any environment (tests included) without touching
the database.
"""

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings


def create_database_engine() -> Engine:
    """Create the SQLAlchemy engine for the configured PostgreSQL database."""
    settings = get_settings()
    return create_engine(
        settings.database_url,
        echo=settings.db_echo,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 3},
    )


engine: Engine = create_database_engine()

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db() -> Iterator[Session]:
    """FastAPI dependency that yields a database session per request."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
