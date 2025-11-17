"""
Database connection management with connection pooling.

This module provides SQLAlchemy engine and session management with
connection pooling for efficient database access.
"""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
from typing import Generator
import logging

from src.utils.config import get_config

logger = logging.getLogger(__name__)

config = get_config()

# Database URL format: postgresql://user:password@host:port/database
DATABASE_URL = config.DATABASE_URL

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=config.DATABASE_POOL_SIZE,
    max_overflow=config.DATABASE_MAX_OVERFLOW,
    pool_timeout=config.DATABASE_POOL_TIMEOUT,
    pool_recycle=config.DATABASE_POOL_RECYCLE,
    pool_pre_ping=True,  # Check connection health before using
    echo=False,  # Set to True for SQL query logging
    future=True  # Use SQLAlchemy 2.0 style
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False
)


@event.listens_for(engine, "connect")
def receive_connect(dbapi_conn, connection_record):
    """Set up connection parameters on connect."""
    logger.debug("Database connection established")


@event.listens_for(engine, "close")
def receive_close(dbapi_conn, connection_record):
    """Log connection close."""
    logger.debug("Database connection closed")


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for getting database session.

    Usage in FastAPI:
        @app.get("/items")
        def get_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """
    Context manager for database session.

    Usage:
        with get_db_context() as db:
            db.query(Model).all()
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db():
    """Initialize database (create all tables)."""
    from src.database.models import Base
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized")


def check_db_connection() -> bool:
    """
    Check database connectivity.

    Returns:
        True if connection successful, False otherwise
    """
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False

