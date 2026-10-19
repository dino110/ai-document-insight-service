"""Database engine and per-request sessions."""

from collections.abc import Iterator

from fastapi import Request
from sqlalchemy import Engine
from sqlmodel import Session, SQLModel, create_engine

# Importing the models registers their tables on SQLModel.metadata before create_all.
from docinsight.storage import models  # noqa: F401


def make_engine(url: str) -> Engine:
    """Create the engine and any missing tables."""
    # SQLite connections refuse by default to be used from a thread other than the one
    # that created them; FastAPI runs `def` endpoints and dependencies in a threadpool.
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    engine = create_engine(url, connect_args=connect_args)
    SQLModel.metadata.create_all(engine)
    return engine


def get_session(request: Request) -> Iterator[Session]:
    """Open one database session per request and close it afterwards."""
    engine: Engine = request.app.state.engine
    with Session(engine) as session:
        yield session
