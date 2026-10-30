"""Shared FastAPI dependencies."""

from typing import Annotated
from uuid import uuid4

from fastapi import Depends, Header, Request
from sqlalchemy import Engine
from sqlmodel import Session

from docinsight.config import Settings, get_settings
from docinsight.extraction.ocr import OcrEngine
from docinsight.indexing.embeddings import Embedder
from docinsight.storage.db import get_engine, get_session

# The session id becomes a directory name, so only allow characters that cannot form a path.
_SESSION_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


def get_or_create_session_id(
    x_session_id: Annotated[str | None, Header(pattern=_SESSION_ID_PATTERN)] = None,
) -> str:
    """Owner for uploads: the X-Session-ID header, or a new session if it is missing."""
    return x_session_id or uuid4().hex


def require_session_id(
    x_session_id: Annotated[str, Header(pattern=_SESSION_ID_PATTERN)],
) -> str:
    """Owner for reads: the X-Session-ID header is required."""
    return x_session_id


def get_ocr_engine(request: Request) -> OcrEngine:
    """The OCR engine created once in lifespan; tests replace it via dependency_overrides."""
    ocr: OcrEngine = request.app.state.ocr
    return ocr


def get_embedder(request: Request) -> Embedder:
    """The embedder created once in lifespan; tests replace it via dependency_overrides."""
    embedder: Embedder = request.app.state.embedder
    return embedder


DbSession = Annotated[Session, Depends(get_session)]
DbEngine = Annotated[Engine, Depends(get_engine)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
OcrDep = Annotated[OcrEngine, Depends(get_ocr_engine)]
EmbedderDep = Annotated[Embedder, Depends(get_embedder)]
