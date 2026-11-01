"""Shared FastAPI dependencies."""

import logging
import math
import re
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy import Engine
from sqlmodel import Session

from docinsight.config import Settings, get_settings
from docinsight.core.rate_limit import RateLimiter
from docinsight.extraction.ocr import OcrEngine
from docinsight.indexing.embeddings import Embedder
from docinsight.nlp.fields import FieldExtractor
from docinsight.nlp.ner import EntityExtractor
from docinsight.qa.generator import AnswerGenerator
from docinsight.qa.guard import QueryRouter
from docinsight.storage.db import get_engine, get_session

logger = logging.getLogger(__name__)

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


def get_generator(request: Request) -> AnswerGenerator:
    """The LLM generator created once in lifespan; tests replace it via dependency_overrides."""
    generator: AnswerGenerator = request.app.state.generator
    return generator


def get_router(request: Request) -> QueryRouter:
    """The question router created once in lifespan; tests replace it via dependency_overrides."""
    router: QueryRouter = request.app.state.router
    return router


def get_entity_extractor(request: Request) -> EntityExtractor:
    """The NER extractor created once in lifespan; tests replace it via dependency_overrides."""
    extractor: EntityExtractor = request.app.state.entity_extractor
    return extractor


def get_rate_limiter(request: Request) -> RateLimiter:
    """The rate limiter created once in lifespan; it keeps the counts between requests."""
    limiter: RateLimiter = request.app.state.rate_limiter
    return limiter


def get_field_extractor(request: Request) -> FieldExtractor:
    """The invoice field extractor created once in lifespan; tests replace it."""
    extractor: FieldExtractor = request.app.state.field_extractor
    return extractor


DbSession = Annotated[Session, Depends(get_session)]
DbEngine = Annotated[Engine, Depends(get_engine)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
OcrDep = Annotated[OcrEngine, Depends(get_ocr_engine)]
EmbedderDep = Annotated[Embedder, Depends(get_embedder)]
GeneratorDep = Annotated[AnswerGenerator, Depends(get_generator)]
RouterDep = Annotated[QueryRouter, Depends(get_router)]
EntityExtractorDep = Annotated[EntityExtractor, Depends(get_entity_extractor)]
FieldExtractorDep = Annotated[FieldExtractor, Depends(get_field_extractor)]
RateLimiterDep = Annotated[RateLimiter, Depends(get_rate_limiter)]


class RateLimit:
    """Dependency of the expensive routes: 429 once the session or the client IP is over its
    limit. Used as ``dependencies=[Depends(RateLimit("llm"))]``, so endpoints need no
    parameter for it.

    The session comes from the raw header, because this runs before the route has validated
    it; an invalid or missing header counts only against the IP. X-Forwarded-For is ignored:
    without a known proxy in front, any client could set it.
    """

    def __init__(self, budget: Literal["llm", "upload"]) -> None:
        self.budget = budget

    def __call__(self, request: Request, settings: SettingsDep, limiter: RateLimiterDep) -> None:
        per_session = (
            settings.rate_limit_llm_per_minute
            if self.budget == "llm"
            else settings.rate_limit_upload_per_minute
        )
        session_id = request.headers.get("x-session-id", "")
        checks: list[tuple[str, str, int]] = []
        if per_session and re.fullmatch(_SESSION_ID_PATTERN, session_id):
            checks.append(("session", f"{self.budget}:session:{session_id}", per_session))
        if settings.rate_limit_ip_per_minute and request.client is not None:
            checks.append(("ip", f"ip:{request.client.host}", settings.rate_limit_ip_per_minute))
        for kind, key, limit in checks:
            wait_s = limiter.hit(key, limit)
            if wait_s is not None:
                retry_after = max(1, math.ceil(wait_s))
                # Never the session id: whoever knows it can read the session's documents.
                logger.warning("Rate limit reached: %s budget, per %s", self.budget, kind)
                raise HTTPException(
                    status.HTTP_429_TOO_MANY_REQUESTS,
                    f"Too many requests, try again in {retry_after} s.",
                    headers={"Retry-After": str(retry_after)},
                )
