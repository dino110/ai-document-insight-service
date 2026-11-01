"""FastAPI application entry point."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

from docinsight import __version__
from docinsight.api import ask, documents, fields, health
from docinsight.config import get_settings
from docinsight.extraction.ocr import create_ocr_engine
from docinsight.indexing.embeddings import create_embedder
from docinsight.nlp.fields import create_field_extractor
from docinsight.nlp.ner import create_entity_extractor
from docinsight.qa.generator import create_generator
from docinsight.qa.guard import RuleBasedRouter
from docinsight.storage.db import make_engine


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create shared resources once at startup and release them at shutdown."""
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    engine = make_engine(settings.database_url)
    app.state.engine = engine
    # Loading EasyOCR models takes seconds; keep the event loop free while it happens.
    app.state.ocr = await run_in_threadpool(create_ocr_engine, settings)
    # Cheap: API clients, no models to load.
    app.state.embedder = create_embedder(settings)
    app.state.generator = create_generator(settings)
    app.state.field_extractor = create_field_extractor(settings)
    app.state.router = RuleBasedRouter()  # compiles its patterns once
    # Loading the spaCy models takes about a second.
    app.state.entity_extractor = await run_in_threadpool(create_entity_extractor, settings)
    yield
    engine.dispose()


app = FastAPI(title="Document Insight Service", version=__version__, lifespan=lifespan)
app.include_router(health.router)
app.include_router(documents.router)
app.include_router(fields.router)
app.include_router(ask.router)
