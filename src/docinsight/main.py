"""FastAPI application entry point."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

from docinsight import __version__
from docinsight.api import documents, health
from docinsight.config import get_settings
from docinsight.extraction.ocr import create_ocr_engine
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
    yield
    engine.dispose()


app = FastAPI(title="Document Insight Service", version=__version__, lifespan=lifespan)
app.include_router(health.router)
app.include_router(documents.router)
