"""FastAPI application entry point."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from docinsight import __version__
from docinsight.api import documents, health
from docinsight.config import get_settings
from docinsight.storage.db import make_engine


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create shared resources once at startup and release them at shutdown."""
    settings = get_settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    engine = make_engine(settings.database_url)
    app.state.engine = engine
    yield
    engine.dispose()


app = FastAPI(title="Document Insight Service", version=__version__, lifespan=lifespan)
app.include_router(health.router)
app.include_router(documents.router)
