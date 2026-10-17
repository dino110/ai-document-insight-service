"""FastAPI application entry point."""

from fastapi import FastAPI
from pydantic import BaseModel

from docinsight import __version__


class HealthResponse(BaseModel):
    """Liveness payload returned by the health endpoint."""

    status: str
    version: str


app = FastAPI(title="Document Insight Service", version=__version__)


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    """Report that the service is running and which version is deployed."""
    return HealthResponse(status="ok", version=__version__)
