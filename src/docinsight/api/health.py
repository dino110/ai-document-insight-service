"""Liveness endpoint."""

from fastapi import APIRouter

from docinsight import __version__
from docinsight.schemas import HealthResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Report that the service is running and which version is deployed."""
    return HealthResponse(status="ok", version=__version__)
