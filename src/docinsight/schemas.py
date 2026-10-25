"""Pydantic models for API requests and responses."""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

DocumentStatus = Literal["pending", "processed", "failed"]


class HealthResponse(BaseModel):
    """Liveness payload returned by the health endpoint."""

    status: str
    version: str


class DocumentOut(BaseModel):
    """Public view of a stored document."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    content_type: str
    size_bytes: int
    created_at: datetime
    status: DocumentStatus
    page_count: int | None
    error: str | None

    @field_validator("created_at")
    @classmethod
    def _assume_utc(cls, value: datetime) -> datetime:
        # SQLite drops the timezone; every timestamp is written in UTC, so restore it.
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


class PageOut(BaseModel):
    """Extracted text of one page."""

    model_config = ConfigDict(from_attributes=True)

    page_number: int
    text: str
    method: Literal["text_layer", "ocr"]


class DocumentDetail(DocumentOut):
    """A document with its extracted pages, in page order."""

    pages: list[PageOut]


class UploadResponse(BaseModel):
    """Result of an upload: the session that owns the files and the stored documents."""

    session_id: str
    documents: list[DocumentOut]
