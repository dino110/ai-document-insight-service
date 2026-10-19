"""Pydantic models for API requests and responses."""

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, field_validator


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

    @field_validator("created_at")
    @classmethod
    def _assume_utc(cls, value: datetime) -> datetime:
        # SQLite drops the timezone; every timestamp is written in UTC, so restore it.
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


class UploadResponse(BaseModel):
    """Result of an upload: the session that owns the files and the stored documents."""

    session_id: str
    documents: list[DocumentOut]
