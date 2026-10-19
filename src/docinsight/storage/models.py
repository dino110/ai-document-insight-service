"""Database tables."""

from datetime import UTC, datetime
from uuid import uuid4

from sqlmodel import Field, SQLModel


def _new_id() -> str:
    return uuid4().hex


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Document(SQLModel, table=True):
    """An uploaded file. Internal fields (stored_path, sha256, owner_id) never leave the API."""

    id: str = Field(default_factory=_new_id, primary_key=True)
    owner_id: str = Field(index=True)
    filename: str
    content_type: str
    size_bytes: int
    sha256: str = Field(index=True)
    stored_path: str
    created_at: datetime = Field(default_factory=_utcnow)
