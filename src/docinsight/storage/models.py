"""Database tables."""

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import UniqueConstraint
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
    # "pending" until background extraction finishes, then "processed" or "failed".
    status: str = "pending"
    page_count: int | None = None
    error: str | None = None  # short, user-facing reason when status is "failed"


class DocumentPage(SQLModel, table=True):
    """Extracted text of one page of a document."""

    __table_args__ = (UniqueConstraint("document_id", "page_number"),)

    id: int | None = Field(default=None, primary_key=True)
    document_id: str = Field(foreign_key="document.id", index=True)
    page_number: int  # 1-based
    text: str
    method: str  # "text_layer" or "ocr"


class DocumentChunk(SQLModel, table=True):
    """A piece of one page with its embedding: the unit that search finds and answers cite."""

    __table_args__ = (UniqueConstraint("document_id", "chunk_index"),)

    id: int | None = Field(default=None, primary_key=True)
    document_id: str = Field(foreign_key="document.id", index=True)
    # Copied from Document, so one owner's chunks are selected by an index on this table.
    owner_id: str = Field(index=True)
    page_number: int  # 1-based, the page that a citation points to
    chunk_index: int  # position in the document
    text: str
    embedding: bytes  # float32 vector as raw bytes: array.tobytes() / np.frombuffer()
    # Vectors of different models are not comparable; search only uses the current one.
    embedding_model: str = Field(index=True)
