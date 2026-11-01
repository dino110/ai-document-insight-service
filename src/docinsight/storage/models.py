"""Database tables."""

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import JSON, Column, UniqueConstraint
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


class DocumentEntity(SQLModel, table=True):
    """A named entity on one page, found by local NER while the document was processed."""

    id: int | None = Field(default=None, primary_key=True)
    document_id: str = Field(foreign_key="document.id", index=True)
    owner_id: str = Field(index=True)  # copied from Document, like on chunks
    page_number: int  # 1-based
    label: str  # PERSON, ORG, LOC, MISC, OIB, IBAN, DATE or MONEY
    text: str  # exactly as on the page: page.text[start:end] == text
    normalized: str  # comparable form: "1575.00", "2026-03-29", an IBAN without spaces
    start: int  # character offsets in DocumentPage.text
    end: int
    # Backend and versions that produced it; reindex replaces entities of other extractors.
    extractor: str = Field(index=True)


class DocumentFields(SQLModel, table=True):
    """Type, summary and fields extracted by the LLM, and the checks our code ran on them.

    One row per document, created on the first GET /documents/{id}/fields; later calls
    read it instead of calling the LLM again. Reindex does not touch it. ``data`` holds
    {"format", "summary", "invoice", "contract"}; a row of an older format is replaced.
    """

    document_id: str = Field(foreign_key="document.id", primary_key=True)
    owner_id: str = Field(index=True)
    document_type: str  # "invoice", "contract" or "other"
    # JSON columns: SQLite stores text, SQLAlchemy converts to and from dicts and lists.
    # object: nested dicts of fields, a summary and the format number.
    data: dict[str, object] = Field(sa_column=Column(JSON, nullable=False))
    checks: dict[str, bool | None] = Field(sa_column=Column(JSON, nullable=False))
    warnings: list[str] = Field(sa_column=Column(JSON, nullable=False))
    model: str  # the LLM that extracted the fields
    created_at: datetime = Field(default_factory=_utcnow)


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
