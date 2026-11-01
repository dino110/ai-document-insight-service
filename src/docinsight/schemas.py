"""Pydantic models for API requests and responses."""

from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from docinsight.nlp.ner import EntityLabel
from docinsight.qa.guard import Route

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


class PageEntityOut(BaseModel):
    """A named entity on a page: page.text[start:end] == text."""

    model_config = ConfigDict(from_attributes=True)

    text: str
    label: EntityLabel
    normalized: str  # comparable form: "1575.00", "2026-03-29", an IBAN without spaces
    start: int
    end: int


class PageOut(BaseModel):
    """Extracted text of one page, with the named entities found in it."""

    page_number: int
    text: str
    method: Literal["text_layer", "ocr"]
    entities: list[PageEntityOut] = Field(default_factory=list)


class DocumentDetail(DocumentOut):
    """A document with its extracted pages, in page order."""

    pages: list[PageOut]
    # Distinct normalised values per label, in order of appearance, at most 10 per label:
    # what the document is about at a glance ({"OIB": ["28196001332", ...], ...}).
    entity_summary: dict[EntityLabel, list[str]] = Field(default_factory=dict)


class UploadResponse(BaseModel):
    """Result of an upload: the session that owns the files and the stored documents."""

    session_id: str
    documents: list[DocumentOut]


class HistoryMessage(BaseModel):
    """An earlier message of the conversation. The client keeps the history, not the server."""

    role: Literal["user", "assistant"]
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class AskRequest(BaseModel):
    """A question about the caller's processed documents."""

    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)
    ]
    # None searches all of the caller's documents; ids of other sessions simply match nothing.
    document_ids: list[str] | None = Field(default=None, min_length=1, max_length=50)
    top_k: int | None = Field(default=None, ge=1, le=20)
    # The last messages before this question, oldest first: 3 questions and 3 answers.
    history: list[HistoryMessage] = Field(default_factory=list, max_length=6)


class SourceOut(BaseModel):
    """A chunk the answer cites: where it comes from and the start of its text."""

    document_id: str
    filename: str
    page_number: int
    snippet: str
    score: float


class EntityOut(BaseModel):
    """A named entity in the answer: answer[start:end] == text (character offsets)."""

    text: str
    label: EntityLabel
    start: int
    end: int


class AskResponse(BaseModel):
    """The answer, how it was produced, its sources and how long each step took."""

    answer: str
    # True when the answer comes from the caller's documents (or their list, for "meta").
    found: bool
    sources: list[SourceOut]
    # "rag" = search and LLM; other routes are answered locally, without either.
    route: Route
    model: str | None  # None when no LLM was called
    timings_ms: dict[str, float]  # only the steps that ran: empty unless route is "rag"
    # Only for a found "rag" answer; canned, document-list and "not found" texts have none.
    entities: list[EntityOut] = Field(default_factory=list)
