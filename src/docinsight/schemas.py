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


DocumentType = Literal["invoice", "contract", "other"]


class InvoiceFields(BaseModel):
    """Fields of an invoice; None when the document does not state them.

    Amounts are strings with a dot and two decimals ("1575.00"), dates are ISO
    ("2026-03-29"), OIBs and IBANs as printed on the document.
    """

    number: str | None = None
    issuer: str | None = None
    issuer_oib: str | None = None
    issuer_iban: str | None = None
    buyer: str | None = None
    buyer_oib: str | None = None
    issue_date: str | None = None
    due_date: str | None = None
    currency: str | None = None
    net: str | None = None
    vat: str | None = None
    total: str | None = None


class InvoiceChecks(BaseModel):
    """What our code verified. None = could not be checked (field or entities missing)."""

    issuer_oib_valid: bool | None = None  # ISO 7064 MOD 11,10 check digit
    buyer_oib_valid: bool | None = None
    issuer_iban_valid: bool | None = None  # ISO 7064 MOD 97-10
    amounts_consistent: bool | None = None  # net + vat == total (to the cent)
    dates_consistent: bool | None = None  # due_date >= issue_date
    # Number in the text; OIBs, IBAN, total and dates among the document's entities.
    values_in_document: bool | None = None
    names_in_entities: bool | None = None  # issuer and buyer among ORG/PERSON entities


class ContractFields(BaseModel):
    """Fields of a contract; None when the document does not state them (formats as above)."""

    client: str | None = None  # naručitelj: who orders and pays
    client_oib: str | None = None
    contractor: str | None = None  # izvršitelj: who does the work
    contractor_oib: str | None = None
    signed_date: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    fee: str | None = None
    currency: str | None = None


class ContractChecks(BaseModel):
    """What our code verified for a contract. None = could not be checked."""

    client_oib_valid: bool | None = None
    contractor_oib_valid: bool | None = None
    dates_consistent: bool | None = None  # signed_date <= start_date <= end_date
    values_in_document: bool | None = None  # OIBs, fee and dates among the entities
    names_in_entities: bool | None = None  # client and contractor among ORG/PERSON


class FieldsOut(BaseModel):
    """Type, summary and fields of one document, with the checks our code ran on them."""

    document_id: str
    document_type: DocumentType
    summary: str  # written by the LLM, not checked
    invoice: InvoiceFields | None  # only for an invoice
    contract: ContractFields | None  # only for a contract
    # The checks of InvoiceChecks or ContractChecks, by name; empty for other documents.
    checks: dict[str, bool | None]
    warnings: list[str]  # short descriptions of the checks that failed
    model: str
    cached: bool  # True: read from the database, no LLM call


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
