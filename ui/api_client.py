"""Typed HTTP client for the Document Insight API, used by the Streamlit UI.

The UI is just another client of the public API: it never imports `docinsight`.
The TypedDicts below mirror the JSON responses (like TS types written from an OpenAPI spec);
they are hints for mypy only, nothing is validated at runtime.
"""

from collections.abc import Sequence
from typing import Literal, TypedDict

import httpx2

SESSION_HEADER = "X-Session-ID"
# /ask and the first /fields call wait for the LLM (and the SDK's retries).
ASK_TIMEOUT_S = 90.0
FIELDS_TIMEOUT_S = 90.0

DocumentStatus = Literal["pending", "processed", "failed"]
# How the API produced an answer: "rag" = search + LLM, the rest without an LLM.
Route = Literal[
    "rejected_injection",
    "rejected_abuse",
    "legal_advice",
    "out_of_scope",
    "smalltalk",
    "meta",
    "rag",
]


# Labels of named entities: names from the spaCy models, the rest from rules.
EntityLabel = Literal["PERSON", "ORG", "LOC", "MISC", "OIB", "IBAN", "DATE", "MONEY"]


class DocumentOut(TypedDict):
    id: str
    filename: str
    content_type: str
    size_bytes: int
    created_at: str  # ISO 8601, UTC
    status: DocumentStatus
    page_count: int | None
    error: str | None


class EntityOut(TypedDict):
    text: str
    label: EntityLabel
    start: int  # character offsets in the answer or page text: text[start:end] == text
    end: int


class PageEntityOut(EntityOut):
    normalized: str  # "1575.00", "2026-03-29", an IBAN without spaces


class PageOut(TypedDict):
    page_number: int
    text: str
    method: Literal["text_layer", "ocr"]
    entities: list[PageEntityOut]


class DocumentDetail(DocumentOut):
    pages: list[PageOut]
    # Distinct normalised values per label (at most 10), labels in a fixed order.
    entity_summary: dict[EntityLabel, list[str]]


class InvoiceFields(TypedDict):
    number: str | None
    issuer: str | None
    issuer_oib: str | None
    issuer_iban: str | None
    buyer: str | None
    buyer_oib: str | None
    issue_date: str | None  # ISO
    due_date: str | None
    currency: str | None
    net: str | None  # "1575.00"
    vat: str | None
    total: str | None


class ContractFields(TypedDict):
    client: str | None  # naručitelj
    client_oib: str | None
    contractor: str | None  # izvršitelj
    contractor_oib: str | None
    signed_date: str | None  # ISO
    start_date: str | None
    end_date: str | None
    fee: str | None  # "6300.00"
    currency: str | None


class FieldsOut(TypedDict):
    document_id: str
    document_type: Literal["invoice", "contract", "other"]
    summary: str  # written by the LLM, not checked
    invoice: InvoiceFields | None
    contract: ContractFields | None
    checks: dict[str, bool | None]  # names depend on the type; None: not checked
    warnings: list[str]
    model: str
    cached: bool  # read from the database, no LLM call


class UploadResponse(TypedDict):
    session_id: str
    documents: list[DocumentOut]


class SourceOut(TypedDict):
    document_id: str
    filename: str
    page_number: int
    snippet: str  # start of the cited chunk
    score: float


class HistoryMessage(TypedDict):
    role: Literal["user", "assistant"]
    text: str


class AskResponse(TypedDict):
    answer: str
    found: bool  # False: the documents do not contain the answer (and sources is empty)
    sources: list[SourceOut]
    route: Route
    model: str | None  # None when no LLM was called
    timings_ms: dict[str, float]  # "retrieval", "generation", "ner"; empty unless route is "rag"
    entities: list[EntityOut]  # only for a found "rag" answer


# (filename, content, content type), as handed over by st.file_uploader.
UploadFile = tuple[str, bytes, str]


class ApiError(Exception):
    """Any failed API call. ``status`` is None when the API could not be reached."""

    def __init__(self, status: int | None, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


def make_client(base_url: str) -> httpx2.Client:
    # OCR runs in the background, so even large uploads return quickly; 30 s is generous.
    return httpx2.Client(base_url=base_url, timeout=30.0)


def upload(
    client: httpx2.Client, files: list[UploadFile], session_id: str | None
) -> UploadResponse:
    """Upload files; without a session id the API creates a new session and returns it."""
    headers = {SESSION_HEADER: session_id} if session_id else {}
    # The multipart field name must match the `files` parameter of POST /upload.
    multipart = [("files", file) for file in files]
    response = _send(client, "POST", "/upload", headers=headers, files=multipart)
    data: UploadResponse = response.json()
    return data


def list_documents(client: httpx2.Client, session_id: str) -> list[DocumentOut]:
    response = _send(client, "GET", "/documents", headers={SESSION_HEADER: session_id})
    data: list[DocumentOut] = response.json()
    return data


def get_document(client: httpx2.Client, session_id: str, document_id: str) -> DocumentDetail:
    response = _send(
        client, "GET", f"/documents/{document_id}", headers={SESSION_HEADER: session_id}
    )
    data: DocumentDetail = response.json()
    return data


def get_fields(client: httpx2.Client, session_id: str, document_id: str) -> FieldsOut:
    """Type, summary, fields and checks; the first call per document waits for the LLM."""
    response = _send(
        client,
        "GET",
        f"/documents/{document_id}/fields",
        headers={SESSION_HEADER: session_id},
        timeout=FIELDS_TIMEOUT_S,
    )
    data: FieldsOut = response.json()
    return data


def ask(
    client: httpx2.Client,
    session_id: str,
    question: str,
    history: Sequence[HistoryMessage] = (),
) -> AskResponse:
    """Ask a question about the session's processed documents.

    ``history`` is the earlier conversation (at most 6 messages): the API keeps none.
    """
    response = _send(
        client,
        "POST",
        "/ask",
        headers={SESSION_HEADER: session_id},
        json={"question": question, "history": list(history)},
        timeout=ASK_TIMEOUT_S,
    )
    data: AskResponse = response.json()
    return data


def _send(
    client: httpx2.Client,
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    files: list[tuple[str, UploadFile]] | None = None,
    json: dict[str, object] | None = None,
    timeout: float | None = None,
) -> httpx2.Response:
    """Send a request and turn every failure into ApiError."""
    try:
        response = client.request(
            method,
            url,
            headers=headers,
            files=files,
            json=json,
            # None keeps the client's default timeout.
            timeout=timeout if timeout is not None else client.timeout,
        )
    except httpx2.TransportError as exc:
        raise ApiError(None, f"API nije dostupan na {client.base_url}") from exc
    if response.status_code >= 400:
        raise ApiError(response.status_code, _error_detail(response))
    return response


def _error_detail(response: httpx2.Response) -> str:
    """The message from FastAPI's ``{"detail": ...}`` body, as one short line."""
    try:
        body = response.json()
    except ValueError:
        return f"HTTP {response.status_code}"
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):  # 422: a list of validation errors
        messages = [item.get("msg", "") for item in detail if isinstance(item, dict)]
        return "; ".join(message for message in messages if message) or "Neispravan zahtjev"
    return f"HTTP {response.status_code}"
