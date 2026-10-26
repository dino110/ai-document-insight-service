"""Typed HTTP client for the Document Insight API, used by the Streamlit UI.

The UI is just another client of the public API: it never imports `docinsight`.
The TypedDicts below mirror the JSON responses (like TS types written from an OpenAPI spec);
they are hints for mypy only, nothing is validated at runtime.
"""

from typing import Literal, TypedDict

import httpx2

SESSION_HEADER = "X-Session-ID"

DocumentStatus = Literal["pending", "processed", "failed"]


class DocumentOut(TypedDict):
    id: str
    filename: str
    content_type: str
    size_bytes: int
    created_at: str  # ISO 8601, UTC
    status: DocumentStatus
    page_count: int | None
    error: str | None


class PageOut(TypedDict):
    page_number: int
    text: str
    method: Literal["text_layer", "ocr"]


class DocumentDetail(DocumentOut):
    pages: list[PageOut]


class UploadResponse(TypedDict):
    session_id: str
    documents: list[DocumentOut]


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


def _send(
    client: httpx2.Client,
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    files: list[tuple[str, UploadFile]] | None = None,
) -> httpx2.Response:
    """Send a request and turn every failure into ApiError."""
    try:
        response = client.request(method, url, headers=headers, files=files)
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
