"""Tests for the UI's API client. MockTransport replaces the network, so no server is needed."""

from collections.abc import Callable

import httpx2
import pytest
from api_client import ApiError, get_document, list_documents, make_client, upload

Handler = Callable[[httpx2.Request], httpx2.Response]

DOCUMENT = {
    "id": "doc-1",
    "filename": "invoice.pdf",
    "content_type": "application/pdf",
    "size_bytes": 10,
    "created_at": "2026-10-25T10:00:00Z",
    "status": "pending",
    "page_count": None,
    "error": None,
}


def _client(handler: Handler) -> httpx2.Client:
    return httpx2.Client(base_url="http://api.test", transport=httpx2.MockTransport(handler))


def test_make_client_points_at_base_url() -> None:
    client = make_client("http://api.example:8000")

    assert str(client.base_url) == "http://api.example:8000"


def test_upload_without_session_sends_no_session_header() -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(201, json={"session_id": "new", "documents": [DOCUMENT]})

    result = upload(_client(handler), [("invoice.pdf", b"%PDF-1.7", "application/pdf")], None)

    [request] = requests
    assert request.method == "POST"
    assert request.url.path == "/upload"
    assert "X-Session-ID" not in request.headers
    # The multipart field must be called "files", like the parameter of POST /upload.
    body = request.read()
    assert b'name="files"; filename="invoice.pdf"' in body
    assert b"%PDF-1.7" in body
    assert result["session_id"] == "new"
    assert result["documents"][0]["filename"] == "invoice.pdf"


def test_upload_with_session_sends_header_and_every_file() -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(201, json={"session_id": "abc", "documents": []})

    files = [("a.pdf", b"%PDF-a", "application/pdf"), ("b.png", b"png-bytes", "image/png")]
    upload(_client(handler), files, "abc")

    [request] = requests
    assert request.headers["X-Session-ID"] == "abc"
    assert request.read().count(b'name="files"') == 2


def test_list_documents_sends_session_and_returns_list() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/documents"
        assert request.headers["X-Session-ID"] == "abc"
        return httpx2.Response(200, json=[DOCUMENT])

    documents = list_documents(_client(handler), "abc")

    assert [d["id"] for d in documents] == ["doc-1"]


def test_get_document_requests_the_document_path() -> None:
    detail = {**DOCUMENT, "status": "processed", "page_count": 1, "pages": []}

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/documents/doc-1"
        return httpx2.Response(200, json=detail)

    assert get_document(_client(handler), "abc", "doc-1")["status"] == "processed"


def test_unreachable_api_raises_api_error_without_status() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("Connection refused", request=request)

    with pytest.raises(ApiError) as excinfo:
        list_documents(_client(handler), "abc")

    assert excinfo.value.status is None
    assert excinfo.value.detail == "API nije dostupan na http://api.test"


def test_validation_error_list_becomes_one_line() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        detail = [{"loc": ["header", "x-session-id"], "msg": "String should match pattern"}]
        return httpx2.Response(422, json={"detail": detail})

    with pytest.raises(ApiError) as excinfo:
        list_documents(_client(handler), "bad id")

    assert excinfo.value.status == 422
    assert excinfo.value.detail == "String should match pattern"


def test_non_json_error_falls_back_to_status_code() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(502, text="<html>Bad Gateway</html>")

    with pytest.raises(ApiError) as excinfo:
        list_documents(_client(handler), "abc")

    assert (excinfo.value.status, excinfo.value.detail) == (502, "HTTP 502")
