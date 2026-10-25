"""API tests for uploading, listing and reading documents."""

from pathlib import Path

import pytest
from factories import FAKE_OCR_TEXT, FakeOcrEngine, make_image_pdf, make_png, make_text_pdf
from fastapi.testclient import TestClient
from sqlmodel import Session

from docinsight.api.deps import get_ocr_engine
from docinsight.config import get_settings
from docinsight.extraction.ocr import UnavailableOcrEngine
from docinsight.main import app
from docinsight.processing import process_document

PDF = b"%PDF-1.4\n1 0 obj << >> endobj\n%%EOF\n"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 32
TEXT = b"just some plain text\n"

FileField = tuple[str, tuple[str, bytes, str]]


def _file(name: str, content: bytes, content_type: str = "application/octet-stream") -> FileField:
    return ("files", (name, content, content_type))


def _upload(
    client: TestClient, *files: FileField, session_id: str | None = None
) -> dict[str, object]:
    headers = {"X-Session-ID": session_id} if session_id else {}
    response = client.post("/upload", files=list(files), headers=headers)
    assert response.status_code == 201, response.text
    body: dict[str, object] = response.json()
    return body


def test_upload_single_pdf_creates_session(client: TestClient) -> None:
    response = client.post("/upload", files=[_file("invoice.pdf", PDF, "application/pdf")])

    assert response.status_code == 201
    body = response.json()
    assert len(body["session_id"]) == 32
    [document] = body["documents"]
    assert document["filename"] == "invoice.pdf"
    assert document["content_type"] == "application/pdf"
    assert document["size_bytes"] == len(PDF)
    assert document["created_at"].endswith("Z") or "+00:00" in document["created_at"]


def test_upload_multiple_files_with_existing_session(client: TestClient) -> None:
    body = _upload(
        client,
        _file("a.pdf", PDF),
        _file("b.png", PNG),
        _file("c.jpg", JPEG),
        session_id="session-a",
    )

    assert body["session_id"] == "session-a"
    documents = body["documents"]
    assert isinstance(documents, list)
    assert [d["content_type"] for d in documents] == ["application/pdf", "image/png", "image/jpeg"]


def test_response_does_not_leak_internal_fields(client: TestClient) -> None:
    body = _upload(client, _file("a.pdf", PDF))
    documents = body["documents"]
    assert isinstance(documents, list)

    assert set(documents[0]) == {
        "id",
        "filename",
        "content_type",
        "size_bytes",
        "created_at",
        "status",
        "page_count",
        "error",
    }


def test_file_is_stored_under_generated_name(client: TestClient, data_dir: Path) -> None:
    body = _upload(client, _file("../../etc/passwd.pdf", PDF), session_id="session-a")
    documents = body["documents"]
    assert isinstance(documents, list)
    document = documents[0]

    stored = data_dir / "uploads" / "session-a" / f"{document['id']}.pdf"
    assert stored.read_bytes() == PDF
    assert document["filename"] == "passwd.pdf"
    assert list((data_dir / "uploads").iterdir()) == [data_dir / "uploads" / "session-a"]


def test_text_file_is_rejected(client: TestClient) -> None:
    response = client.post("/upload", files=[_file("notes.txt", TEXT, "text/plain")])

    assert response.status_code == 415


def test_text_file_renamed_to_pdf_is_rejected(client: TestClient) -> None:
    response = client.post("/upload", files=[_file("fake.pdf", TEXT, "application/pdf")])

    assert response.status_code == 415


def test_empty_file_is_rejected(client: TestClient) -> None:
    response = client.post("/upload", files=[_file("empty.pdf", b"")])

    assert response.status_code == 400


def test_too_large_file_is_rejected(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCINSIGHT_MAX_UPLOAD_MB", "1")
    get_settings.cache_clear()
    too_large = PDF + b"\x00" * (1024 * 1024 + 1 - len(PDF))

    response = client.post("/upload", files=[_file("big.pdf", too_large)])

    assert response.status_code == 413


def test_too_many_files_are_rejected(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCINSIGHT_MAX_FILES_PER_REQUEST", "2")
    get_settings.cache_clear()

    response = client.post("/upload", files=[_file(f"{i}.pdf", PDF) for i in range(3)])

    assert response.status_code == 400


def test_one_bad_file_rejects_the_whole_request(client: TestClient, data_dir: Path) -> None:
    response = client.post(
        "/upload",
        files=[_file("good.pdf", PDF), _file("bad.txt", TEXT)],
        headers={"X-Session-ID": "session-a"},
    )

    assert response.status_code == 415
    assert client.get("/documents", headers={"X-Session-ID": "session-a"}).json() == []
    assert not (data_dir / "uploads" / "session-a").exists()


def test_failed_commit_removes_written_files(
    client: TestClient, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_commit(self: Session) -> None:
        raise RuntimeError("database is unavailable")

    monkeypatch.setattr(Session, "commit", failing_commit)

    with pytest.raises(RuntimeError, match="database is unavailable"):
        client.post(
            "/upload",
            files=[_file("a.pdf", PDF), _file("b.pdf", PDF)],
            headers={"X-Session-ID": "session-a"},
        )

    assert list((data_dir / "uploads" / "session-a").iterdir()) == []


def test_invalid_session_id_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/upload", files=[_file("a.pdf", PDF)], headers={"X-Session-ID": "../escape"}
    )

    assert response.status_code == 422


def test_list_documents_requires_session_header(client: TestClient) -> None:
    response = client.get("/documents")

    assert response.status_code == 422


def test_list_documents_newest_first(client: TestClient) -> None:
    _upload(client, _file("first.pdf", PDF), session_id="session-a")
    _upload(client, _file("second.pdf", PDF), session_id="session-a")

    response = client.get("/documents", headers={"X-Session-ID": "session-a"})

    assert response.status_code == 200
    assert [d["filename"] for d in response.json()] == ["second.pdf", "first.pdf"]


def test_sessions_do_not_see_each_others_documents(client: TestClient) -> None:
    _upload(client, _file("a.pdf", PDF), session_id="session-a")
    _upload(client, _file("b.pdf", PDF), session_id="session-b")

    a_docs = client.get("/documents", headers={"X-Session-ID": "session-a"}).json()
    b_docs = client.get("/documents", headers={"X-Session-ID": "session-b"}).json()

    assert [d["filename"] for d in a_docs] == ["a.pdf"]
    assert [d["filename"] for d in b_docs] == ["b.pdf"]


# --- Processing and document detail -----------------------------------------------


def _upload_one(client: TestClient, name: str, content: bytes, session_id: str = "owner") -> str:
    body = _upload(client, _file(name, content), session_id=session_id)
    documents = body["documents"]
    assert isinstance(documents, list)
    [document] = documents
    assert document["status"] == "pending"  # the response is sent before processing
    document_id: str = document["id"]
    return document_id


def _detail(client: TestClient, document_id: str, session_id: str = "owner") -> dict[str, object]:
    response = client.get(f"/documents/{document_id}", headers={"X-Session-ID": session_id})
    assert response.status_code == 200, response.text
    body: dict[str, object] = response.json()
    return body


def test_digital_pdf_is_processed_from_text_layer(
    client: TestClient, fake_ocr: FakeOcrEngine
) -> None:
    content = make_text_pdf("Invoice 2026-001. Total amount due: 1.575,00 EUR", pages=2)
    document_id = _upload_one(client, "invoice.pdf", content)

    # TestClient runs background tasks before returning the response, so processing is done.
    detail = _detail(client, document_id)

    assert detail["status"] == "processed"
    assert detail["page_count"] == 2
    assert detail["error"] is None
    pages = detail["pages"]
    assert isinstance(pages, list)
    assert [p["page_number"] for p in pages] == [1, 2]
    assert {p["method"] for p in pages} == {"text_layer"}
    assert "1.575,00 EUR" in pages[0]["text"]
    assert set(pages[0]) == {"page_number", "text", "method"}
    assert fake_ocr.calls == 0


def test_image_is_processed_with_ocr(client: TestClient, fake_ocr: FakeOcrEngine) -> None:
    document_id = _upload_one(client, "photo.png", make_png())

    detail = _detail(client, document_id)

    assert detail["status"] == "processed"
    assert detail["pages"] == [{"page_number": 1, "text": FAKE_OCR_TEXT, "method": "ocr"}]
    assert fake_ocr.calls == 1


def test_document_list_shows_processing_status(client: TestClient) -> None:
    _upload_one(client, "invoice.pdf", make_text_pdf("Some text that is long enough to keep."))

    [document] = client.get("/documents", headers={"X-Session-ID": "owner"}).json()

    assert document["status"] == "processed"
    assert document["page_count"] == 1


def test_scan_fails_clearly_when_ocr_is_unavailable(client: TestClient) -> None:
    app.dependency_overrides[get_ocr_engine] = lambda: UnavailableOcrEngine("OCR is disabled")
    document_id = _upload_one(client, "scan.pdf", make_image_pdf())

    detail = _detail(client, document_id)

    assert detail["status"] == "failed"
    assert detail["error"] == "OCR is not available, so scanned pages cannot be read."
    assert detail["page_count"] is None
    assert detail["pages"] == []


def test_unreadable_image_fails(client: TestClient) -> None:
    document_id = _upload_one(client, "broken.png", PNG)  # valid signature, no image data

    detail = _detail(client, document_id)

    assert detail["status"] == "failed"
    assert detail["error"] == "The image file could not be read."


def test_unexpected_error_gives_generic_message(
    client: TestClient, fake_ocr: FakeOcrEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(image: object) -> str:
        raise RuntimeError("/secret/path/model.bin is corrupt")

    monkeypatch.setattr(fake_ocr, "read", explode)
    document_id = _upload_one(client, "photo.png", make_png())

    detail = _detail(client, document_id)

    assert detail["status"] == "failed"
    assert detail["error"] == "Text extraction failed."  # never the raw exception text


def test_one_failed_document_does_not_affect_others(client: TestClient) -> None:
    body = _upload(
        client,
        _file("broken.png", PNG),
        _file("good.pdf", make_text_pdf("A perfectly normal digital document.")),
        session_id="owner",
    )
    documents = body["documents"]
    assert isinstance(documents, list)

    statuses = [_detail(client, d["id"])["status"] for d in documents]

    assert statuses == ["failed", "processed"]


def test_other_sessions_document_is_not_found(client: TestClient) -> None:
    document_id = _upload_one(client, "invoice.pdf", make_text_pdf("Private text"))

    response = client.get(f"/documents/{document_id}", headers={"X-Session-ID": "someone-else"})

    assert response.status_code == 404
    assert response.json() == {"detail": "Document not found."}


def test_missing_document_is_not_found(client: TestClient) -> None:
    response = client.get("/documents/does-not-exist", headers={"X-Session-ID": "owner"})

    assert response.status_code == 404


def test_document_detail_requires_session_header(client: TestClient) -> None:
    assert client.get("/documents/anything").status_code == 422


def test_processing_a_deleted_document_is_skipped(client: TestClient) -> None:
    settings = get_settings()

    # Must not raise, e.g. when the document was removed before the task ran.
    process_document("no-such-id", app.state.engine, FakeOcrEngine(), settings)


def test_broken_pdf_is_marked_failed(client: TestClient) -> None:
    broken = b"%PDF-1.7\n" + b"x" * 100  # passes the magic-bytes check, but is not a PDF
    document_id = _upload_one(client, "broken.pdf", broken)

    detail = _detail(client, document_id)

    assert detail["status"] == "failed"
    assert detail["error"] == "The PDF file could not be read."
    assert detail["pages"] == []
