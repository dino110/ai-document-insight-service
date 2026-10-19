"""API tests for uploading and listing documents."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from docinsight.config import get_settings

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

    assert set(documents[0]) == {"id", "filename", "content_type", "size_bytes", "created_at"}


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
