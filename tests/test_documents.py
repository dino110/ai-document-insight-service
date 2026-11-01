"""API tests for uploading, listing and reading documents."""

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
from factories import FAKE_OCR_TEXT, FakeOcrEngine, make_image_pdf, make_png, make_text_pdf
from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlmodel import Session, col, select

from docinsight.api.deps import get_entity_extractor, get_ocr_engine
from docinsight.config import get_settings
from docinsight.extraction.ocr import UnavailableOcrEngine
from docinsight.indexing.chunking import index_name
from docinsight.indexing.embeddings import HashingEmbedder
from docinsight.main import app
from docinsight.nlp.ner import Entity, RuleEntityExtractor
from docinsight.processing import process_document
from docinsight.qa.guard import Language
from docinsight.schemas import DocumentDetail
from docinsight.storage.models import DocumentChunk, DocumentEntity, DocumentPage

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
    assert set(pages[0]) == {"page_number", "text", "method", "entities"}
    assert fake_ocr.calls == 0


def test_image_is_processed_with_ocr(client: TestClient, fake_ocr: FakeOcrEngine) -> None:
    document_id = _upload_one(client, "photo.png", make_png())

    detail = DocumentDetail.model_validate(_detail(client, document_id))

    assert detail.status == "processed"
    [page] = detail.pages
    assert (page.page_number, page.text, page.method) == (1, FAKE_OCR_TEXT, "ocr")
    # Entities are found in OCR text too, with offsets into it.
    assert [(e.label, e.normalized) for e in page.entities] == [("MONEY", "1575.00")]
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
    process_document(
        "no-such-id",
        app.state.engine,
        FakeOcrEngine(),
        HashingEmbedder(),
        RuleEntityExtractor(),
        settings,
    )


def test_broken_pdf_is_marked_failed(client: TestClient) -> None:
    broken = b"%PDF-1.7\n" + b"x" * 100  # passes the magic-bytes check, but is not a PDF
    document_id = _upload_one(client, "broken.pdf", broken)

    detail = _detail(client, document_id)

    assert detail["status"] == "failed"
    assert detail["error"] == "The PDF file could not be read."
    assert detail["pages"] == []


# --- Indexing ---


def _chunks(document_id: str) -> list[DocumentChunk]:
    with Session(app.state.engine) as session:
        statement = select(DocumentChunk).where(DocumentChunk.document_id == document_id)
        return list(session.exec(statement.order_by(col(DocumentChunk.chunk_index))))


def test_processed_document_is_indexed_per_page(
    client: TestClient, embedder: HashingEmbedder
) -> None:
    content = make_text_pdf("Invoice 2026-001. Total amount due: 1.575,00 EUR", pages=2)
    document_id = _upload_one(client, "invoice.pdf", content)

    chunks = _chunks(document_id)

    assert _detail(client, document_id)["status"] == "processed"
    assert [chunk.page_number for chunk in chunks] == [1, 2]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1]
    assert {chunk.owner_id for chunk in chunks} == {"owner"}
    assert {chunk.embedding_model for chunk in chunks} == {index_name(embedder.name)}
    assert "1.575,00 EUR" in chunks[0].text
    assert len(chunks[0].embedding) == embedder.dim * 4  # float32


def test_document_without_text_is_processed_with_no_chunks(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_ocr: FakeOcrEngine
) -> None:
    monkeypatch.setattr(fake_ocr, "text", "   ")  # a blank scan
    document_id = _upload_one(client, "blank.png", make_png())

    assert _detail(client, document_id)["status"] == "processed"
    assert _chunks(document_id) == []


def test_embedding_error_marks_document_failed(
    client: TestClient, embedder: HashingEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(texts: list[str]) -> object:
        raise RuntimeError("upstream said: sk-secret is invalid")

    monkeypatch.setattr(embedder, "embed_documents", explode)
    document_id = _upload_one(client, "invoice.pdf", make_text_pdf("Some text worth indexing."))

    detail = _detail(client, document_id)

    assert detail["status"] == "failed"
    assert detail["error"] == "Indexing failed."  # never the raw exception text
    # Rolled back together: no pages and no chunks of a half-processed document.
    assert detail["pages"] == []
    assert _chunks(document_id) == []


# --- Entities found while processing --------------------------------------------

SAMPLE_DOCS = Path(__file__).parent.parent / "sample_docs"


def _entities(document_id: str) -> list[DocumentEntity]:
    with Session(app.state.engine) as session:
        return list(
            session.exec(
                select(DocumentEntity)
                .where(DocumentEntity.document_id == document_id)
                .order_by(col(DocumentEntity.page_number), col(DocumentEntity.start))
            ).all()
        )


def test_processing_stores_the_entities_of_an_invoice(client: TestClient) -> None:
    content = (SAMPLE_DOCS / "racun-2026-001.pdf").read_bytes()
    expected = json.loads((SAMPLE_DOCS / "expected.json").read_text())["racun-2026-001.pdf"]

    document_id = _upload_one(client, "racun-2026-001.pdf", content)

    entities = _entities(document_id)
    found = {(e.label, e.normalized) for e in entities}
    assert ("OIB", expected["issuer_oib"]) in found
    assert ("OIB", expected["buyer_oib"]) in found
    assert ("IBAN", expected["issuer_iban"]) in found  # printed in groups and compact
    assert ("MONEY", expected["total"]) in found
    assert ("DATE", expected["due_date"]) in found
    assert ("ORG", expected["issuer"]) in found
    assert ("ORG", expected["buyer"]) in found
    # Both printed forms of the IBAN normalise to the same value.
    assert sorted(e.text for e in entities if e.label == "IBAN") == sorted(
        [" ".join(expected["issuer_iban"][i : i + 4] for i in range(0, 21, 4)),
         expected["issuer_iban"]]
    )  # fmt: skip
    assert {e.extractor for e in entities} == {"rules-3"}  # the tests use rules only
    assert {e.owner_id for e in entities} == {"owner"}


def test_entity_offsets_point_into_the_page_text(client: TestClient) -> None:
    content = make_text_pdf("Kupac Horvat d.o.o., OIB 99083863799, ukupno 1.575,00 EUR", pages=2)
    document_id = _upload_one(client, "racun.pdf", content)

    entities = _entities(document_id)
    with Session(app.state.engine) as session:
        pages = {
            page.page_number: page.text
            for page in session.exec(
                select(DocumentPage).where(DocumentPage.document_id == document_id)
            )
        }

    assert {e.page_number for e in entities} == {1, 2}  # make_text_pdf repeats the text
    assert all(pages[e.page_number][e.start : e.end] == e.text for e in entities)


class FailingExtractor:
    name = "failing"

    def extract(self, text: str, language: Language) -> list[Entity]:
        raise RuntimeError("model crashed")


def test_ner_failure_does_not_fail_processing(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    app.dependency_overrides[get_entity_extractor] = FailingExtractor
    content = make_text_pdf("Kupac Horvat d.o.o., OIB 99083863799", pages=1)

    with caplog.at_level(logging.WARNING, logger="docinsight.processing"):
        document_id = _upload_one(client, "racun.pdf", content)

    assert _detail(client, document_id)["status"] == "processed"
    with Session(app.state.engine) as session:
        chunks = session.exec(
            select(DocumentChunk).where(DocumentChunk.document_id == document_id)
        ).all()
    assert chunks  # still searchable
    assert _entities(document_id) == []
    assert "entity extraction failed" in caplog.text
    assert "99083863799" not in caplog.text  # never the document text


# --- Entities in the document detail --------------------------------------------


def test_detail_has_entities_per_page_and_a_summary(client: TestClient) -> None:
    content = (SAMPLE_DOCS / "racun-2026-001.pdf").read_bytes()
    expected = json.loads((SAMPLE_DOCS / "expected.json").read_text())["racun-2026-001.pdf"]
    document_id = _upload_one(client, "racun-2026-001.pdf", content)

    body = _detail(client, document_id)
    detail = DocumentDetail.model_validate(body)  # typed access, and the shape is checked

    [page] = detail.pages
    assert all(page.text[e.start : e.end] == e.text for e in page.entities)
    pages = body["pages"]
    assert isinstance(pages, list)
    assert set(pages[0]["entities"][0]) == {"text", "label", "normalized", "start", "end"}
    summary = detail.entity_summary
    assert summary["OIB"] == [expected["issuer_oib"], expected["buyer_oib"]]  # page order
    assert summary["IBAN"] == [expected["issuer_iban"]]  # printed twice, listed once
    assert summary["ORG"] == [expected["issuer"], expected["buyer"]]
    assert expected["total"] in summary["MONEY"]
    # Labels come in a fixed order, not in order of appearance.
    assert list(summary) == [
        label
        for label in ("PERSON", "ORG", "LOC", "MISC", "OIB", "IBAN", "DATE", "MONEY")
        if label in summary
    ]


def test_summary_lists_a_value_once_across_pages(client: TestClient) -> None:
    content = make_text_pdf("Kupac Horvat d.o.o., OIB 99083863799", pages=3)
    document_id = _upload_one(client, "racun.pdf", content)

    detail = DocumentDetail.model_validate(_detail(client, document_id))

    assert [len(page.entities) for page in detail.pages] == [2, 2, 2]
    assert detail.entity_summary == {"ORG": ["Horvat d.o.o."], "OIB": ["99083863799"]}


def test_summary_keeps_the_first_ten_values_per_label(client: TestClient) -> None:
    amounts = ", ".join(f"{n},00 EUR" for n in range(1, 13))
    document_id = _upload_one(client, "racun.pdf", make_text_pdf(f"Stavke: {amounts}"))

    detail = DocumentDetail.model_validate(_detail(client, document_id))

    assert len(detail.pages[0].entities) == 12
    assert detail.entity_summary["MONEY"] == [f"{n}.00" for n in range(1, 11)]


def test_document_without_entities_has_empty_lists(client: TestClient) -> None:
    # Long enough for the text layer (a short page would go to the fake OCR instead).
    text = "Ovdje nema nikakvih podataka koje bi pravila ili model prepoznali."
    document_id = _upload_one(client, "tekst.pdf", make_text_pdf(text))

    detail = DocumentDetail.model_validate(_detail(client, document_id))

    assert detail.pages[0].entities == []
    assert detail.entity_summary == {}


@pytest.fixture
def select_count(client: TestClient) -> Iterator[list[str]]:
    """The SELECT statements the app's engine runs, to spot one query per page (N+1)."""
    statements: list[str] = []

    def record(conn: object, cursor: object, statement: str, *args: object) -> None:
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append(statement)

    event.listen(app.state.engine, "before_cursor_execute", record)
    yield statements
    event.remove(app.state.engine, "before_cursor_execute", record)


def test_detail_query_count_does_not_grow_with_pages(
    client: TestClient, select_count: list[str]
) -> None:
    text = "Kupac Horvat d.o.o., OIB 99083863799"
    one = _upload_one(client, "one.pdf", make_text_pdf(text, pages=1))
    five = _upload_one(client, "five.pdf", make_text_pdf(text, pages=5))

    select_count.clear()
    _detail(client, one)
    for_one_page = len(select_count)
    select_count.clear()
    _detail(client, five)

    assert len(select_count) == for_one_page == 3  # document, pages, entities


def test_summary_labels_have_a_fixed_order_not_the_order_on_the_page(
    client: TestClient,
) -> None:
    text = "Ukupno 1.575,00 EUR, rok placanja 29.03.2026., OIB 99083863799"  # Latin-1 font
    document_id = _upload_one(client, "racun.pdf", make_text_pdf(text))

    detail = DocumentDetail.model_validate(_detail(client, document_id))

    assert list(detail.entity_summary) == ["OIB", "DATE", "MONEY"]  # page: MONEY, DATE, OIB
