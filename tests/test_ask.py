"""API tests for POST /ask, with a fake LLM and the local HashingEmbedder."""

import pytest
from factories import FakeGenerator, make_text_pdf
from fastapi.testclient import TestClient
from httpx2 import Response

from docinsight.api.deps import get_generator
from docinsight.main import app
from docinsight.qa.generator import LlmAnswer, LlmRequestError, UnavailableGenerator
from docinsight.qa.prompts import ContextChunk

INVOICE_TEXT = "Racun 2026-001. Kupac: Horvat d.o.o. Ukupno za platiti: 1.575,00 EUR"


def _upload(client: TestClient, name: str, text: str, session_id: str = "owner") -> str:
    response = client.post(
        "/upload",
        files=[("files", (name, make_text_pdf(text, pages=2), "application/pdf"))],
        headers={"X-Session-ID": session_id},
    )
    assert response.status_code == 201, response.text
    document_id: str = response.json()["documents"][0]["id"]
    return document_id


def _ask(client: TestClient, body: dict[str, object], session_id: str = "owner") -> Response:
    return client.post("/ask", json=body, headers={"X-Session-ID": session_id})


def test_answer_cites_file_and_page(client: TestClient, generator: FakeGenerator) -> None:
    document_id = _upload(client, "racun-2026-001.pdf", INVOICE_TEXT)
    generator.answer = LlmAnswer(answer="1.575,00 EUR [1]", found=True, citations=[1])

    response = _ask(client, {"question": "  Koliki je ukupni iznos?  "})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answer"] == "1.575,00 EUR [1]"
    assert body["found"] is True
    assert body["model"] == "fake-llm"
    assert set(body["timings_ms"]) == {"retrieval", "generation"}
    [source] = body["sources"]
    assert source["document_id"] == document_id
    assert source["filename"] == "racun-2026-001.pdf"
    assert source["page_number"] in (1, 2)
    assert "1.575,00 EUR" in source["snippet"]
    assert set(source) == {"document_id", "filename", "page_number", "snippet", "score"}
    # The question reaches the LLM stripped, the excerpts with the start of their document.
    [(question, contexts)] = generator.calls
    assert question == "Koliki je ukupni iznos?"
    assert contexts[0].intro.startswith("Racun 2026-001. Kupac: Horvat d.o.o.")


def test_not_found_answer_has_no_sources(client: TestClient, generator: FakeGenerator) -> None:
    _upload(client, "racun.pdf", INVOICE_TEXT)
    generator.answer = LlmAnswer(answer="Nema podatka.", found=False, citations=[])

    body = _ask(client, {"question": "Koja je kamatna stopa?"}).json()

    assert body["found"] is False
    assert body["sources"] == []


def test_long_snippet_is_cut_at_a_word(client: TestClient, generator: FakeGenerator) -> None:
    _upload(
        client, "dugi.pdf", "rijec " * 120
    )  # one chunk of about 720 characters (ASCII: core PDF font)

    [source] = _ask(client, {"question": "rijec"}).json()["sources"]

    assert len(source["snippet"]) <= 303
    assert source["snippet"].endswith("rijec...")


def test_no_documents_is_409(client: TestClient, generator: FakeGenerator) -> None:
    response = _ask(client, {"question": "Koliki je iznos?"})

    assert response.status_code == 409
    assert response.json() == {"detail": "No processed documents to search."}
    assert generator.calls == []


def test_sessions_do_not_see_each_others_documents(
    client: TestClient, generator: FakeGenerator
) -> None:
    _upload(client, "tajno.pdf", INVOICE_TEXT, session_id="session-a")

    response = _ask(client, {"question": "Koliki je ukupni iznos?"}, session_id="session-b")

    # Session B has no documents of its own, so A's documents are never searched.
    assert response.status_code == 409
    assert generator.calls == []


def test_document_ids_limit_the_search(client: TestClient, generator: FakeGenerator) -> None:
    _upload(client, "a.pdf", INVOICE_TEXT)
    other = _upload(client, "b.pdf", INVOICE_TEXT)

    _ask(client, {"question": "Koliki je iznos?", "document_ids": [other], "top_k": 20})

    [(_, contexts)] = generator.calls
    assert {c.filename for c in contexts} == {"b.pdf"}


@pytest.mark.parametrize(
    "body",
    [
        {"question": ""},
        {"question": "   "},
        {"question": "x" * 1001},
        {"question": "Pitanje?", "top_k": 0},
        {"question": "Pitanje?", "top_k": 21},
        {"question": "Pitanje?", "document_ids": []},
        {},
    ],
)
def test_invalid_request_is_422(client: TestClient, body: dict[str, object]) -> None:
    assert _ask(client, body).status_code == 422


def test_session_header_is_required(client: TestClient) -> None:
    assert client.post("/ask", json={"question": "Pitanje?"}).status_code == 422


def test_unconfigured_llm_is_503(client: TestClient) -> None:
    _upload(client, "racun.pdf", INVOICE_TEXT)
    app.dependency_overrides[get_generator] = lambda: UnavailableGenerator("gpt-6-luna")

    response = _ask(client, {"question": "Koliki je iznos?"})

    assert response.status_code == 503
    assert response.json() == {"detail": "The answer service is not configured."}


def test_failed_llm_call_is_502(
    client: TestClient, generator: FakeGenerator, monkeypatch: pytest.MonkeyPatch
) -> None:
    _upload(client, "racun.pdf", INVOICE_TEXT)

    def explode(question: str, contexts: list[ContextChunk]) -> LlmAnswer:
        raise LlmRequestError("upstream timeout at https://internal.example")

    monkeypatch.setattr(generator, "generate", explode)

    response = _ask(client, {"question": "Koliki je iznos?"})

    assert response.status_code == 502
    assert response.json() == {"detail": "The answer service failed, try again."}
