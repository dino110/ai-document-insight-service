"""API tests for GET /documents/{id}/fields, with a fake field extractor."""

import json
import logging
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from factories import FakeFieldExtractor, make_text_pdf
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from docinsight.api.deps import get_field_extractor
from docinsight.main import app
from docinsight.nlp.fields import FIELDS_FORMAT, ContractData, DocumentExtraction, InvoiceData
from docinsight.qa.generator import LlmRequestError
from docinsight.schemas import ContractFields, FieldsOut
from docinsight.storage.models import Document, DocumentFields

SAMPLE_DOCS = Path(__file__).parent.parent / "sample_docs"
EXPECTED = json.loads((SAMPLE_DOCS / "expected.json").read_text(encoding="utf-8"))
INVOICE = EXPECTED["racun-2026-001.pdf"]


def correct_extraction(**changes: str | None) -> DocumentExtraction:
    """What a perfect LLM returns for racun-2026-001.pdf, with optional wrong fields."""
    values = {name: INVOICE[name] for name in InvoiceData.model_fields if name != "currency"}
    invoice = InvoiceData(currency="EUR", **{**values, **changes})
    return DocumentExtraction(
        document_type="invoice", summary="Račun 2026-001.", invoice=invoice, contract=None
    )


def contract_extraction() -> DocumentExtraction:
    client, contractor = EXPECTED["ugovor-01.pdf"]["parties"]
    contract = ContractData.model_validate(
        {**dict.fromkeys(ContractData.model_fields), "client": client, "contractor": contractor}
    )
    return DocumentExtraction(
        document_type="contract", summary="Ugovor o djelu.", invoice=None, contract=contract
    )


@pytest.fixture
def extractor(client: TestClient) -> Iterator[FakeFieldExtractor]:
    fake = FakeFieldExtractor(correct_extraction())
    app.dependency_overrides[get_field_extractor] = lambda: fake
    yield fake


def upload(client: TestClient, name: str, content: bytes, session_id: str = "owner") -> str:
    response = client.post(
        "/upload",
        files=[("files", (name, content, "application/pdf"))],
        headers={"X-Session-ID": session_id},
    )
    assert response.status_code == 201, response.text
    document_id: str = response.json()["documents"][0]["id"]
    return document_id


def fields(client: TestClient, document_id: str, session_id: str = "owner") -> FieldsOut:
    response = client.get(f"/documents/{document_id}/fields", headers={"X-Session-ID": session_id})
    assert response.status_code == 200, response.text
    return FieldsOut.model_validate(response.json())


def stored_rows() -> list[DocumentFields]:
    with Session(app.state.engine) as session:
        return list(session.exec(select(DocumentFields)).all())


def test_first_call_extracts_checks_and_stores(
    client: TestClient, extractor: FakeFieldExtractor
) -> None:
    content = (SAMPLE_DOCS / "racun-2026-001.pdf").read_bytes()
    document_id = upload(client, "racun-2026-001.pdf", content)

    result = fields(client, document_id)

    assert result.cached is False
    assert result.document_type == "invoice"
    assert result.summary == "Račun 2026-001."
    assert result.contract is None
    assert result.invoice is not None
    assert result.invoice.issuer_oib == INVOICE["issuer_oib"]
    assert result.invoice.total == INVOICE["total"]
    # Every check passes against the real page text and the entities from processing.
    assert result.checks == {
        "issuer_oib_valid": True,
        "buyer_oib_valid": True,
        "issuer_iban_valid": True,
        "amounts_consistent": True,
        "dates_consistent": True,
        "values_in_document": True,
        "names_in_entities": True,
    }
    assert result.warnings == []
    assert result.model == "fake-llm"
    [pages] = extractor.calls
    assert [n for n, _ in pages] == [1]
    assert "RAČUN br. 2026-001" in pages[0][1]
    assert [row.document_id for row in stored_rows()] == [document_id]


def test_second_call_is_read_from_the_database(
    client: TestClient, extractor: FakeFieldExtractor
) -> None:
    document_id = upload(client, "racun.pdf", (SAMPLE_DOCS / "racun-2026-001.pdf").read_bytes())

    first = fields(client, document_id)
    second = fields(client, document_id)

    assert (first.cached, second.cached) == (False, True)
    assert second.invoice == first.invoice and second.checks == first.checks
    assert second.summary == first.summary
    assert len(extractor.calls) == 1  # no second LLM call


def test_wrong_fields_come_back_with_warnings(
    client: TestClient, extractor: FakeFieldExtractor
) -> None:
    extractor.result = correct_extraction(number="2026-999")
    document_id = upload(client, "racun.pdf", (SAMPLE_DOCS / "racun-2026-001.pdf").read_bytes())

    result = fields(client, document_id)

    assert result.checks["values_in_document"] is False
    assert result.warnings == ["values not found in the document: number"]


def test_contract_has_contract_fields_and_checks(
    client: TestClient, extractor: FakeFieldExtractor
) -> None:
    extractor.result = contract_extraction()
    document_id = upload(client, "ugovor-01.pdf", (SAMPLE_DOCS / "ugovor-01.pdf").read_bytes())

    result = fields(client, document_id)

    client_name, contractor = EXPECTED["ugovor-01.pdf"]["parties"]
    assert result.document_type == "contract"
    assert result.summary == "Ugovor o djelu."
    assert result.invoice is None
    assert result.contract == ContractFields(client=client_name, contractor=contractor)
    assert set(result.checks) == {
        "client_oib_valid",
        "contractor_oib_valid",
        "dates_consistent",
        "values_in_document",
        "names_in_entities",
    }
    # The tests use rules only: the client (a company) is found, the contractor (a person)
    # needs the spaCy model, so the names check fails here.
    assert result.checks["names_in_entities"] is False


def test_other_document_has_a_summary_and_no_fields(
    client: TestClient, extractor: FakeFieldExtractor
) -> None:
    extractor.result = DocumentExtraction(
        document_type="other", summary="Dopis.", invoice=None, contract=None
    )
    document_id = upload(client, "dopis.pdf", make_text_pdf("Postovani, saljemo vam dopis."))

    result = fields(client, document_id)

    assert (result.document_type, result.summary) == ("other", "Dopis.")
    assert (result.invoice, result.contract, result.checks, result.warnings) == (None, None, {}, [])


def test_a_result_of_an_older_format_is_extracted_again(
    client: TestClient, extractor: FakeFieldExtractor
) -> None:
    document_id = upload(client, "racun.pdf", (SAMPLE_DOCS / "racun-2026-001.pdf").read_bytes())
    with Session(app.state.engine) as session:  # as stored before summaries existed
        session.add(
            DocumentFields(
                document_id=document_id,
                owner_id="owner",
                document_type="invoice",
                data={"number": "2026-001"},
                checks={},
                warnings=[],
                model="old",
            )
        )
        session.commit()

    result = fields(client, document_id)

    assert (result.cached, result.summary) == (False, "Račun 2026-001.")
    [row] = stored_rows()
    assert row.data["format"] == FIELDS_FORMAT
    assert fields(client, document_id).cached is True


def test_other_sessions_and_missing_documents_look_the_same(
    client: TestClient, extractor: FakeFieldExtractor
) -> None:
    document_id = upload(client, "racun.pdf", make_text_pdf("Racun 2026-001, ukupno 10,00 EUR"))

    other = client.get(f"/documents/{document_id}/fields", headers={"X-Session-ID": "other"})
    missing = client.get("/documents/nope/fields", headers={"X-Session-ID": "owner"})

    assert other.status_code == missing.status_code == 404
    assert other.json() == missing.json() == {"detail": "Document not found."}
    assert extractor.calls == []


def test_session_header_is_required(client: TestClient) -> None:
    assert client.get("/documents/anything/fields").status_code == 422


@pytest.mark.parametrize("document_status", ["pending", "failed"])
def test_document_that_is_not_processed_is_a_conflict(
    client: TestClient, extractor: FakeFieldExtractor, document_status: str
) -> None:
    with Session(app.state.engine) as session:
        document = Document(
            owner_id="owner",
            filename="racun.pdf",
            content_type="application/pdf",
            size_bytes=1,
            sha256="0" * 64,
            stored_path="x",
            status=document_status,
        )
        session.add(document)
        session.commit()
        document_id = document.id

    response = client.get(f"/documents/{document_id}/fields", headers={"X-Session-ID": "owner"})

    assert response.status_code == 409
    assert extractor.calls == []


def test_without_a_key_the_service_is_unavailable(client: TestClient) -> None:
    # No override: tests run without an API key, so lifespan created the unavailable one.
    document_id = upload(client, "racun.pdf", (SAMPLE_DOCS / "racun-2026-001.pdf").read_bytes())

    response = client.get(f"/documents/{document_id}/fields", headers={"X-Session-ID": "owner"})

    assert response.status_code == 503
    assert stored_rows() == []


def test_llm_failure_is_a_bad_gateway_and_stores_nothing(
    client: TestClient, extractor: FakeFieldExtractor
) -> None:
    extractor.result = LlmRequestError("timeout with secret details")
    document_id = upload(client, "racun.pdf", (SAMPLE_DOCS / "racun-2026-001.pdf").read_bytes())

    response = client.get(f"/documents/{document_id}/fields", headers={"X-Session-ID": "owner"})

    assert response.status_code == 502
    assert response.json() == {"detail": "The field extraction service failed, try again."}
    assert stored_rows() == []
    # The next call tries again instead of serving a stored failure.
    extractor.result = correct_extraction()
    assert fields(client, document_id).cached is False


def test_log_has_no_field_values(
    client: TestClient, extractor: FakeFieldExtractor, caplog: pytest.LogCaptureFixture
) -> None:
    extractor.result = correct_extraction(total="1234.56")
    document_id = upload(client, "racun.pdf", (SAMPLE_DOCS / "racun-2026-001.pdf").read_bytes())

    with caplog.at_level(logging.INFO, logger="docinsight.api.fields"):
        fields(client, document_id)

    assert "invoice, failed checks ['amounts_consistent', 'values_in_document']" in caplog.text
    for value in (INVOICE["issuer_oib"], INVOICE["buyer_oib"], INVOICE["issuer"], "1234.56"):
        assert value not in caplog.text


class RacingExtractor(FakeFieldExtractor):
    """Stores a row while "thinking", as a parallel first request would."""

    def extract(self, pages: Sequence[tuple[int, str]]) -> DocumentExtraction:
        with Session(app.state.engine) as session:
            session.add(
                DocumentFields(
                    document_id=self.document_id,
                    owner_id="owner",
                    document_type="invoice",
                    data={
                        "format": FIELDS_FORMAT,
                        "summary": "from the other request",
                        "invoice": None,
                        "contract": None,
                    },
                    checks={},
                    warnings=[],
                    model="fake-llm",
                )
            )
            session.commit()
        return super().extract(pages)

    document_id = ""


def test_parallel_first_calls_return_the_row_stored_first(client: TestClient) -> None:
    racing = RacingExtractor(correct_extraction())
    app.dependency_overrides[get_field_extractor] = lambda: racing
    racing.document_id = upload(
        client, "racun.pdf", (SAMPLE_DOCS / "racun-2026-001.pdf").read_bytes()
    )

    result = fields(client, racing.document_id)

    assert result.cached is True
    assert result.summary == "from the other request"
    assert len(stored_rows()) == 1
