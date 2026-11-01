"""Tests for the document field evaluation script."""

from collections.abc import Sequence

import pytest
from evaluate_fields import FIELDS, Kind, documents, expected_values, field_matches, run
from fastapi.testclient import TestClient

from docinsight.api.deps import get_field_extractor
from docinsight.main import app
from docinsight.nlp.fields import ContractData, DocumentExtraction, InvoiceData


@pytest.mark.parametrize(
    ("name", "got", "expected", "ok"),
    [
        ("total", "1575", "1575.00", True),
        ("total", "1575.01", "1575.00", False),
        ("total", "1.575,00", "1575.00", False),  # not the requested format
        ("fee", "6300", "6300.00", True),
        ("due_date", "2026-03-29", "2026-03-29", True),
        ("end_date", "29.10.2026.", "2026-10-29", False),
        ("issuer_iban", "HR72 2360 0004 2751 7979 0", "HR7223600004275179790", True),
        ("buyer_oib", "99083863799", "99083863799", True),
        ("issuer", " Car d.o.o. ", "Car d.o.o.", True),
        ("issuer", "Car d.o.o", "Car d.o.o.", False),  # names must be exact
        ("number", None, "2026-001", False),
    ],
)
def test_field_matches(name: str, got: str | None, expected: str, ok: bool) -> None:
    assert field_matches(name, got, expected) is ok


def test_documents_are_the_sample_documents() -> None:
    assert list(documents(include_scans=False)) == [
        "racun-2026-001.pdf",
        "racun-2026-002.pdf",
        "racun-2026-003.pdf",
        "ugovor-01.pdf",
        "ugovor-02.pdf",
        "racun-2026-004.pdf",
        "racun-2026-005.pdf",
        "racun-2026-006.pdf",
        "racun-2026-007.pdf",
        "ugovor-03.pdf",
        "ugovor-04.pdf",
    ]
    assert len(documents(include_scans=True)) == 15  # plus two scans of each of two documents


def test_contract_parties_become_client_and_contractor() -> None:
    expected = documents(include_scans=False)["ugovor-01.pdf"]

    values = expected_values("contract", expected)

    assert isinstance(expected["parties"], list)
    assert (values["client"], values["contractor"]) == tuple(expected["parties"])
    assert set(FIELDS["contract"]) <= set(values)


class ExpectedExtractor:
    """A perfect LLM: recognises the sample document and returns expected.json."""

    model = "fake-llm"

    def extract(self, pages: Sequence[tuple[int, str]]) -> DocumentExtraction:
        kind, values = self._sample(pages[0][1])
        if kind == "invoice":
            invoice = InvoiceData(currency="EUR", **{f: values[f] for f in FIELDS["invoice"]})
            return DocumentExtraction(
                document_type="invoice", summary="Račun.", invoice=invoice, contract=None
            )
        contract = ContractData(
            client_oib=None,
            contractor_oib=None,
            signed_date=None,
            currency="EUR",
            **{f: values[f] for f in FIELDS["contract"]},
        )
        return DocumentExtraction(
            document_type="contract", summary="Ugovor.", invoice=None, contract=contract
        )

    @staticmethod
    def _sample(text: str) -> tuple[Kind, dict[str, str]]:
        """The invoice number or the client's name tells the sample documents apart."""
        for name, expected in documents(include_scans=False).items():
            kind: Kind = "invoice" if name.startswith("racun-") else "contract"
            values = expected_values(kind, expected)
            if values["number" if kind == "invoice" else "client"] in text:
                return kind, values
        raise ValueError("not a sample document")


def test_run_against_the_app_produces_a_report(client: TestClient) -> None:
    app.dependency_overrides[get_field_extractor] = ExpectedExtractor

    text, results = run(client, label="test run", include_scans=False, timeout_s=30)

    assert [r.status_code for r in results] == [200] * 11
    assert all(r.type_ok and all(r.fields_ok.values()) for r in results)
    assert all(r.summary_chars > 0 for r in results)
    assert all(all(r.ner_found.values()) for r in results)
    assert "### Field extraction: test run" in text
    assert "| **all fields** | 97/97 (100%) |" in text
    assert "| Identifiers and amounts among the entities (local NER) | 32/32 (100%) |" in text
    # Rules only in tests: the contractors (people) are not among the entities.
    contracts = [r for r in results if r.kind == "contract"]
    assert all(r.failed_checks == ["names_in_entities"] for r in contracts)


def test_report_shows_http_errors(client: TestClient) -> None:
    # Without an override the extractor is unavailable (no API key in tests): 503.
    text, results = run(client, label="no key", include_scans=False, timeout_s=30)

    assert {r.status_code for r in results} == {503}
    assert "| racun-2026-001.pdf | HTTP 503 | - | - |" in text
