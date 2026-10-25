"""Real OCR on the generated scans. Needs easyocr, so it runs only in Docker:

    docker compose run --rm test

Locally these tests are skipped.
"""

import json
from decimal import Decimal
from pathlib import Path

import pytest

from docinsight.config import Settings
from docinsight.extraction.ocr import EasyOcrEngine
from docinsight.extraction.service import extract_document

pytestmark = pytest.mark.slow

SAMPLE_DOCS = Path(__file__).parent.parent / "sample_docs"


@pytest.fixture(scope="module")
def easyocr_engine() -> EasyOcrEngine:
    pytest.importorskip("easyocr")
    return EasyOcrEngine(["hr", "en"])  # loads the models once for the whole module


def _croatian_amount(value: str) -> str:
    """'1575.00' -> '1.575,00', as printed on the invoice."""
    return f"{Decimal(value):,.2f}".replace(",", " ").replace(".", ",").replace(" ", ".")


@pytest.mark.parametrize(
    ("filename", "content_type"),
    [
        ("racun-2026-001-scan.jpg", "image/jpeg"),
        ("racun-2026-001-scan.pdf", "application/pdf"),
    ],
)
def test_scanned_invoice_is_readable(
    easyocr_engine: EasyOcrEngine, filename: str, content_type: str
) -> None:
    expected = json.loads((SAMPLE_DOCS / "expected.json").read_text(encoding="utf-8"))[filename]
    content = (SAMPLE_DOCS / filename).read_bytes()

    [page] = extract_document(content, content_type, easyocr_engine, Settings())

    assert page.method == "ocr"
    # Numbers are what later phases depend on; diacritics are known to get lost (Č -> C).
    assert expected["number"] in page.text
    assert expected["issuer_oib"] in page.text
    assert _croatian_amount(expected["total"]) in page.text
