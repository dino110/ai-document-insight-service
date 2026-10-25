"""Unit tests for text extraction: no API, no database, fake OCR."""

import sys
from types import ModuleType

import numpy as np
import pymupdf
import pytest
from factories import FakeOcrEngine, make_image_pdf, make_png, make_text_pdf
from numpy.typing import NDArray

from docinsight.config import Settings
from docinsight.extraction.ocr import (
    EasyOcrEngine,
    OcrUnavailableError,
    UnavailableOcrEngine,
    create_ocr_engine,
    image_to_array,
)
from docinsight.extraction.pdf import extract_text_layer, render_page
from docinsight.extraction.service import extract_document


@pytest.fixture
def settings() -> Settings:
    # Low DPI keeps rendering fast; the threshold is the documented default.
    return Settings(ocr_min_chars=30, ocr_dpi=72)


def test_digital_pdf_uses_text_layer(settings: Settings) -> None:
    ocr = FakeOcrEngine("should not be used")
    content = make_text_pdf("Invoice 2026-001. Total amount due: 1.575,00 EUR")

    [page] = extract_document(content, "application/pdf", ocr, settings)

    assert page.page == 1
    assert page.method == "text_layer"
    assert "1.575,00 EUR" in page.text
    assert ocr.calls == 0


def test_image_only_pdf_goes_through_ocr(settings: Settings) -> None:
    ocr = FakeOcrEngine("text from the scan")

    [page] = extract_document(make_image_pdf(), "application/pdf", ocr, settings)

    assert page.method == "ocr"
    assert page.text == "text from the scan"
    assert ocr.calls == 1
    height, width, channels = ocr.shapes[0]
    assert channels == 3
    assert height > width  # A4 portrait


def test_ocr_runs_only_for_pages_without_text(settings: Settings) -> None:
    text_pdf = make_text_pdf("A page with plenty of real text on it.")
    with pymupdf.open(stream=text_pdf, filetype="pdf") as doc:
        with pymupdf.open(stream=make_image_pdf(), filetype="pdf") as scan:
            doc.insert_pdf(scan)
        mixed = doc.tobytes()
    ocr = FakeOcrEngine("scanned page")

    pages = extract_document(mixed, "application/pdf", ocr, settings)

    assert [(p.page, p.method) for p in pages] == [(1, "text_layer"), (2, "ocr")]
    assert ocr.calls == 1


def test_empty_ocr_result_is_an_empty_page_not_an_error(settings: Settings) -> None:
    [page] = extract_document(make_image_pdf(), "application/pdf", FakeOcrEngine(""), settings)

    assert page.method == "ocr"
    assert page.text == ""


def test_image_is_a_single_ocr_page(settings: Settings) -> None:
    ocr = FakeOcrEngine("from the photo")

    pages = extract_document(make_png(width=300, height=120), "image/png", ocr, settings)

    assert [(p.page, p.method, p.text) for p in pages] == [(1, "ocr", "from the photo")]
    assert ocr.shapes == [(120, 300, 3)]


@pytest.mark.parametrize(
    ("char_count", "expected_method"),
    [(0, "ocr"), (29, "ocr"), (30, "text_layer"), (31, "text_layer")],
)
def test_ocr_threshold(settings: Settings, char_count: int, expected_method: str) -> None:
    # settings.ocr_min_chars is 30: fewer non-whitespace characters means OCR.
    content = make_text_pdf("x" * char_count)

    [page] = extract_document(content, "application/pdf", FakeOcrEngine("ocr"), settings)

    assert page.method == expected_method


def test_unsupported_content_type_is_rejected(settings: Settings) -> None:
    with pytest.raises(ValueError, match="Unsupported content type"):
        extract_document(b"hello", "text/plain", FakeOcrEngine(), settings)


def test_text_layer_is_returned_per_page() -> None:
    pages = extract_text_layer(make_text_pdf("Same text", pages=3))

    assert len(pages) == 3
    assert all("Same text" in page for page in pages)


def test_render_page_returns_rgb_array() -> None:
    image = render_page(make_text_pdf("x"), 0, dpi=72)

    height, width, channels = image.shape
    assert image.dtype == np.uint8
    assert channels == 3
    # A4 is 595.28 x 841.89 points and 72 dpi is 1 px per point; PyMuPDF rounds up.
    assert (height, width) == pytest.approx((842, 595), abs=1)


def test_broken_pdf_raises() -> None:
    with pytest.raises(pymupdf.FileDataError):
        extract_text_layer(b"%PDF-1.7\n" + b"x" * 100)


def test_image_to_array_converts_to_rgb() -> None:
    assert image_to_array(make_png(width=40, height=20)).shape == (20, 40, 3)


# --- OCR engine selection -------------------------------------------------------


class _FakeReader:
    """Stands in for easyocr.Reader."""

    def __init__(self, languages: list[str], gpu: bool) -> None:
        self.languages = languages
        self.gpu = gpu

    def readtext(self, image: NDArray[np.uint8], detail: int, paragraph: bool) -> list[str]:
        return ["first paragraph", "second paragraph"]


@pytest.fixture
def fake_easyocr(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make `import easyocr` return a tiny fake module instead of the real package."""
    module = ModuleType("easyocr")
    module.Reader = _FakeReader  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "easyocr", module)


def test_disabled_ocr_gives_unavailable_engine() -> None:
    engine = create_ocr_engine(Settings(ocr_enabled=False))

    assert isinstance(engine, UnavailableOcrEngine)
    with pytest.raises(OcrUnavailableError, match="disabled"):
        engine.read(np.zeros((1, 1, 3), dtype=np.uint8))


def test_enabled_ocr_without_easyocr_gives_unavailable_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "easyocr", None)  # makes `import easyocr` fail

    engine = create_ocr_engine(Settings(ocr_enabled=True))

    assert isinstance(engine, UnavailableOcrEngine)
    assert "not installed" in engine.reason


@pytest.mark.usefixtures("fake_easyocr")
def test_enabled_ocr_with_easyocr_gives_easyocr_engine() -> None:
    engine = create_ocr_engine(Settings(ocr_enabled=True, ocr_languages=["hr"]))

    assert isinstance(engine, EasyOcrEngine)
    assert engine.read(np.zeros((10, 10, 3), dtype=np.uint8)) == "first paragraph\nsecond paragraph"
