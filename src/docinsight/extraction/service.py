"""Turn an uploaded file into text per page, using OCR only where the text layer is missing."""

from dataclasses import dataclass
from typing import Literal

from docinsight.config import Settings
from docinsight.extraction.ocr import OcrEngine, image_to_array
from docinsight.extraction.pdf import extract_text_layer, render_page

ExtractionMethod = Literal["text_layer", "ocr"]


@dataclass(frozen=True)
class PageText:
    page: int  # 1-based
    text: str
    method: ExtractionMethod


def needs_ocr(text: str, min_chars: int) -> bool:
    """True when a text layer has fewer than ``min_chars`` non-whitespace characters."""
    return len("".join(text.split())) < min_chars


def extract_document(
    content: bytes, content_type: str, ocr: OcrEngine, settings: Settings
) -> list[PageText]:
    """Extract text from a PDF (text layer, OCR fallback per page) or an image (OCR).

    Errors from PyMuPDF, Pillow and the OCR engine are not caught here.
    """
    if content_type in ("image/png", "image/jpeg"):
        return [PageText(page=1, text=ocr.read(image_to_array(content)).strip(), method="ocr")]
    if content_type != "application/pdf":
        raise ValueError(f"Unsupported content type: {content_type}")

    pages: list[PageText] = []
    for index, text in enumerate(extract_text_layer(content)):
        if needs_ocr(text, settings.ocr_min_chars):
            image = render_page(content, index, settings.ocr_dpi)
            pages.append(PageText(page=index + 1, text=ocr.read(image).strip(), method="ocr"))
        else:
            pages.append(PageText(page=index + 1, text=text.strip(), method="text_layer"))
    return pages
