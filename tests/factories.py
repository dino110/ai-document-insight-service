"""Test inputs built in memory (PDFs, images) and a fake OCR engine."""

from io import BytesIO

import numpy as np
import pymupdf
from fpdf import FPDF
from numpy.typing import NDArray
from PIL import Image, ImageDraw

FAKE_OCR_TEXT = "Scanned invoice 2026-001, total 1.575,00 EUR"


def make_text_pdf(text: str, pages: int = 1) -> bytes:
    """A digital PDF with ``text`` on every page. Core fonts only cover Latin-1."""
    pdf = FPDF(format="A4")
    pdf.set_font("Helvetica", size=12)
    for _ in range(pages):
        pdf.add_page()
        if text:
            pdf.multi_cell(0, 6, text)
    return bytes(pdf.output())


def make_png(width: int = 200, height: int = 100) -> bytes:
    """A small but valid PNG: white background with a black bar."""
    img = Image.new("RGB", (width, height), "white")
    ImageDraw.Draw(img).rectangle((10, 40, width - 10, 60), fill="black")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_image_pdf(pages: int = 1) -> bytes:
    """A 'scanned' PDF: every page is just an image, there is no text layer."""
    png = make_png()
    with pymupdf.open() as doc:
        for _ in range(pages):
            page = doc.new_page()
            page.insert_image(page.rect, stream=png)
        return bytes(doc.tobytes())  # PyMuPDF is untyped here (Any)


class FakeOcrEngine:
    """Satisfies the OcrEngine protocol: returns fixed text and counts calls."""

    def __init__(self, text: str = "") -> None:
        self.text = text
        self.calls = 0
        self.shapes: list[tuple[int, ...]] = []

    def read(self, image: NDArray[np.uint8]) -> str:
        self.calls += 1
        self.shapes.append(image.shape)
        return self.text
