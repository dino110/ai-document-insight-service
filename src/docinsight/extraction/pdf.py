"""PDF helpers built on PyMuPDF: read the text layer and render pages for OCR."""

import numpy as np
import pymupdf
from numpy.typing import NDArray


def extract_text_layer(pdf_bytes: bytes) -> list[str]:
    """Return the embedded text of every page, in page order.

    A scanned page has no text layer, so its entry is an empty (or near-empty) string.
    Raises PyMuPDF's exceptions for files it cannot open; the caller decides what to do.
    """
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        return [str(page.get_text()) for page in doc]


def render_page(pdf_bytes: bytes, index: int, dpi: int) -> NDArray[np.uint8]:
    """Render one page (0-based index) to an RGB image of shape (height, width, 3)."""
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        pix = doc[index].get_pixmap(dpi=dpi)  # RGB, no alpha by default
    # pix.samples is a flat byte string: row by row, pix.n bytes per pixel.
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
