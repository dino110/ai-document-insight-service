"""Background processing of uploaded documents: extract text and store it per page."""

import logging
import time
from pathlib import Path

import pymupdf
from PIL import UnidentifiedImageError
from sqlalchemy import Engine
from sqlmodel import Session

from docinsight.config import Settings
from docinsight.extraction.ocr import OcrEngine, OcrUnavailableError
from docinsight.extraction.service import extract_document
from docinsight.storage.models import Document, DocumentPage

logger = logging.getLogger(__name__)


def _error_message(exc: Exception) -> str:
    """Short, user-facing reason. Never the raw exception text, which may contain paths."""
    if isinstance(exc, OcrUnavailableError):
        return "OCR is not available, so scanned pages cannot be read."
    if isinstance(exc, pymupdf.FileDataError):
        return "The PDF file could not be read."
    if isinstance(exc, UnidentifiedImageError):
        return "The image file could not be read."
    return "Text extraction failed."


def process_document(document_id: str, engine: Engine, ocr: OcrEngine, settings: Settings) -> None:
    """Extract the text of one document and mark it as processed or failed.

    Runs as a background task after the upload response is sent, so it opens its own
    database session: the request's session is already closed by then.
    """
    started = time.perf_counter()
    with Session(engine) as session:
        document = session.get(Document, document_id)
        if document is None:
            logger.warning("Document %s not found, skipping processing", document_id)
            return

        try:
            content = Path(document.stored_path).read_bytes()
            pages = extract_document(content, document.content_type, ocr, settings)
            session.add_all(
                DocumentPage(
                    document_id=document_id,
                    page_number=page.page,
                    text=page.text,
                    method=page.method,
                )
                for page in pages
            )
            document.status = "processed"
            document.page_count = len(pages)
            session.commit()
        except Exception as exc:
            # One broken document must not affect others: record the failure and move on.
            logger.exception("Processing failed for document %s", document_id)
            session.rollback()
            document.status = "failed"
            document.error = _error_message(exc)
            session.commit()

        logger.info(
            "Document %s %s in %.2f s (pages=%s)",
            document_id,
            document.status,
            time.perf_counter() - started,
            document.page_count,
        )
