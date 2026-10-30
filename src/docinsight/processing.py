"""Background processing of uploaded documents: extract text per page, then index it."""

import logging
import time
from pathlib import Path

import pymupdf
from PIL import UnidentifiedImageError
from sqlalchemy import Engine
from sqlmodel import Session

from docinsight.config import Settings
from docinsight.extraction.ocr import OcrEngine, OcrUnavailableError
from docinsight.extraction.service import PageText, extract_document
from docinsight.indexing.chunking import chunk_pages
from docinsight.indexing.embeddings import Embedder
from docinsight.storage.models import Document, DocumentChunk, DocumentPage

logger = logging.getLogger(__name__)


class IndexingError(RuntimeError):
    """Chunking or embedding failed after the text was extracted."""


def _error_message(exc: Exception) -> str:
    """Short, user-facing reason. Never the raw exception text, which may contain paths."""
    if isinstance(exc, IndexingError):
        return "Indexing failed."
    if isinstance(exc, OcrUnavailableError):
        return "OCR is not available, so scanned pages cannot be read."
    if isinstance(exc, pymupdf.FileDataError):
        return "The PDF file could not be read."
    if isinstance(exc, UnidentifiedImageError):
        return "The image file could not be read."
    return "Text extraction failed."


def _build_chunks(
    document: Document, pages: list[PageText], embedder: Embedder, settings: Settings
) -> list[DocumentChunk]:
    """Split the pages into chunks and embed them all in one call (batched by the embedder)."""
    try:
        chunks = chunk_pages(
            [(page.page, page.text) for page in pages], settings.chunk_size, settings.chunk_overlap
        )
        if not chunks:  # e.g. a blank scan: nothing to search, but not an error
            return []
        started = time.perf_counter()
        vectors = embedder.embed_documents([chunk.text for chunk in chunks])
    except Exception as exc:
        raise IndexingError from exc

    logger.info(
        "Document %s: embedded %d chunks with %s in %.2f s",
        document.id,
        len(chunks),
        embedder.name,
        time.perf_counter() - started,
    )
    return [
        DocumentChunk(
            document_id=document.id,
            owner_id=document.owner_id,
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            embedding=vector.tobytes(),
            embedding_model=embedder.name,
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]


def process_document(
    document_id: str, engine: Engine, ocr: OcrEngine, embedder: Embedder, settings: Settings
) -> None:
    """Extract and index the text of one document and mark it as processed or failed.

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
            # Pages and chunks are committed together: "processed" always means searchable.
            session.add_all(_build_chunks(document, pages, embedder, settings))
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
