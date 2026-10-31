"""Re-index processed documents whose chunks are missing or come from another embedder.

Documents processed before indexing existed have pages but no chunks, and chunks embedded
with another model (e.g. hashing while no API key was set) or another text format (before
chunk headers) are not searchable with the current one. Chunks are rebuilt from the stored
page text, so OCR never runs again.

    python -m docinsight.reindex [--dry-run]          # locally: uv run python -m ...
    docker compose exec api python -m docinsight.reindex
"""

import argparse
import logging
from dataclasses import dataclass

from sqlalchemy import Engine
from sqlmodel import Session, col, delete, select

from docinsight.config import Settings, get_settings
from docinsight.extraction.service import ExtractionMethod, PageText
from docinsight.indexing.chunking import index_name
from docinsight.indexing.embeddings import Embedder, create_embedder
from docinsight.processing import build_chunks
from docinsight.storage.db import make_engine
from docinsight.storage.models import Document, DocumentChunk, DocumentPage

logger = logging.getLogger(__name__)


@dataclass
class ReindexStats:
    checked: int = 0
    up_to_date: int = 0
    reindexed: int = 0  # with --dry-run: would be re-indexed
    failed: int = 0
    chunks_written: int = 0


def reindex(
    engine: Engine, embedder: Embedder, settings: Settings, *, dry_run: bool = False
) -> ReindexStats:
    """Rebuild the chunks of every processed document that is not indexed with ``embedder``.

    One transaction per document: old chunks are replaced only when the new ones are
    ready, so a failure leaves the document as searchable as it was before.
    """
    stats = ReindexStats()
    current = index_name(embedder.name)
    with Session(engine) as session:
        document_ids = session.exec(
            select(Document.id).where(Document.status == "processed").order_by(col(Document.id))
        ).all()

    for document_id in document_ids:
        stats.checked += 1
        with Session(engine) as session:
            models = set(
                session.exec(
                    select(DocumentChunk.embedding_model).where(
                        DocumentChunk.document_id == document_id
                    )
                ).all()
            )
            if models == {current}:
                stats.up_to_date += 1
                continue
            if dry_run:
                stats.reindexed += 1
                continue

            document = session.get(Document, document_id)
            if document is None:  # deleted since the list was read
                continue
            try:
                pages = [
                    PageText(page.page_number, page.text, _method(page.method))
                    for page in session.exec(
                        select(DocumentPage)
                        .where(DocumentPage.document_id == document_id)
                        .order_by(col(DocumentPage.page_number))
                    )
                ]
                chunks = build_chunks(document, pages, embedder, settings)
                session.exec(
                    delete(DocumentChunk).where(col(DocumentChunk.document_id) == document_id)
                )
                session.add_all(chunks)
                session.commit()
            except Exception:
                logger.exception("Re-indexing failed for document %s", document_id)
                session.rollback()
                stats.failed += 1
                continue
            stats.reindexed += 1
            stats.chunks_written += len(chunks)
    return stats


def _method(value: str) -> ExtractionMethod:
    # Only "text_layer" and "ocr" are ever stored; the method does not affect chunking.
    return "ocr" if value == "ocr" else "text_layer"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m docinsight.reindex",
        description="Re-index processed documents whose chunks are missing or outdated.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="only report what would be re-indexed"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    engine = make_engine(settings.database_url)
    embedder = create_embedder(settings)
    stats = reindex(engine, embedder, settings, dry_run=args.dry_run)
    engine.dispose()

    action = "would re-index" if args.dry_run else "re-indexed"
    print(
        f"Index {index_name(embedder.name)}: checked {stats.checked} processed documents, "
        f"{stats.up_to_date} up to date, {action} {stats.reindexed}, failed {stats.failed}, "
        f"{stats.chunks_written} chunks written."
    )
    return 1 if stats.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
