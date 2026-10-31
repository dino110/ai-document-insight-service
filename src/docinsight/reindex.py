"""Re-index processed documents whose chunks or entities are missing or outdated.

Documents processed before indexing existed have pages but no chunks, and chunks embedded
with another model (e.g. hashing while no API key was set) or another text format (before
chunk headers) are not searchable with the current one. Likewise, documents processed
before NER have no entities, and entities of another extractor (other rules or model
versions) are stale. Both are rebuilt from the stored page text, so OCR never runs again.

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
from docinsight.nlp.ner import EntityExtractor, create_entity_extractor
from docinsight.processing import build_chunks, extract_entities
from docinsight.storage.db import make_engine
from docinsight.storage.models import Document, DocumentChunk, DocumentEntity, DocumentPage

logger = logging.getLogger(__name__)


@dataclass
class ReindexStats:
    checked: int = 0
    up_to_date: int = 0  # both chunks and entities are current
    reindexed: int = 0  # new chunks; with --dry-run: would get them
    entities_refreshed: int = 0  # new entities; with --dry-run: would get them
    failed: int = 0
    chunks_written: int = 0
    entities_written: int = 0


def reindex(
    engine: Engine,
    embedder: Embedder,
    entity_extractor: EntityExtractor,
    settings: Settings,
    *,
    dry_run: bool = False,
) -> ReindexStats:
    """Rebuild what is outdated in every processed document: chunks, entities or both.

    One transaction per document: old rows are replaced only when the new ones are ready,
    so a failure leaves the document as it was. A document without any entity is always
    rebuilt (a few milliseconds): without a marker, "none found" looks like "never run".
    """
    stats = ReindexStats()
    current_index = index_name(embedder.name)
    with Session(engine) as session:
        document_ids = session.exec(
            select(Document.id).where(Document.status == "processed").order_by(col(Document.id))
        ).all()

    for document_id in document_ids:
        stats.checked += 1
        with Session(engine) as session:
            index_names = session.exec(
                select(DocumentChunk.embedding_model).where(
                    DocumentChunk.document_id == document_id
                )
            ).all()
            extractors = session.exec(
                select(DocumentEntity.extractor).where(DocumentEntity.document_id == document_id)
            ).all()
            chunks_stale = set(index_names) != {current_index}
            entities_stale = set(extractors) != {entity_extractor.name}
            if not chunks_stale and not entities_stale:
                stats.up_to_date += 1
                continue
            if dry_run:
                stats.reindexed += chunks_stale
                stats.entities_refreshed += entities_stale
                continue

            document = session.get(Document, document_id)
            if document is None:  # deleted since the list was read
                continue
            chunks: list[DocumentChunk] = []
            entities: list[DocumentEntity] = []
            try:
                pages = [
                    PageText(page.page_number, page.text, _method(page.method))
                    for page in session.exec(
                        select(DocumentPage)
                        .where(DocumentPage.document_id == document_id)
                        .order_by(col(DocumentPage.page_number))
                    )
                ]
                if chunks_stale:
                    chunks = build_chunks(document, pages, embedder, settings)
                    session.exec(
                        delete(DocumentChunk).where(col(DocumentChunk.document_id) == document_id)
                    )
                    session.add_all(chunks)
                if entities_stale:
                    # extract_entities, not build_entities: a failure must roll back here,
                    # not replace the old entities with none.
                    entities = extract_entities(document, pages, entity_extractor)
                    session.exec(
                        delete(DocumentEntity).where(col(DocumentEntity.document_id) == document_id)
                    )
                    session.add_all(entities)
                session.commit()
            except Exception:
                logger.exception("Re-indexing failed for document %s", document_id)
                session.rollback()
                stats.failed += 1
                continue
            stats.reindexed += chunks_stale
            stats.entities_refreshed += entities_stale
            stats.chunks_written += len(chunks)
            stats.entities_written += len(entities)
    return stats


def _method(value: str) -> ExtractionMethod:
    # Only "text_layer" and "ocr" are ever stored; the method does not affect chunking.
    return "ocr" if value == "ocr" else "text_layer"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m docinsight.reindex",
        description="Re-index processed documents whose chunks or entities are outdated.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="only report what would be re-indexed"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    engine = make_engine(settings.database_url)
    embedder = create_embedder(settings)
    entity_extractor = create_entity_extractor(settings)
    stats = reindex(engine, embedder, entity_extractor, settings, dry_run=args.dry_run)
    engine.dispose()

    action = "would re-index" if args.dry_run else "re-indexed"
    refresh = "would refresh" if args.dry_run else "refreshed"
    print(
        f"Index {index_name(embedder.name)}: checked {stats.checked} processed documents, "
        f"{stats.up_to_date} up to date, {action} {stats.reindexed}, failed {stats.failed}, "
        f"{stats.chunks_written} chunks written.\n"
        f"Entities {entity_extractor.name}: {refresh} {stats.entities_refreshed}, "
        f"{stats.entities_written} entities written."
    )
    return 1 if stats.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
