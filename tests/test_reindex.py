"""Tests for re-indexing documents from their stored page text."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from sqlalchemy import Engine
from sqlmodel import Session, col, select

from docinsight.config import Settings, get_settings
from docinsight.indexing.chunking import index_name
from docinsight.indexing.embeddings import HashingEmbedder
from docinsight.nlp.ner import Entity, RuleEntityExtractor
from docinsight.qa.guard import Language
from docinsight.reindex import main, reindex
from docinsight.storage.db import make_engine
from docinsight.storage.models import Document, DocumentChunk, DocumentEntity, DocumentPage

SETTINGS = Settings()
EXTRACTOR = RuleEntityExtractor()


class CountingEmbedder(HashingEmbedder):
    """HashingEmbedder that counts calls, so tests can see when nothing was embedded."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def embed_documents(self, texts: list[str]) -> NDArray[np.float32]:
        self.calls += 1
        return super().embed_documents(texts)


@pytest.fixture
def engine(tmp_path: Path) -> Engine:
    return make_engine(Settings(data_dir=tmp_path).database_url)


def add_document(
    engine: Engine,
    pages: list[str],
    *,
    status: str = "processed",
    chunk_model: str | None = None,
    entity_extractor: str | None = None,
) -> str:
    """A document with stored page text, optionally one chunk and one entity.

    The chunk has the text "stale" and the entity the text "Stara d.o.o.", so tests can
    see whether they were replaced.
    """
    with Session(engine) as session:
        document = Document(
            owner_id="owner",
            filename="racun.pdf",
            content_type="application/pdf",
            size_bytes=1,
            sha256="0" * 64,
            stored_path="unused",
            status=status,
        )
        session.add(document)
        session.add_all(
            DocumentPage(document_id=document.id, page_number=i, text=text, method="ocr")
            for i, text in enumerate(pages, start=1)
        )
        if chunk_model is not None:
            session.add(
                DocumentChunk(
                    document_id=document.id,
                    owner_id="owner",
                    page_number=1,
                    chunk_index=0,
                    text="stale",
                    embedding=b"\x00" * 16,
                    embedding_model=chunk_model,
                )
            )
        if entity_extractor is not None:
            session.add(
                DocumentEntity(
                    document_id=document.id,
                    owner_id="owner",
                    page_number=1,
                    label="ORG",
                    text="Stara d.o.o.",
                    normalized="Stara d.o.o.",
                    start=0,
                    end=12,
                    extractor=entity_extractor,
                )
            )
        session.commit()
        return document.id


def chunks_of(engine: Engine, document_id: str) -> list[DocumentChunk]:
    with Session(engine) as session:
        return list(
            session.exec(
                select(DocumentChunk)
                .where(DocumentChunk.document_id == document_id)
                .order_by(col(DocumentChunk.chunk_index))
            )
        )


def test_document_without_chunks_is_indexed_from_its_pages(engine: Engine) -> None:
    embedder = CountingEmbedder()
    document_id = add_document(engine, ["Kupac: Horvat d.o.o.", "Ukupno: 1.575,00 EUR"])

    stats = reindex(engine, embedder, EXTRACTOR, SETTINGS)

    chunks = chunks_of(engine, document_id)
    assert [(c.page_number, c.text) for c in chunks] == [
        (1, "Kupac: Horvat d.o.o."),
        (2, "Ukupno: 1.575,00 EUR"),
    ]
    assert {c.embedding_model for c in chunks} == {index_name(embedder.name)}
    assert (stats.checked, stats.reindexed, stats.chunks_written) == (1, 1, 2)


def test_chunks_of_another_model_are_replaced(engine: Engine) -> None:
    embedder = CountingEmbedder()
    document_id = add_document(engine, ["Datum dospijeća: 19.01.2026."], chunk_model="old-model")

    reindex(engine, embedder, EXTRACTOR, SETTINGS)

    [chunk] = chunks_of(engine, document_id)
    assert chunk.embedding_model == index_name(embedder.name)
    assert chunk.text == "Datum dospijeća: 19.01.2026."
    assert len(chunk.embedding) == embedder.dim * 4


def test_chunks_without_headers_are_replaced(engine: Engine) -> None:
    embedder = CountingEmbedder()
    # Indexed by the same embedder before chunk headers existed.
    document_id = add_document(engine, ["Tekst."], chunk_model=embedder.name)

    stats = reindex(engine, embedder, EXTRACTOR, SETTINGS)

    [chunk] = chunks_of(engine, document_id)
    assert chunk.embedding_model == index_name(embedder.name)
    assert stats.reindexed == 1


def test_up_to_date_documents_are_not_embedded_again(engine: Engine) -> None:
    embedder = CountingEmbedder()
    add_document(
        engine, ["Tekst."], chunk_model=index_name(embedder.name), entity_extractor=EXTRACTOR.name
    )

    stats = reindex(engine, embedder, EXTRACTOR, SETTINGS)

    assert embedder.calls == 0
    assert (stats.up_to_date, stats.reindexed) == (1, 0)


def test_documents_that_are_not_processed_are_ignored(engine: Engine) -> None:
    add_document(engine, ["Tekst."], status="failed")
    add_document(engine, ["Tekst."], status="pending")

    assert reindex(engine, CountingEmbedder(), EXTRACTOR, SETTINGS).checked == 0


def test_dry_run_changes_nothing(engine: Engine) -> None:
    embedder = CountingEmbedder()
    document_id = add_document(engine, ["Tekst."], chunk_model="old-model")

    stats = reindex(engine, embedder, EXTRACTOR, SETTINGS, dry_run=True)

    assert stats.reindexed == 1
    assert embedder.calls == 0
    assert [c.embedding_model for c in chunks_of(engine, document_id)] == ["old-model"]


def test_failure_keeps_old_chunks_and_continues(
    engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    embedder = CountingEmbedder()
    broken = add_document(engine, ["Prvi."], chunk_model="old-model")
    fine = add_document(engine, ["Drugi."])
    real_embed = embedder.embed_documents

    def fail_once(texts: list[str]) -> NDArray[np.float32]:
        if any(text.endswith("Prvi.") for text in texts):  # embedded with its header
            raise RuntimeError("API down")
        return real_embed(texts)

    monkeypatch.setattr(embedder, "embed_documents", fail_once)

    stats = reindex(engine, embedder, EXTRACTOR, SETTINGS)

    assert [c.text for c in chunks_of(engine, broken)] == ["stale"]  # rolled back
    assert [c.text for c in chunks_of(engine, fine)] == ["Drugi."]
    assert (stats.failed, stats.reindexed) == (1, 1)


@pytest.fixture
def fresh_settings(data_dir: Path) -> Iterator[Path]:
    """main() reads the cached get_settings(): make it see this test's DOCINSIGHT_DATA_DIR."""
    get_settings.cache_clear()
    yield data_dir
    get_settings.cache_clear()


def test_main_uses_settings_and_prints_a_summary(
    fresh_settings: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_dir = fresh_settings
    engine = make_engine(Settings(data_dir=data_dir).database_url)
    add_document(engine, ["Kupac: Horvat d.o.o."])

    assert main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "would re-index 1" in out
    assert "Entities rules-3: would refresh 1, 0 entities written." in out

    assert main([]) == 0  # no API key in tests, so the hashing embedder is used
    out = capsys.readouterr().out
    assert "re-indexed 1, failed 0, 1 chunks written" in out
    assert "Entities rules-3: refreshed 1, 1 entities written." in out  # Horvat d.o.o.
    assert main([]) == 0
    assert "1 up to date, re-indexed 0" in capsys.readouterr().out


# --- Entities -------------------------------------------------------------------


def entities_of(engine: Engine, document_id: str) -> list[DocumentEntity]:
    with Session(engine) as session:
        return list(
            session.exec(
                select(DocumentEntity)
                .where(DocumentEntity.document_id == document_id)
                .order_by(col(DocumentEntity.start))
            )
        )


def test_document_without_entities_gets_them_without_embedding(engine: Engine) -> None:
    embedder = CountingEmbedder()
    document_id = add_document(
        engine,
        ["Kupac Horvat d.o.o., OIB 99083863799", "Ukupno 1.575,00 EUR"],
        chunk_model=index_name(embedder.name),
    )

    stats = reindex(engine, embedder, EXTRACTOR, SETTINGS)

    entities = entities_of(engine, document_id)
    assert sorted((e.page_number, e.label, e.normalized) for e in entities) == [
        (1, "OIB", "99083863799"),
        (1, "ORG", "Horvat d.o.o."),
        (2, "MONEY", "1575.00"),
    ]
    assert {e.extractor for e in entities} == {"rules-3"}
    assert embedder.calls == 0  # the chunks were current
    assert (stats.reindexed, stats.entities_refreshed, stats.entities_written) == (0, 1, 3)


def test_entities_of_another_extractor_are_replaced(engine: Engine) -> None:
    document_id = add_document(engine, ["Kupac Horvat d.o.o."], entity_extractor="rules-0")

    reindex(engine, CountingEmbedder(), EXTRACTOR, SETTINGS)

    assert [(e.text, e.extractor) for e in entities_of(engine, document_id)] == [
        ("Horvat d.o.o.", "rules-3")
    ]


def test_current_entities_are_kept_when_only_chunks_are_rebuilt(engine: Engine) -> None:
    document_id = add_document(
        engine, ["Kupac Horvat d.o.o."], chunk_model="old-model", entity_extractor=EXTRACTOR.name
    )

    stats = reindex(engine, CountingEmbedder(), EXTRACTOR, SETTINGS)

    assert [e.text for e in entities_of(engine, document_id)] == ["Stara d.o.o."]  # untouched
    assert [c.text for c in chunks_of(engine, document_id)] == ["Kupac Horvat d.o.o."]
    assert (stats.reindexed, stats.entities_refreshed) == (1, 0)


class FailingExtractor:
    name = "failing"

    def extract(self, text: str, language: Language) -> list[Entity]:
        raise RuntimeError("model crashed")


def test_ner_failure_rolls_back_the_whole_document(engine: Engine) -> None:
    document_id = add_document(
        engine, ["Kupac Horvat d.o.o."], chunk_model="old-model", entity_extractor="rules-1"
    )

    stats = reindex(engine, CountingEmbedder(), FailingExtractor(), SETTINGS)

    # Neither the new chunks nor an empty entity list replaced the old rows.
    assert [c.text for c in chunks_of(engine, document_id)] == ["stale"]
    assert [e.text for e in entities_of(engine, document_id)] == ["Stara d.o.o."]
    assert (stats.failed, stats.reindexed, stats.entities_refreshed) == (1, 0, 0)


def test_dry_run_counts_entities_and_changes_nothing(engine: Engine) -> None:
    embedder = CountingEmbedder()
    document_id = add_document(
        engine, ["Kupac Horvat d.o.o."], chunk_model=index_name(embedder.name)
    )

    stats = reindex(engine, embedder, EXTRACTOR, SETTINGS, dry_run=True)

    assert (stats.reindexed, stats.entities_refreshed) == (0, 1)
    assert entities_of(engine, document_id) == []
