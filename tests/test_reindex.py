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
from docinsight.reindex import main, reindex
from docinsight.storage.db import make_engine
from docinsight.storage.models import Document, DocumentChunk, DocumentPage

SETTINGS = Settings()


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
) -> str:
    """A document with stored page text and, with ``chunk_model``, one stale chunk."""
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

    stats = reindex(engine, embedder, SETTINGS)

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

    reindex(engine, embedder, SETTINGS)

    [chunk] = chunks_of(engine, document_id)
    assert chunk.embedding_model == index_name(embedder.name)
    assert chunk.text == "Datum dospijeća: 19.01.2026."
    assert len(chunk.embedding) == embedder.dim * 4


def test_chunks_without_headers_are_replaced(engine: Engine) -> None:
    embedder = CountingEmbedder()
    # Indexed by the same embedder before chunk headers existed.
    document_id = add_document(engine, ["Tekst."], chunk_model=embedder.name)

    stats = reindex(engine, embedder, SETTINGS)

    [chunk] = chunks_of(engine, document_id)
    assert chunk.embedding_model == index_name(embedder.name)
    assert stats.reindexed == 1


def test_up_to_date_documents_are_not_embedded_again(engine: Engine) -> None:
    embedder = CountingEmbedder()
    add_document(engine, ["Tekst."], chunk_model=index_name(embedder.name))

    stats = reindex(engine, embedder, SETTINGS)

    assert embedder.calls == 0
    assert (stats.up_to_date, stats.reindexed) == (1, 0)


def test_documents_that_are_not_processed_are_ignored(engine: Engine) -> None:
    add_document(engine, ["Tekst."], status="failed")
    add_document(engine, ["Tekst."], status="pending")

    assert reindex(engine, CountingEmbedder(), SETTINGS).checked == 0


def test_dry_run_changes_nothing(engine: Engine) -> None:
    embedder = CountingEmbedder()
    document_id = add_document(engine, ["Tekst."], chunk_model="old-model")

    stats = reindex(engine, embedder, SETTINGS, dry_run=True)

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

    stats = reindex(engine, embedder, SETTINGS)

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
    assert "would re-index 1" in capsys.readouterr().out

    assert main([]) == 0  # no API key in tests, so the hashing embedder is used
    assert "re-indexed 1, failed 0, 1 chunks written" in capsys.readouterr().out
    assert main([]) == 0
    assert "1 up to date, re-indexed 0" in capsys.readouterr().out
