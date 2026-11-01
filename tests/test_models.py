"""Tests for database tables that are not covered through the API."""

from pathlib import Path

import numpy as np
from sqlmodel import Session, SQLModel, create_engine, select

from docinsight.config import Settings
from docinsight.storage.db import make_engine
from docinsight.storage.models import Document, DocumentChunk, DocumentEntity, DocumentFields


def test_chunk_embedding_round_trips_through_sqlite(tmp_path: Path) -> None:
    engine = make_engine(Settings(data_dir=tmp_path).database_url)
    vector = np.random.default_rng(0).random(1536, dtype=np.float32)
    document = Document(
        owner_id="session-a",
        filename="racun.pdf",
        content_type="application/pdf",
        size_bytes=1,
        sha256="0" * 64,
        stored_path="x",
    )

    with Session(engine) as session:
        session.add(document)
        session.add(
            DocumentChunk(
                document_id=document.id,
                owner_id=document.owner_id,
                page_number=1,
                chunk_index=0,
                text="Ukupno: 1.575,00 EUR",
                embedding=vector.tobytes(),
                embedding_model="text-embedding-3-small",
            )
        )
        session.commit()

    with Session(engine) as session:
        chunk = session.exec(select(DocumentChunk)).one()

    assert len(chunk.embedding) == 1536 * 4  # float32 = 4 bytes
    assert np.array_equal(np.frombuffer(chunk.embedding, dtype=np.float32), vector)


def test_new_tables_are_added_to_an_existing_database(tmp_path: Path) -> None:
    url = Settings(data_dir=tmp_path).database_url
    # A database from before phase 5: every table except the entities, with a document.
    old = create_engine(url)
    new_tables = {"documententity", "documentfields"}
    tables = [t for t in SQLModel.metadata.sorted_tables if t.name not in new_tables]
    SQLModel.metadata.create_all(old, tables=tables)
    with Session(old) as session:
        session.add(
            Document(
                owner_id="a",
                filename="racun.pdf",
                content_type="application/pdf",
                size_bytes=1,
                sha256="0" * 64,
                stored_path="x",
            )
        )
        session.commit()
    old.dispose()

    engine = make_engine(url)  # what lifespan does at startup

    with Session(engine) as session:
        assert len(session.exec(select(Document)).all()) == 1  # nothing was deleted
        assert session.exec(select(DocumentEntity)).all() == []  # the new tables exist
        assert session.exec(select(DocumentFields)).all() == []


def test_fields_json_columns_round_trip(tmp_path: Path) -> None:
    engine = make_engine(Settings(data_dir=tmp_path).database_url)
    document = Document(
        owner_id="a",
        filename="racun.pdf",
        content_type="application/pdf",
        size_bytes=1,
        sha256="0" * 64,
        stored_path="x",
    )
    with Session(engine) as session:
        session.add(document)
        session.add(
            DocumentFields(
                document_id=document.id,
                owner_id="a",
                document_type="invoice",
                data={"number": "2026-001", "issuer": None},
                checks={"amounts_consistent": True, "names_in_entities": None},
                warnings=["buyer_oib has an invalid check digit"],
                model="fake-llm",
            )
        )
        session.commit()

    with Session(engine) as session:
        stored = session.exec(select(DocumentFields)).one()
    assert stored.data == {"number": "2026-001", "issuer": None}
    assert stored.checks == {"amounts_consistent": True, "names_in_entities": None}
    assert stored.warnings == ["buyer_oib has an invalid check digit"]
