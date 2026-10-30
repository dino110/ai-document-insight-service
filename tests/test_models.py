"""Tests for database tables that are not covered through the API."""

from pathlib import Path

import numpy as np
from sqlmodel import Session, select

from docinsight.config import Settings
from docinsight.storage.db import make_engine
from docinsight.storage.models import Document, DocumentChunk


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
