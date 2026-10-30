"""Tests for vector search over stored chunks, with the local HashingEmbedder (no OpenAI)."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlmodel import Session

from docinsight.config import Settings
from docinsight.indexing.embeddings import HashingEmbedder
from docinsight.indexing.search import ScoredChunk, search_chunks
from docinsight.storage.db import make_engine
from docinsight.storage.models import Document, DocumentChunk

EMBEDDER = HashingEmbedder()


@pytest.fixture
def session(tmp_path: Path) -> Iterator[Session]:
    engine = make_engine(Settings(data_dir=tmp_path).database_url)
    with Session(engine) as session:
        yield session


@pytest.fixture(params=[True, False], ids=["faiss", "numpy"])
def use_faiss(request: pytest.FixtureRequest) -> bool:
    """Every search test runs against both implementations."""
    value: bool = request.param
    return value


def add_document(
    session: Session,
    texts: list[str],
    *,
    owner_id: str = "owner",
    filename: str = "racun.pdf",
    status: str = "processed",
    model: str = EMBEDDER.name,
) -> str:
    """Store a document with one chunk per text, the n-th text on page n."""
    document = Document(
        owner_id=owner_id,
        filename=filename,
        content_type="application/pdf",
        size_bytes=1,
        sha256="0" * 64,
        stored_path="unused",
        status=status,
    )
    session.add(document)
    vectors = EMBEDDER.embed_documents(texts)
    session.add_all(
        DocumentChunk(
            document_id=document.id,
            owner_id=owner_id,
            page_number=i + 1,
            chunk_index=i,
            text=text,
            embedding=vector.tobytes(),
            embedding_model=model,
        )
        for i, (text, vector) in enumerate(zip(texts, vectors, strict=True))
    )
    session.commit()
    return document.id


def search(
    session: Session,
    question: str,
    use_faiss: bool,
    *,
    owner_id: str = "owner",
    top_k: int = 5,
    document_ids: list[str] | None = None,
) -> list[ScoredChunk]:
    return search_chunks(
        session,
        owner_id,
        EMBEDDER.embed_query(question),
        EMBEDDER.name,
        top_k,
        document_ids,
        use_faiss=use_faiss,
    )


INVOICE = [
    "Račun br. 2026-001, Kupac: Horvat d.o.o., Zagreb",
    "Datum dospijeća: 15.11.2026. Ukupno za platiti: 1.575,00 EUR",
    "Hvala na suradnji. Dostava: kurirska služba",
]


def test_best_matching_chunk_comes_first(session: Session, use_faiss: bool) -> None:
    document_id = add_document(session, INVOICE)

    results = search(session, "Koji je datum dospijeća?", use_faiss)

    assert results[0].page_number == 2
    assert results[0].document_id == document_id
    assert results[0].filename == "racun.pdf"
    assert results[0].text == INVOICE[1]
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)


def test_top_k_larger_than_number_of_chunks(session: Session, use_faiss: bool) -> None:
    add_document(session, INVOICE)

    assert len(search(session, "račun", use_faiss, top_k=20)) == 3
    assert len(search(session, "račun", use_faiss, top_k=2)) == 2


def test_other_owners_chunks_are_never_returned(session: Session, use_faiss: bool) -> None:
    add_document(session, ["Tajni ugovor o djelu, naknada 9.000,00 EUR"], owner_id="someone-else")
    add_document(session, INVOICE)

    results = search(session, "ugovor o djelu naknada", use_faiss)

    assert {r.filename for r in results} == {"racun.pdf"}
    assert all("Tajni" not in r.text for r in results)


def test_chunks_of_another_model_are_ignored(session: Session, use_faiss: bool) -> None:
    add_document(session, ["Datum dospijeća: 1.1.2027."], filename="old.pdf", model="old-model")

    assert search(session, "datum dospijeća", use_faiss) == []


def test_documents_that_are_not_processed_are_ignored(session: Session, use_faiss: bool) -> None:
    add_document(session, ["Datum dospijeća: 1.1.2027."], status="failed")

    assert search(session, "datum dospijeća", use_faiss) == []


def test_document_ids_limit_the_search(session: Session, use_faiss: bool) -> None:
    add_document(session, INVOICE, filename="a.pdf")
    other = add_document(session, INVOICE, filename="b.pdf")

    results = search(session, "datum dospijeća", use_faiss, document_ids=[other])

    assert {r.filename for r in results} == {"b.pdf"}


def test_no_chunks_returns_empty_list(session: Session, use_faiss: bool) -> None:
    assert search(session, "bilo što", use_faiss) == []


def test_faiss_and_numpy_find_the_same_chunks(session: Session) -> None:
    add_document(session, INVOICE)
    add_document(session, ["Ugovor o djelu, naknada 2.000,00 EUR", "Rok: 31.12.2026."])

    with_faiss = search(session, "Koliko iznosi naknada za platiti?", use_faiss=True)
    with_numpy = search(session, "Koliko iznosi naknada za platiti?", use_faiss=False)

    # Same scores in the same order...
    assert [r.score for r in with_faiss] == pytest.approx([r.score for r in with_numpy])

    # ...and the same chunks. Chunks with equal scores (here several with no shared word,
    # score 0) may come in a different order, so compare them sorted by score, then text.
    def ranked(results: list[ScoredChunk]) -> list[tuple[float, str]]:
        return sorted((-round(r.score, 5), r.text) for r in results)

    assert ranked(with_faiss) == ranked(with_numpy)
