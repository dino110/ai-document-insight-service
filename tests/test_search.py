"""Tests for vector search over stored chunks, with the local HashingEmbedder (no OpenAI)."""

import pytest
from factories import add_indexed_document
from sqlmodel import Session

from docinsight.indexing.embeddings import HashingEmbedder
from docinsight.indexing.search import ScoredChunk, search_chunks

EMBEDDER = HashingEmbedder()


@pytest.fixture(params=[True, False], ids=["faiss", "numpy"])
def use_faiss(request: pytest.FixtureRequest) -> bool:
    """Every search test runs against both implementations."""
    value: bool = request.param
    return value


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
    document_id = add_indexed_document(session, INVOICE, EMBEDDER)

    results = search(session, "Koji je datum dospijeća?", use_faiss)

    assert results[0].page_number == 2
    assert results[0].document_id == document_id
    assert results[0].filename == "racun.pdf"
    assert results[0].text == INVOICE[1]
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)


def test_top_k_larger_than_number_of_chunks(session: Session, use_faiss: bool) -> None:
    add_indexed_document(session, INVOICE, EMBEDDER)

    assert len(search(session, "račun", use_faiss, top_k=20)) == 3
    assert len(search(session, "račun", use_faiss, top_k=2)) == 2


def test_other_owners_chunks_are_never_returned(session: Session, use_faiss: bool) -> None:
    add_indexed_document(
        session, ["Tajni ugovor o djelu, naknada 9.000,00 EUR"], EMBEDDER, owner_id="someone-else"
    )
    add_indexed_document(session, INVOICE, EMBEDDER)

    results = search(session, "ugovor o djelu naknada", use_faiss)

    assert {r.filename for r in results} == {"racun.pdf"}
    assert all("Tajni" not in r.text for r in results)


def test_chunks_of_another_model_are_ignored(session: Session, use_faiss: bool) -> None:
    add_indexed_document(
        session, ["Datum dospijeća: 1.1.2027."], EMBEDDER, filename="old.pdf", model="old-model"
    )

    assert search(session, "datum dospijeća", use_faiss) == []


def test_documents_that_are_not_processed_are_ignored(session: Session, use_faiss: bool) -> None:
    add_indexed_document(session, ["Datum dospijeća: 1.1.2027."], EMBEDDER, status="failed")

    assert search(session, "datum dospijeća", use_faiss) == []


def test_document_ids_limit_the_search(session: Session, use_faiss: bool) -> None:
    add_indexed_document(session, INVOICE, EMBEDDER, filename="a.pdf")
    other = add_indexed_document(session, INVOICE, EMBEDDER, filename="b.pdf")

    results = search(session, "datum dospijeća", use_faiss, document_ids=[other])

    assert {r.filename for r in results} == {"b.pdf"}


def test_no_chunks_returns_empty_list(session: Session, use_faiss: bool) -> None:
    assert search(session, "bilo što", use_faiss) == []


def test_faiss_and_numpy_find_the_same_chunks(session: Session) -> None:
    add_indexed_document(session, INVOICE, EMBEDDER)
    add_indexed_document(
        session, ["Ugovor o djelu, naknada 2.000,00 EUR", "Rok: 31.12.2026."], EMBEDDER
    )

    with_faiss = search(session, "Koliko iznosi naknada za platiti?", use_faiss=True)
    with_numpy = search(session, "Koliko iznosi naknada za platiti?", use_faiss=False)

    # Same scores in the same order...
    assert [r.score for r in with_faiss] == pytest.approx([r.score for r in with_numpy])

    # ...and the same chunks. Chunks with equal scores (here several with no shared word,
    # score 0) may come in a different order, so compare them sorted by score, then text.
    def ranked(results: list[ScoredChunk]) -> list[tuple[float, str]]:
        return sorted((-round(r.score, 5), r.text) for r in results)

    assert ranked(with_faiss) == ranked(with_numpy)


# --- Hybrid search ---

INVOICES = [
    "Račun br. INV-7781. Datum dospijeća: 19.01.2026. Ukupno za platiti: 9.537,50 EUR",
    "Račun br. INV-1203. Datum dospijeća: 29.03.2026. Ukupno za platiti: 1.575,00 EUR",
    "Ugovor o djelu: izrada mobilne aplikacije, naknada po fazama",
]


def test_hybrid_finds_an_identifier_the_embedding_misses(session: Session, use_faiss: bool) -> None:
    add_indexed_document(session, INVOICES, EMBEDDER)
    # Stand-in for a semantic model that "understands" the question but ignores the
    # number: its vector points at the contract, not at invoice INV-7781.
    misleading_vector = EMBEDDER.embed_query(INVOICES[2])

    def first_page(query_text: str | None) -> int:
        [best, *_] = search_chunks(
            session,
            "owner",
            misleading_vector,
            EMBEDDER.name,
            3,
            query_text=query_text,
            use_faiss=use_faiss,
        )
        return best.page_number

    assert first_page(None) == 3  # dense only: the contract
    assert first_page("Kada dospijeva račun INV-7781?") == 1  # hybrid: the right invoice


def test_hybrid_scores_are_between_zero_and_one(session: Session, use_faiss: bool) -> None:
    add_indexed_document(session, INVOICES, EMBEDDER)
    question = "Ukupno za platiti INV-1203"

    results = search_chunks(
        session,
        "owner",
        EMBEDDER.embed_query(question),
        EMBEDDER.name,
        5,
        query_text=question,
        use_faiss=use_faiss,
    )

    assert len(results) == 3  # chunks without a shared token still come from the dense list
    assert results[0].page_number == 2
    assert results[0].score == pytest.approx(1.0)  # first in both rankings
    assert all(0 < r.score <= 1 for r in results)
