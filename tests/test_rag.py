"""Tests for the RAG flow with a fake LLM and the local HashingEmbedder."""

from typing import Any

import pytest
from factories import FakeGenerator, add_indexed_document
from sqlmodel import Session

from docinsight.config import Settings
from docinsight.indexing.embeddings import EmbeddingError, HashingEmbedder
from docinsight.indexing.search import ScoredChunk, search_chunks
from docinsight.qa import rag as rag_module
from docinsight.qa.generator import LlmAnswer, LlmRequestError
from docinsight.qa.rag import NoDocumentsError, RagResult, answer_question

EMBEDDER = HashingEmbedder()
SETTINGS = Settings(top_k=3)
INVOICE = [
    "Račun br. 2026-001, Kupac: Horvat d.o.o., Zagreb",
    "Datum dospijeća: 15.11.2026. Ukupno za platiti: 1.575,00 EUR",
]


def ask(
    session: Session,
    generator: FakeGenerator,
    question: str = "Koliko je ukupno za platiti?",
    *,
    document_ids: list[str] | None = None,
    top_k: int | None = None,
) -> RagResult:
    return answer_question(
        question, "owner", session, EMBEDDER, generator, SETTINGS, document_ids, top_k
    )


def test_citations_map_to_file_and_page(session: Session) -> None:
    add_indexed_document(session, INVOICE, EMBEDDER, filename="racun-2026-001.pdf")
    generator = FakeGenerator(LlmAnswer(answer="1.575,00 EUR [1]", found=True, citations=[1]))

    result = ask(session, generator)

    # The best match is context [1]: page 2 holds "Ukupno za platiti".
    [(question, contexts)] = generator.calls
    assert question == "Koliko je ukupno za platiti?"
    assert [c.n for c in contexts] == list(range(1, len(contexts) + 1))
    assert contexts[0].page_number == 2
    [source] = result.sources
    assert (source.filename, source.page_number) == ("racun-2026-001.pdf", 2)
    assert result.answer == "1.575,00 EUR [1]"
    assert result.found is True
    assert result.model == "fake-llm"
    assert set(result.timings_ms) == {"retrieval", "generation"}


def test_unknown_and_repeated_citations_are_dropped(session: Session) -> None:
    add_indexed_document(session, INVOICE, EMBEDDER)
    generator = FakeGenerator(LlmAnswer(answer="...", found=True, citations=[2, 9, 0, 2, 1]))

    result = ask(session, generator)

    [(_, contexts)] = generator.calls
    assert [s.text for s in result.sources] == [contexts[1].text, contexts[0].text]


def test_not_found_has_no_sources(session: Session) -> None:
    add_indexed_document(session, INVOICE, EMBEDDER)
    generator = FakeGenerator(LlmAnswer(answer="Nema podatka.", found=False, citations=[1]))

    result = ask(session, generator, "Koja je kamatna stopa?")

    assert result.found is False
    assert result.sources == []


def test_generator_gets_at_most_top_k_contexts(session: Session) -> None:
    add_indexed_document(session, [f"Stavka {i}: usluga račun" for i in range(10)], EMBEDDER)
    generator = FakeGenerator()

    ask(session, generator)
    ask(session, generator, top_k=5)

    assert [len(contexts) for _, contexts in generator.calls] == [3, 5]


def test_document_ids_limit_the_contexts(session: Session) -> None:
    add_indexed_document(session, INVOICE, EMBEDDER, filename="a.pdf")
    other = add_indexed_document(session, INVOICE, EMBEDDER, filename="b.pdf")
    generator = FakeGenerator()

    ask(session, generator, document_ids=[other])

    [(_, contexts)] = generator.calls
    assert {c.filename for c in contexts} == {"b.pdf"}


def test_no_documents_raises(session: Session) -> None:
    generator = FakeGenerator()

    with pytest.raises(NoDocumentsError):
        ask(session, generator)
    assert generator.calls == []  # the LLM is not called without context


def test_embedding_error_becomes_llm_request_error(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(text: str) -> object:
        raise EmbeddingError("timeout")

    monkeypatch.setattr(EMBEDDER, "embed_query", explode)

    with pytest.raises(LlmRequestError):
        ask(session, FakeGenerator())


def test_dense_mode_does_not_use_the_question_text(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    add_indexed_document(session, INVOICE, EMBEDDER)
    received: list[str | None] = []

    # Any: a spy that forwards whatever rag passes on to the real function.
    def spy(*args: Any, **kwargs: Any) -> list[ScoredChunk]:
        received.append(kwargs.get("query_text"))
        return search_chunks(*args, **kwargs)

    monkeypatch.setattr(rag_module, "search_chunks", spy)
    generator = FakeGenerator()

    answer_question("Tko je kupac?", "owner", session, EMBEDDER, generator, SETTINGS)
    dense = Settings(top_k=3, retrieval_mode="dense")
    answer_question("Tko je kupac?", "owner", session, EMBEDDER, generator, dense)

    assert received == ["Tko je kupac?", None]
