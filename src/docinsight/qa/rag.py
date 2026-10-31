"""Retrieval-augmented answering: embed the question, find chunks, answer from them only."""

import logging
import time
from dataclasses import dataclass

from sqlmodel import Session

from docinsight.config import Settings
from docinsight.indexing.chunking import index_name
from docinsight.indexing.embeddings import Embedder, EmbeddingError
from docinsight.indexing.search import ScoredChunk, search_chunks
from docinsight.qa.generator import AnswerGenerator, LlmRequestError
from docinsight.qa.prompts import ContextChunk

logger = logging.getLogger(__name__)


class NoDocumentsError(RuntimeError):
    """The owner has no processed, indexed chunks to search (in the selected documents)."""


@dataclass(frozen=True)
class RagResult:
    answer: str
    found: bool
    sources: list[ScoredChunk]  # only the chunks the answer cites, in citation order
    model: str
    timings_ms: dict[str, float]


def answer_question(
    question: str,
    owner_id: str,
    session: Session,
    embedder: Embedder,
    generator: AnswerGenerator,
    settings: Settings,
    document_ids: list[str] | None = None,
    top_k: int | None = None,
) -> RagResult:
    """Answer ``question`` from the owner's documents, with the cited chunks as sources.

    Raises NoDocumentsError when there is nothing to search, LlmRequestError when the
    embedding or LLM call fails and LlmUnavailableError when no LLM is configured.
    """
    started = time.perf_counter()
    try:
        query_vector = embedder.embed_query(question)
    except EmbeddingError as exc:
        # For the user it is the same failure as the LLM call: the answer service failed.
        raise LlmRequestError("Embedding the question failed") from exc
    hits = search_chunks(
        session,
        owner_id,
        query_vector,
        index_name(embedder.name),
        top_k or settings.top_k,
        document_ids,
        query_text=question if settings.retrieval_mode == "hybrid" else None,
    )
    retrieval_ms = _ms_since(started)
    if not hits:
        raise NoDocumentsError("No processed documents to search")

    contexts = [
        ContextChunk(n, hit.filename, hit.page_number, hit.text, hit.intro)
        for n, hit in enumerate(hits, start=1)
    ]
    started = time.perf_counter()
    llm_answer = generator.generate(question, contexts)
    generation_ms = _ms_since(started)

    sources = _cited_chunks(llm_answer.citations, hits) if llm_answer.found else []
    # Counts and timings only: never the question or document text.
    logger.info(
        "Answered from %d chunks: found=%s, %d sources, retrieval %.0f ms, generation %.0f ms",
        len(hits),
        llm_answer.found,
        len(sources),
        retrieval_ms,
        generation_ms,
    )
    return RagResult(
        answer=llm_answer.answer,
        found=llm_answer.found,
        sources=sources,
        model=generator.model,
        timings_ms={"retrieval": retrieval_ms, "generation": generation_ms},
    )


def _cited_chunks(citations: list[int], hits: list[ScoredChunk]) -> list[ScoredChunk]:
    """Map the model's [n] numbers back to chunks. Unknown numbers and repeats are dropped,
    so a source can only ever be a chunk that was really in the context."""
    seen: set[int] = set()
    cited: list[ScoredChunk] = []
    for n in citations:
        if 1 <= n <= len(hits) and n not in seen:
            seen.add(n)
            cited.append(hits[n - 1])
    return cited


def _ms_since(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)
