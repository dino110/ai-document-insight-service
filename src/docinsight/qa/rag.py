"""Retrieval-augmented answering: embed the question, find chunks, answer from them only."""

import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass

from sqlmodel import Session

from docinsight.config import Settings
from docinsight.indexing.chunking import index_name
from docinsight.indexing.embeddings import Embedder, EmbeddingError
from docinsight.indexing.search import ScoredChunk, search_chunks
from docinsight.nlp.masking import IdentifierMasker, mask_identifiers
from docinsight.qa.generator import AnswerGenerator, LlmRequestError
from docinsight.qa.prompts import ContextChunk, HistoryTurn

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
    history: Sequence[HistoryTurn] = (),
) -> RagResult:
    """Answer ``question`` from the owner's documents, with the cited chunks as sources.

    OIBs and IBANs never reach OpenAI: the question is embedded with them masked, and the
    LLM sees numbered placeholders that are replaced with the real values in its answer.
    Search itself (BM25) runs locally on the unmasked text.

    Raises NoDocumentsError when there is nothing to search, LlmRequestError when the
    embedding or LLM call fails and LlmUnavailableError when no LLM is configured.
    """
    started = time.perf_counter()
    query = search_text(question, history)
    try:
        query_vector = embedder.embed_query(mask_identifiers(query))
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
        query_text=query if settings.retrieval_mode == "hybrid" else None,
    )
    retrieval_ms = _ms_since(started)
    if not hits:
        raise NoDocumentsError("No processed documents to search")

    masker = IdentifierMasker()  # one per request: the same OIB is [OIB_1] everywhere
    contexts = [
        ContextChunk(
            n,
            masker.mask(hit.filename),
            hit.page_number,
            masker.mask(hit.text),
            masker.mask(hit.intro),
        )
        for n, hit in enumerate(hits, start=1)
    ]
    masked_history = [HistoryTurn(turn.role, masker.mask(turn.text)) for turn in history]
    started = time.perf_counter()
    llm_answer = generator.generate(masker.mask(question), contexts, masked_history)
    generation_ms = _ms_since(started)

    sources, numbers = _cited_chunks(llm_answer.citations, hits) if llm_answer.found else ([], {})
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
        answer=masker.unmask(_renumber_markers(llm_answer.answer, numbers)),
        found=llm_answer.found,
        sources=sources,
        model=generator.model,
        timings_ms={"retrieval": retrieval_ms, "generation": generation_ms},
    )


def search_text(question: str, history: Sequence[HistoryTurn]) -> str:
    """The text to search with: the previous question and this one.

    "A PDV?" alone finds any invoice; "Koliki je iznos računa 2026-002? A PDV?" keeps the
    invoice number in the search, without an extra LLM call to rewrite the question.
    """
    previous = next((turn.text for turn in reversed(history) if turn.role == "user"), None)
    return f"{previous} {question}" if previous else question


def _cited_chunks(
    citations: list[int], hits: list[ScoredChunk]
) -> tuple[list[ScoredChunk], dict[int, int]]:
    """Map the model's [n] numbers back to chunks. Unknown numbers and repeats are dropped,
    so a source can only ever be a chunk that was really in the context.

    Also returns excerpt number -> position in the sources (1-based), for the markers."""
    numbers: dict[int, int] = {}
    cited: list[ScoredChunk] = []
    for n in citations:
        if 1 <= n <= len(hits) and n not in numbers:
            cited.append(hits[n - 1])
            numbers[n] = len(cited)
    return cited, numbers


_MARKER = re.compile(r"(\s*)\[(\d+)\]")


def _renumber_markers(answer: str, numbers: dict[int, int]) -> str:
    """Make [n] point at ``sources``: the model numbers all excerpts it saw, the client only
    gets the cited ones. A marker of an excerpt that is not a source is removed."""

    def replace(match: re.Match[str]) -> str:
        position = numbers.get(int(match.group(2)))
        return f"{match.group(1)}[{position}]" if position else ""

    return _MARKER.sub(replace, answer)


def _ms_since(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)
