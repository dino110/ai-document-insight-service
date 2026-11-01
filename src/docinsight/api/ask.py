"""Question answering over the caller's documents.

A local router decides first: most questions go to RAG (search + LLM), the rest get a
canned answer or the document list without any embedding or LLM call.
"""

import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import col, select

from docinsight.api.deps import (
    DbSession,
    EmbedderDep,
    EntityExtractorDep,
    GeneratorDep,
    RateLimit,
    RouterDep,
    SettingsDep,
    require_session_id,
)
from docinsight.qa.canned import canned_answer, meta_answer, not_found_answer
from docinsight.qa.generator import LlmRequestError, LlmUnavailableError
from docinsight.qa.guard import Language, route_conversation
from docinsight.qa.prompts import HistoryTurn
from docinsight.qa.rag import NoDocumentsError, answer_question
from docinsight.schemas import AskRequest, AskResponse, EntityOut, SourceOut
from docinsight.storage.models import Document

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ask"])

SNIPPET_CHARS = 300


@router.post(
    "/ask",
    response_model=AskResponse,
    # Every question counts, also those the router answers without the LLM.
    dependencies=[Depends(RateLimit("llm"))],
    responses={
        # Only on the "rag" route: the other routes need neither documents nor an LLM.
        status.HTTP_409_CONFLICT: {"description": "No processed documents to search."},
        status.HTTP_429_TOO_MANY_REQUESTS: {"description": "Rate limit reached; see Retry-After."},
        status.HTTP_502_BAD_GATEWAY: {"description": "The embedding or LLM call failed."},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "No LLM is configured."},
    },
)
def ask(
    body: AskRequest,
    owner_id: Annotated[str, Depends(require_session_id)],
    db: DbSession,
    embedder: EmbedderDep,
    generator: GeneratorDep,
    query_router: RouterDep,
    entity_extractor: EntityExtractorDep,
    settings: SettingsDep,
) -> AskResponse:
    """Answer a question from the caller's documents only, citing file and page.

    A plain `def`: the OpenAI calls block, so FastAPI runs this in its threadpool.
    """
    history = [HistoryTurn(message.role, message.text) for message in body.history]
    decision = route_conversation(
        query_router, body.question, [turn.text for turn in history if turn.role == "user"]
    )
    # The route, the rule and the history length only: never the question or the history.
    logger.info(
        "Question routed to %s (rule %s, %d history messages)",
        decision.route,
        decision.rule_id,
        len(history),
    )
    match decision.route:
        case "rag":
            pass  # below
        case "meta":
            return _document_list(db, owner_id, decision.language)
        case canned_route:  # mypy narrows this to the canned routes
            return AskResponse(
                answer=canned_answer(canned_route, decision.language),
                found=False,
                sources=[],
                route=canned_route,
                model=None,
                timings_ms={},
            )

    try:
        result = answer_question(
            body.question,
            owner_id,
            db,
            embedder,
            generator,
            settings,
            body.document_ids,
            body.top_k,
            history,
        )
    except NoDocumentsError:
        raise HTTPException(status.HTTP_409_CONFLICT, "No processed documents to search.") from None
    except LlmUnavailableError:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "The answer service is not configured."
        ) from None
    except LlmRequestError:
        logger.warning("Answer service failed", exc_info=True)
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, "The answer service failed, try again."
        ) from None

    timings_ms = dict(result.timings_ms)
    entities: list[EntityOut] = []
    if result.found:
        # The language of the router, so the answer is analysed by the model of its language.
        started = time.perf_counter()
        entities = [
            EntityOut(text=e.text, label=e.label, start=e.start, end=e.end)
            for e in entity_extractor.extract(result.answer, decision.language)
        ]
        timings_ms["ner"] = round((time.perf_counter() - started) * 1000, 1)

    return AskResponse(
        # The model's own "not found" text could suggest the question was valid but
        # unanswerable, or out of scope; one fixed text covers both honestly.
        answer=result.answer if result.found else not_found_answer(decision.language),
        found=result.found,
        sources=[
            SourceOut(
                document_id=source.document_id,
                filename=source.filename,
                page_number=source.page_number,
                snippet=_snippet(source.text),
                score=round(source.score, 4),
            )
            for source in result.sources
        ],
        route="rag",
        model=result.model,
        timings_ms=timings_ms,
        entities=entities,
    )


def _document_list(db: DbSession, owner_id: str, language: Language) -> AskResponse:
    """The "meta" route: the caller's documents from the database, no search, no LLM."""
    rows = db.exec(
        select(Document.filename, Document.status)
        .where(Document.owner_id == owner_id)
        .order_by(col(Document.created_at))
    ).all()
    return AskResponse(
        answer=meta_answer([(filename, status) for filename, status in rows], language),
        found=True,
        sources=[],
        route="meta",
        model=None,
        timings_ms={},
    )


def _snippet(text: str) -> str:
    """The start of a chunk, cut at a word boundary."""
    if len(text) <= SNIPPET_CHARS:
        return text
    return text[:SNIPPET_CHARS].rsplit(maxsplit=1)[0] + "..."
