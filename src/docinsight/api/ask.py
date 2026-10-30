"""Question answering over the caller's documents."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from docinsight.api.deps import (
    DbSession,
    EmbedderDep,
    GeneratorDep,
    SettingsDep,
    require_session_id,
)
from docinsight.qa.generator import LlmRequestError, LlmUnavailableError
from docinsight.qa.rag import NoDocumentsError, answer_question
from docinsight.schemas import AskRequest, AskResponse, SourceOut

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ask"])

SNIPPET_CHARS = 300


@router.post(
    "/ask",
    response_model=AskResponse,
    responses={
        status.HTTP_409_CONFLICT: {"description": "No processed documents to search."},
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
    settings: SettingsDep,
) -> AskResponse:
    """Answer a question from the caller's documents only, citing file and page.

    A plain `def`: the OpenAI calls block, so FastAPI runs this in its threadpool.
    """
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

    return AskResponse(
        answer=result.answer,
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
        model=result.model,
        timings_ms=result.timings_ms,
    )


def _snippet(text: str) -> str:
    """The start of a chunk, cut at a word boundary."""
    if len(text) <= SNIPPET_CHARS:
        return text
    return text[:SNIPPET_CHARS].rsplit(maxsplit=1)[0] + "..."
