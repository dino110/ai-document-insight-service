"""Type, summary and fields of one document as JSON, with the checks our code ran on them."""

import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlmodel import col, select

from docinsight.api.deps import DbSession, FieldExtractorDep, RateLimit, require_session_id
from docinsight.nlp.fields import FIELDS_FORMAT, check_document
from docinsight.qa.generator import LlmRequestError, LlmUnavailableError
from docinsight.schemas import FieldsOut
from docinsight.storage.models import Document, DocumentEntity, DocumentFields, DocumentPage

logger = logging.getLogger(__name__)

router = APIRouter(tags=["documents"])


@router.get(
    "/documents/{document_id}/fields",
    response_model=FieldsOut,
    # Also stored results count: the budget is per request, checked before the cache.
    dependencies=[Depends(RateLimit("llm"))],
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "No such document for this session."},
        status.HTTP_409_CONFLICT: {"description": "The document is not processed."},
        status.HTTP_429_TOO_MANY_REQUESTS: {"description": "Rate limit reached; see Retry-After."},
        status.HTTP_502_BAD_GATEWAY: {"description": "The LLM call failed."},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "No LLM is configured."},
    },
)
def get_fields(
    document_id: str,
    owner_id: Annotated[str, Depends(require_session_id)],
    db: DbSession,
    extractor: FieldExtractorDep,
) -> FieldsOut:
    """Type, summary, fields and checks; the first call asks the LLM, later calls read it.

    A GET although the first call stores something: the stored row is only a cache, every
    call returns the same fields for the same document. Re-extracting on purpose (and
    overwriting) would be a POST. A plain `def`: the OpenAI call blocks.
    """
    document = db.get(Document, document_id)
    # Someone else's document is reported exactly like a missing one, so ids cannot be probed.
    if document is None or document.owner_id != owner_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found.")
    stored = db.get(DocumentFields, document_id)
    if stored is not None and stored.data.get("format") == FIELDS_FORMAT:
        return _fields_out(stored, cached=True)
    if document.status != "processed":
        raise HTTPException(status.HTTP_409_CONFLICT, "The document is not processed.")

    pages = db.exec(
        select(DocumentPage)
        .where(DocumentPage.document_id == document_id)
        .order_by(col(DocumentPage.page_number))
    ).all()
    entities = db.exec(
        select(DocumentEntity.label, DocumentEntity.normalized).where(
            DocumentEntity.document_id == document_id
        )
    ).all()

    started = time.perf_counter()
    try:
        extraction = extractor.extract([(page.page_number, page.text) for page in pages])
    except LlmUnavailableError:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "The field extraction service is not configured."
        ) from None
    except LlmRequestError:
        logger.warning("Field extraction failed for document %s", document_id, exc_info=True)
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, "The field extraction service failed, try again."
        ) from None

    checks, warnings = check_document(
        extraction, "\n".join(page.text for page in pages), set(entities)
    )
    invoice, contract = extraction.invoice_fields(), extraction.contract_fields()
    row = DocumentFields(
        document_id=document_id,
        owner_id=owner_id,
        document_type=extraction.document_type,
        data={
            "format": FIELDS_FORMAT,
            "summary": extraction.summary,
            "invoice": invoice.model_dump() if invoice else None,
            "contract": contract.model_dump() if contract else None,
        },
        checks=checks,
        warnings=warnings,
        model=extractor.model,
    )
    if stored is not None:  # a result of an older format: replaced in the same transaction
        db.delete(stored)
        db.flush()  # the DELETE goes first, so the new row can take the same primary key
    db.add(row)
    try:
        db.commit()
    except IntegrityError:  # a parallel first call stored it a moment earlier
        db.rollback()
        stored = db.get(DocumentFields, document_id)
        if stored is None:
            raise
        return _fields_out(stored, cached=True)

    # Names of the failed checks and the timing only: field values are personal data.
    failed = [name for name, passed in checks.items() if passed is False]
    logger.info(
        "Fields of document %s: %s, failed checks %s, %.2f s",
        document_id,
        extraction.document_type,
        failed or "none",
        time.perf_counter() - started,
    )
    return _fields_out(row, cached=False)


def _fields_out(row: DocumentFields, *, cached: bool) -> FieldsOut:
    # model_validate checks the stored strings (document_type) against the schema.
    return FieldsOut.model_validate(
        {
            "document_id": row.document_id,
            "document_type": row.document_type,
            "summary": row.data["summary"],
            "invoice": row.data["invoice"],
            "contract": row.data["contract"],
            "checks": row.checks,
            "warnings": row.warnings,
            "model": row.model,
            "cached": cached,
        }
    )
