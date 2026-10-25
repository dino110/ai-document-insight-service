"""Document upload and listing."""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status
from sqlmodel import col, select

from docinsight.api.deps import (
    DbEngine,
    DbSession,
    OcrDep,
    SettingsDep,
    get_or_create_session_id,
    require_session_id,
)
from docinsight.core.files import DetectedType, detect_file_type
from docinsight.processing import process_document
from docinsight.schemas import DocumentDetail, DocumentOut, PageOut, UploadResponse
from docinsight.storage.models import Document, DocumentPage

router = APIRouter(tags=["documents"])


@dataclass(frozen=True)
class _ValidatedFile:
    filename: str
    content: bytes
    detected: DetectedType


def _read_and_validate(upload: UploadFile, max_bytes: int) -> _ValidatedFile:
    # Endpoints are sync, so read the underlying file object directly instead of `await read()`.
    # Reading one byte past the limit is enough to know the file is too large.
    content = upload.file.read(max_bytes + 1)
    filename = Path(upload.filename or "unnamed").name

    if not content:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"File '{filename}' is empty.")
    if len(content) > max_bytes:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"File '{filename}' exceeds the {max_bytes // (1024 * 1024)} MB limit.",
        )
    detected = detect_file_type(content)
    if detected is None:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"File '{filename}' is not a PDF, PNG or JPEG.",
        )
    return _ValidatedFile(filename=filename, content=content, detected=detected)


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
)
def upload_documents(
    files: Annotated[list[UploadFile], File(description="PDF, PNG or JPEG files.")],
    owner_id: Annotated[str, Depends(get_or_create_session_id)],
    db: DbSession,
    engine: DbEngine,
    settings: SettingsDep,
    ocr: OcrDep,
    background_tasks: BackgroundTasks,
) -> UploadResponse:
    """Validate and store documents; text extraction runs after the response is sent."""
    if len(files) > settings.max_files_per_request:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"At most {settings.max_files_per_request} files per request.",
        )

    # Validate everything first so a bad file rejects the whole request before anything is saved.
    validated = [_read_and_validate(upload, settings.max_upload_bytes) for upload in files]

    owner_dir = settings.upload_dir / owner_id
    owner_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    documents: list[Document] = []
    try:
        for item in validated:
            document = Document(
                owner_id=owner_id,
                filename=item.filename,
                content_type=item.detected.content_type,
                size_bytes=len(item.content),
                sha256=hashlib.sha256(item.content).hexdigest(),
                stored_path="",
            )
            # The client's filename is never part of the path, which rules out path traversal.
            path = owner_dir / f"{document.id}{item.detected.extension}"
            path.write_bytes(item.content)
            written.append(path)
            document.stored_path = str(path)
            db.add(document)
            documents.append(document)
        db.commit()
    except Exception:
        db.rollback()
        for path in written:
            path.unlink(missing_ok=True)
        raise

    # Scheduled only after the commit, so the task is guaranteed to find the rows.
    for document in documents:
        background_tasks.add_task(process_document, document.id, engine, ocr, settings)

    return UploadResponse(
        session_id=owner_id,
        documents=[DocumentOut.model_validate(document) for document in documents],
    )


@router.get("/documents", response_model=list[DocumentOut])
def list_documents(
    owner_id: Annotated[str, Depends(require_session_id)],
    db: DbSession,
) -> list[DocumentOut]:
    """List the caller's documents, newest first."""
    statement = (
        select(Document)
        .where(Document.owner_id == owner_id)
        .order_by(col(Document.created_at).desc())
    )
    return [DocumentOut.model_validate(document) for document in db.exec(statement)]


@router.get(
    "/documents/{document_id}",
    response_model=DocumentDetail,
    responses={status.HTTP_404_NOT_FOUND: {"description": "No such document for this session."}},
)
def get_document(
    document_id: str,
    owner_id: Annotated[str, Depends(require_session_id)],
    db: DbSession,
) -> DocumentDetail:
    """Return one of the caller's documents with its extracted text per page."""
    document = db.get(Document, document_id)
    # Someone else's document is reported exactly like a missing one, so ids cannot be probed.
    if document is None or document.owner_id != owner_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found.")

    pages = db.exec(
        select(DocumentPage)
        .where(DocumentPage.document_id == document_id)
        .order_by(col(DocumentPage.page_number))
    )
    return DocumentDetail(
        **DocumentOut.model_validate(document).model_dump(),
        pages=[PageOut.model_validate(page) for page in pages],
    )
