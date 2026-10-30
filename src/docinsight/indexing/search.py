"""Find the chunks most similar to a query among one owner's processed documents.

Chunks in SQLite are the source of truth; the FAISS index is built in memory for every
search. At this scale (hundreds to a few thousand chunks per owner) that takes
milliseconds and there is no index file that could drift from the database.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sqlmodel import Session, col, select

from docinsight.storage.models import Document, DocumentChunk

try:
    import faiss

    HAS_FAISS = True
except ImportError:  # pragma: no cover - faiss-cpu has wheels for every platform we use
    HAS_FAISS = False


@dataclass(frozen=True)
class ScoredChunk:
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    text: str
    score: float  # cosine similarity, because all vectors are L2-normalised


def search_chunks(
    session: Session,
    owner_id: str,
    query_vector: NDArray[np.float32],
    embedding_model: str,
    top_k: int,
    document_ids: list[str] | None = None,
    *,
    use_faiss: bool = HAS_FAISS,
) -> list[ScoredChunk]:
    """Return up to ``top_k`` chunks, most similar first.

    Only chunks of this owner's ``processed`` documents embedded with ``embedding_model``
    are searched: vectors of other models are not comparable.
    """
    statement = (
        select(DocumentChunk, Document.filename)
        .join(Document, col(DocumentChunk.document_id) == col(Document.id))
        .where(
            DocumentChunk.owner_id == owner_id,
            DocumentChunk.embedding_model == embedding_model,
            Document.status == "processed",
        )
    )
    if document_ids is not None:
        statement = statement.where(col(DocumentChunk.document_id).in_(document_ids))
    rows = session.exec(statement).all()
    if not rows:
        return []

    matrix = np.vstack([np.frombuffer(chunk.embedding, dtype=np.float32) for chunk, _ in rows])
    query = np.ascontiguousarray(query_vector, dtype=np.float32).reshape(1, -1)
    k = min(top_k, len(rows))
    hits = _faiss_top_k(matrix, query, k) if use_faiss else _numpy_top_k(matrix, query, k)

    results: list[ScoredChunk] = []
    for position, score in hits:
        chunk, filename = rows[position]
        results.append(
            ScoredChunk(
                document_id=chunk.document_id,
                filename=filename,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                score=score,
            )
        )
    return results


def _faiss_top_k(
    matrix: NDArray[np.float32], query: NDArray[np.float32], k: int
) -> list[tuple[int, float]]:
    # "Flat" = exact search over every vector; "IP" = inner (dot) product.
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    scores, positions = index.search(query, k)
    return [(int(p), float(s)) for p, s in zip(positions[0], scores[0], strict=True)]


def _numpy_top_k(
    matrix: NDArray[np.float32], query: NDArray[np.float32], k: int
) -> list[tuple[int, float]]:
    """The same exact search without faiss: one matrix-vector product and a sort."""
    scores = matrix @ query[0]
    positions = np.argsort(-scores, kind="stable")[:k]
    return [(int(p), float(scores[p])) for p in positions]
