"""Text embeddings behind a protocol: OpenAI in production, a local hashing fallback."""

import hashlib
import logging
from typing import Protocol

import httpx2
import numpy as np
from numpy.typing import NDArray
from openai import OpenAI, OpenAIError, omit

from docinsight.config import Settings
from docinsight.indexing.lexical import tokenize

logger = logging.getLogger(__name__)

OPENAI_BATCH_SIZE = 100
# Full vector size of known OpenAI models, used when embedding_dimensions is not set.
OPENAI_MODEL_DIMENSIONS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}


class EmbeddingError(RuntimeError):
    """Raised when the embedding service fails or returns something unexpected."""


class Embedder(Protocol):
    # Stored with every chunk: vectors of different models (or sizes) are not comparable.
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> NDArray[np.float32]:
        """Return L2-normalised vectors of shape (len(texts), dim)."""
        ...

    def embed_query(self, text: str) -> NDArray[np.float32]:
        """Return one L2-normalised vector of shape (dim,)."""
        ...


class HashingEmbedder:
    """Lexical "embedding": each word is hashed to one of ``dim`` slots.

    No model and no network, so it is deterministic and free. Used in tests and as a
    fallback without an API key. It matches words, not meaning: no synonyms or paraphrases.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def embed_documents(self, texts: list[str]) -> NDArray[np.float32]:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                # blake2b, not hash(): Python's hash() of a str changes with every process
                # (PYTHONHASHSEED), so stored vectors would not match after a restart.
                value = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest())
                # A random sign per word makes hash collisions cancel out on average
                # instead of always adding up.
                sign = 1.0 if value & 1 else -1.0
                vectors[row, (value >> 1) % self.dim] += sign
        return _l2_normalize(vectors)

    def embed_query(self, text: str) -> NDArray[np.float32]:
        return self.embed_documents([text]).reshape(-1)  # (1, dim) -> (dim,)


class OpenAiEmbedder:
    """OpenAI embeddings API, in batches of ``OPENAI_BATCH_SIZE`` texts per request."""

    def __init__(self, settings: Settings, http_client: httpx2.Client | None = None) -> None:
        if settings.openai_api_key is None:
            raise ValueError("OpenAI embeddings need DOCINSIGHT_OPENAI_API_KEY")
        model = settings.embedding_model
        dim = settings.embedding_dimensions or OPENAI_MODEL_DIMENSIONS.get(model)
        if dim is None:
            raise ValueError(f"Unknown size of {model!r}, set DOCINSIGHT_EMBEDDING_DIMENSIONS")

        self._client = OpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            timeout=settings.llm_timeout_s,
            max_retries=2,
            http_client=http_client,
        )
        self._model = model
        self._dimensions = settings.embedding_dimensions
        self.dim = dim
        # A shortened vector is a different vector space than the full one.
        self.name = model if self._dimensions is None else f"{model}@{self._dimensions}"

    def embed_documents(self, texts: list[str]) -> NDArray[np.float32]:
        rows: list[list[float]] = []
        for start in range(0, len(texts), OPENAI_BATCH_SIZE):
            batch = texts[start : start + OPENAI_BATCH_SIZE]
            try:
                response = self._client.embeddings.create(
                    model=self._model,
                    input=batch,
                    dimensions=omit if self._dimensions is None else self._dimensions,
                    encoding_format="float",
                )
            except OpenAIError as exc:
                raise EmbeddingError("Embedding request failed") from exc
            # The API documents `index` as the input position; do not rely on list order.
            rows.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))

        vectors = np.asarray(rows, dtype=np.float32).reshape(-1, self.dim)
        if vectors.shape[0] != len(texts):
            raise EmbeddingError(f"Expected {len(texts)} embeddings, got {vectors.shape[0]}")
        return _l2_normalize(vectors)

    def embed_query(self, text: str) -> NDArray[np.float32]:
        return self.embed_documents([text]).reshape(-1)  # (1, dim) -> (dim,)


def create_embedder(settings: Settings) -> Embedder:
    """OpenAI when configured (or when ``auto`` finds a key), otherwise hashing."""
    backend = settings.embedding_backend
    if backend == "auto":
        backend = "openai" if settings.openai_api_key is not None else "hashing"
        if backend == "hashing":
            logger.warning("No OpenAI API key: falling back to lexical hashing embeddings")
    embedder: Embedder = OpenAiEmbedder(settings) if backend == "openai" else HashingEmbedder()
    logger.info("Embedder: %s %s", backend, embedder.name)
    return embedder


def _l2_normalize(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    """Scale every row to length 1, so a dot product is the cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # an empty text stays a zero vector instead of NaN
    return (vectors / norms).astype(np.float32)
