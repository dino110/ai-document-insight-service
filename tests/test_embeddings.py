"""Tests for the embedders. OpenAI is replaced by httpx2.MockTransport, never called for real."""

import json
from collections.abc import Callable

import httpx2
import numpy as np
import pytest
from pydantic import SecretStr

from docinsight.config import Settings
from docinsight.indexing.embeddings import (
    EmbeddingError,
    HashingEmbedder,
    OpenAiEmbedder,
    create_embedder,
)

Handler = Callable[[httpx2.Request], httpx2.Response]
DIM = 64


# --- HashingEmbedder ---


def test_hashing_shapes_dtype_and_unit_norm() -> None:
    embedder = HashingEmbedder(dim=DIM)

    documents = embedder.embed_documents(["Račun br. 1", "Ugovor o djelu"])
    query = embedder.embed_query("račun")

    assert documents.shape == (2, DIM)
    assert query.shape == (DIM,)
    assert documents.dtype == query.dtype == np.float32
    assert np.allclose(np.linalg.norm(documents, axis=1), 1.0)


def test_hashing_ignores_case_and_diacritics() -> None:
    embedder = HashingEmbedder()

    typed = embedder.embed_query("Račun dospijeća, Đakovo")
    from_ocr = embedder.embed_query("RACUN DOSPIJECA, DAKOVO")

    assert np.array_equal(typed, from_ocr)


def test_hashing_is_deterministic_across_instances() -> None:
    assert np.array_equal(
        HashingEmbedder().embed_query("Ukupno za platiti"),
        HashingEmbedder().embed_query("Ukupno za platiti"),
    )


def test_hashing_prefers_text_with_shared_words() -> None:
    embedder = HashingEmbedder()
    query = embedder.embed_query("Koji je datum dospijeća računa?")
    related, unrelated = embedder.embed_documents(
        ["Datum dospijeća: 15.11.2026.", "Naručitelj: Horvat d.o.o., Zagreb"]
    )

    assert query @ related > query @ unrelated


def test_hashing_empty_text_is_zero_vector() -> None:
    assert not HashingEmbedder().embed_query("  ").any()


# --- create_embedder ---


def test_auto_without_key_uses_hashing() -> None:
    assert isinstance(create_embedder(Settings()), HashingEmbedder)


def test_auto_with_key_uses_openai() -> None:
    embedder = create_embedder(Settings(openai_api_key=SecretStr("sk-test")))

    assert isinstance(embedder, OpenAiEmbedder)
    assert (embedder.name, embedder.dim) == ("text-embedding-3-small", 1536)


def test_explicit_hashing_ignores_key() -> None:
    settings = Settings(openai_api_key=SecretStr("sk-test"), embedding_backend="hashing")

    assert isinstance(create_embedder(settings), HashingEmbedder)


def test_explicit_openai_without_key_fails_at_startup() -> None:
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        create_embedder(Settings(embedding_backend="openai"))


# --- OpenAiEmbedder ---


def _openai_embedder(handler: Handler) -> OpenAiEmbedder:
    settings = Settings(openai_api_key=SecretStr("sk-test"), embedding_dimensions=DIM)
    http_client = httpx2.Client(transport=httpx2.MockTransport(handler))
    return OpenAiEmbedder(settings, http_client=http_client)


def _fake_embeddings(request: httpx2.Request) -> httpx2.Response:
    """Vector for input i points along axis i % DIM (length 3, so normalisation shows).

    Items come back in reverse order: the embedder must sort them by `index`.
    """
    body = json.loads(request.content)
    data = []
    for i, _ in enumerate(body["input"]):
        vector = [0.0] * body["dimensions"]
        vector[i % body["dimensions"]] = 3.0
        data.append({"object": "embedding", "index": i, "embedding": vector})
    return httpx2.Response(
        200,
        json={
            "object": "list",
            "data": data[::-1],
            "model": body["model"],
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        },
    )


def test_openai_batches_orders_and_normalises() -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return _fake_embeddings(request)

    embedder = _openai_embedder(handler)
    vectors = embedder.embed_documents([f"chunk {i}" for i in range(250)])

    bodies = [json.loads(request.content) for request in requests]
    assert [len(body["input"]) for body in bodies] == [100, 100, 50]
    assert bodies[0]["model"] == "text-embedding-3-small"
    assert bodies[0]["dimensions"] == DIM
    assert requests[0].headers["authorization"] == "Bearer sk-test"

    assert vectors.shape == (250, DIM)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    # Row order follows the input order, not the order of the response items.
    assert list(vectors[:100].argmax(axis=1)) == [i % DIM for i in range(100)]
    assert embedder.name == f"text-embedding-3-small@{DIM}"


def test_openai_query_is_one_vector() -> None:
    assert _openai_embedder(_fake_embeddings).embed_query("pitanje").shape == (DIM,)


def test_openai_server_error_is_retried_then_becomes_embedding_error() -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(500, json={"error": {"message": "boom"}})

    # The SDK retries 5xx with a backoff (about 1 s here), then gives up.
    with pytest.raises(EmbeddingError):
        _openai_embedder(handler).embed_documents(["tekst"])
    assert len(requests) == 3  # first try + max_retries=2


def test_openai_missing_items_become_embedding_error() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        response = _fake_embeddings(request)
        body = response.json()
        body["data"] = body["data"][:1]
        return httpx2.Response(200, json=body)

    with pytest.raises(EmbeddingError):
        _openai_embedder(handler).embed_documents(["a", "b"])


def test_openai_unknown_model_needs_dimensions() -> None:
    settings = Settings(openai_api_key=SecretStr("sk-test"), embedding_model="some-new-model")

    with pytest.raises(ValueError, match="EMBEDDING_DIMENSIONS"):
        OpenAiEmbedder(settings)
