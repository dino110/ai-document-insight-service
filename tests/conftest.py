"""Shared pytest fixtures."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from factories import FAKE_OCR_TEXT, FakeOcrEngine
from fastapi.testclient import TestClient

from docinsight.api.deps import get_embedder, get_ocr_engine
from docinsight.config import get_settings
from docinsight.indexing.embeddings import HashingEmbedder
from docinsight.main import app


@pytest.fixture(autouse=True)
def _no_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ignore the developer's real key from .env: tests never call OpenAI by accident.

    An environment variable wins over .env, and an empty key is read as "no key".
    """
    monkeypatch.setenv("DOCINSIGHT_OPENAI_API_KEY", "")


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the app at a fresh, empty data directory for this test."""
    monkeypatch.setenv("DOCINSIGHT_DATA_DIR", str(tmp_path))
    # API tests use FakeOcrEngine; never load real OCR models in lifespan, even in Docker.
    monkeypatch.setenv("DOCINSIGHT_OCR_ENABLED", "false")
    return tmp_path


@pytest.fixture
def fake_ocr() -> FakeOcrEngine:
    return FakeOcrEngine(FAKE_OCR_TEXT)


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture
def client(
    data_dir: Path, fake_ocr: FakeOcrEngine, embedder: HashingEmbedder
) -> Iterator[TestClient]:
    """A test client against an isolated data directory, with fake OCR and local embeddings."""
    get_settings.cache_clear()
    app.dependency_overrides[get_ocr_engine] = lambda: fake_ocr
    app.dependency_overrides[get_embedder] = lambda: embedder
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    get_settings.cache_clear()
