"""Shared pytest fixtures."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from factories import FAKE_OCR_TEXT, FakeOcrEngine
from fastapi.testclient import TestClient

from docinsight.api.deps import get_ocr_engine
from docinsight.config import get_settings
from docinsight.main import app


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the app at a fresh, empty data directory for this test."""
    monkeypatch.setenv("DOCINSIGHT_DATA_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def fake_ocr() -> FakeOcrEngine:
    return FakeOcrEngine(FAKE_OCR_TEXT)


@pytest.fixture
def client(data_dir: Path, fake_ocr: FakeOcrEngine) -> Iterator[TestClient]:
    """A test client with lifespan running against an isolated data directory and fake OCR."""
    get_settings.cache_clear()
    app.dependency_overrides[get_ocr_engine] = lambda: fake_ocr
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    get_settings.cache_clear()
