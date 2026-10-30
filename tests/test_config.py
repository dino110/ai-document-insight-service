"""Tests for settings validation."""

import pytest
from pydantic import SecretStr, ValidationError

from docinsight.config import Settings


def test_chunk_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError, match="chunk_overlap"):
        Settings(chunk_size=300, chunk_overlap=300)


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_api_key_in_env_means_no_key(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("DOCINSIGHT_OPENAI_API_KEY", value)

    assert Settings().openai_api_key is None


def test_api_key_is_read_from_env_and_hidden_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCINSIGHT_OPENAI_API_KEY", "sk-test-123")

    settings = Settings()

    assert "sk-test-123" not in repr(settings)
    assert settings.openai_api_key == SecretStr("sk-test-123")
    assert settings.openai_api_key.get_secret_value() == "sk-test-123"
