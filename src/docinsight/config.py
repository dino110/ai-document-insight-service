"""Application settings loaded from environment variables."""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. Every field can be set as DOCINSIGHT_<FIELD_NAME>."""

    model_config = SettingsConfigDict(env_prefix="DOCINSIGHT_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    max_upload_mb: int = Field(default=20, gt=0)
    max_files_per_request: int = Field(default=10, gt=0)

    ocr_enabled: bool = False
    ocr_languages: list[str] = ["hr", "en"]
    ocr_min_chars: int = Field(default=30, ge=0)
    ocr_dpi: int = Field(default=200, gt=0)

    # "auto": OpenAI when an API key is set, otherwise the local hashing embedder.
    embedding_backend: Literal["auto", "openai", "hashing"] = "auto"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int | None = Field(default=None, ge=64)  # None = model's full size
    chunk_size: int = Field(default=800, ge=200)
    chunk_overlap: int = Field(default=150, ge=0)
    top_k: int = Field(default=5, ge=1, le=20)

    openai_api_key: SecretStr | None = None
    llm_model: str = "gpt-6-luna"
    # Reasoning tokens are billed as output; "low" is enough to answer from a short context.
    llm_reasoning_effort: Literal["none", "low", "medium"] = "low"
    llm_base_url: str | None = None  # any OpenAI-compatible server, e.g. Ollama
    llm_timeout_s: float = Field(default=30, gt=0)

    @field_validator("openai_api_key", mode="before")
    @classmethod
    def _empty_key_is_none(cls, value: object) -> object:
        # `DOCINSIGHT_OPENAI_API_KEY=` (as in .env.example) means "no key", not an empty key.
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _overlap_smaller_than_chunk(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def database_url(self) -> str:
        return f"sqlite:///{self.data_dir / 'app.db'}"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """Return the settings, read from the environment once per process."""
    return Settings()
