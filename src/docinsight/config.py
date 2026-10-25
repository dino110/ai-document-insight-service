"""Application settings loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
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
