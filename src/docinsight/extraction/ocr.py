"""OCR behind a protocol, so tests and machines without easyocr can swap the engine."""

import logging
import threading
from io import BytesIO
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from docinsight.config import Settings

logger = logging.getLogger(__name__)


class OcrUnavailableError(RuntimeError):
    """Raised when a document needs OCR but no OCR engine is available."""


class OcrEngine(Protocol):
    def read(self, image: NDArray[np.uint8]) -> str:
        """Return the text found in an RGB image of shape (height, width, 3)."""
        ...


class EasyOcrEngine:
    """EasyOCR on CPU. Loading the models takes seconds, so create it once per process."""

    def __init__(self, languages: list[str]) -> None:
        # Imported here so this module can be imported where easyocr is not installed.
        import easyocr

        self._reader = easyocr.Reader(languages, gpu=False)
        # One Reader is shared by the threadpool; PyTorch inference is not guaranteed
        # to be thread-safe, so calls are serialised.
        self._lock = threading.Lock()

    def read(self, image: NDArray[np.uint8]) -> str:
        with self._lock:
            paragraphs: list[str] = self._reader.readtext(image, detail=0, paragraph=True)
        return "\n".join(paragraphs)


class UnavailableOcrEngine:
    """Placeholder used when OCR is disabled or easyocr is missing."""

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def read(self, image: NDArray[np.uint8]) -> str:
        raise OcrUnavailableError(self.reason)


def create_ocr_engine(settings: Settings) -> OcrEngine:
    """Return EasyOCR when it is enabled and installed, otherwise an unavailable engine."""
    if not settings.ocr_enabled:
        logger.info("OCR is disabled (DOCINSIGHT_OCR_ENABLED=false)")
        return UnavailableOcrEngine("OCR is disabled")
    try:
        engine = EasyOcrEngine(settings.ocr_languages)
    except ImportError:
        logger.warning("OCR is enabled but easyocr is not installed")
        return UnavailableOcrEngine("easyocr is not installed")
    logger.info("EasyOCR loaded for languages %s", settings.ocr_languages)
    return engine


def image_to_array(content: bytes) -> NDArray[np.uint8]:
    """Decode PNG/JPEG bytes into an RGB array of shape (height, width, 3)."""
    with Image.open(BytesIO(content)) as img:
        return np.asarray(img.convert("RGB"))
