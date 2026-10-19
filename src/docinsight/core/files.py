"""File type detection from content (magic bytes)."""

from typing import NamedTuple


class DetectedType(NamedTuple):
    content_type: str
    extension: str


# The client controls both the filename and the Content-Type header, so neither is trusted.
_SIGNATURES: tuple[tuple[bytes, DetectedType], ...] = (
    (b"%PDF-", DetectedType("application/pdf", ".pdf")),
    (b"\x89PNG\r\n\x1a\n", DetectedType("image/png", ".png")),
    (b"\xff\xd8\xff", DetectedType("image/jpeg", ".jpg")),
)


def detect_file_type(head: bytes) -> DetectedType | None:
    """Return the type of a supported file from its first bytes, or None if unsupported."""
    for signature, detected in _SIGNATURES:
        if head.startswith(signature):
            return detected
    return None
