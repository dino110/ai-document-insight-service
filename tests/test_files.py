"""Tests for content-based file type detection."""

import pytest

from docinsight.core.files import DetectedType, detect_file_type


@pytest.mark.parametrize(
    ("head", "expected"),
    [
        (b"%PDF-1.7\n%\xe2\xe3\xcf\xd3", DetectedType("application/pdf", ".pdf")),
        (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR", DetectedType("image/png", ".png")),
        (b"\xff\xd8\xff\xe0\x00\x10JFIF", DetectedType("image/jpeg", ".jpg")),
        (b"Hello, this is plain text.", None),
        (b"", None),
        (b"%PD", None),  # truncated PDF signature
        (b"<html>%PDF-</html>", None),  # signature must be at the start
    ],
    ids=["pdf", "png", "jpeg", "text", "empty", "truncated-pdf", "pdf-not-at-start"],
)
def test_detect_file_type(head: bytes, expected: DetectedType | None) -> None:
    assert detect_file_type(head) == expected
