"""Fix OCR errors that are certain: Croatian legal forms such as "d.0.0." for "d.o.o.".

Only legal forms are fixed. A general 0 -> o swap would break amounts, OIBs and IBANs, and
lost diacritics ("Nacinovic") cannot be restored without a dictionary.
"""

import re

_SEP = r"\s*[.,]\s*"  # OCR reads a dot as a comma and adds spaces around it
_O = "[oO0]"  # EasyOCR reads a small "o" as a zero

# Canonical form -> its OCR spellings. The order does not matter: "j.d.0.0" becomes
# "j.d.o.o." whether its "d.0.0" or the whole form is replaced first.
_FORMS = {
    "j.d.o.o.": rf"j{_SEP}d{_SEP}{_O}{_SEP}{_O}",
    "d.o.o.": rf"d{_SEP}{_O}{_SEP}{_O}",
    "j.t.d.": rf"j{_SEP}t{_SEP}d",
    "d.d.": rf"d{_SEP}d",
    "k.d.": rf"k{_SEP}d",
}
# Whole words only; the final dot is optional ("Horvat d.0.0 Kraljevecki").
_PATTERNS = [
    (re.compile(rf"(?<!\w){spelling}(?!\w)(?:\s*\.)?"), canonical)
    for canonical, spelling in _FORMS.items()
]


def clean_ocr_text(text: str) -> str:
    """Return OCR text with every legal form written canonically ("d.0.0" -> "d.o.o.")."""
    for pattern, canonical in _PATTERNS:
        text = pattern.sub(canonical, text)
    return text
