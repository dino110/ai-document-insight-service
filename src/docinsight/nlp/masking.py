"""Keep OIBs and IBANs inside the service: mask them before any call to OpenAI.

Embedded text gets generic placeholders ("[OIB]"): a vector never has to be mapped back.
Text for the LLM gets numbered ones ("[OIB_1]"), one IdentifierMasker per request, so the
answer can be restored with the real values. Only identifiers with check digits are
masked: the rules for them are reliable, unlike the model for names.
"""

import re

from docinsight.nlp.patterns import Span, find_ibans, find_oibs, normalize_entity

_PLACEHOLDER = re.compile(r"\[(?:OIB|IBAN)_\d+\]")


def _identifier_spans(text: str) -> list[tuple[str, Span]]:
    """IBANs and OIBs, sorted by position; they never overlap (an OIB needs non-digits
    around it, and the digits of an IBAN are preceded by "HR" and its check digits)."""
    spans = [("IBAN", span) for span in find_ibans(text)]
    spans += [("OIB", span) for span in find_oibs(text)]
    return sorted(spans, key=lambda item: item[1])


def _replace(text: str, spans: list[tuple[str, Span]], placeholder: list[str]) -> str:
    parts: list[str] = []
    position = 0
    for (_, (start, end)), replacement in zip(spans, placeholder, strict=True):
        parts.append(text[position:start])
        parts.append(replacement)
        position = end
    parts.append(text[position:])
    return "".join(parts)


def mask_identifiers(text: str) -> str:
    """For embeddings: every OIB becomes "[OIB]" and every IBAN "[IBAN]"."""
    spans = _identifier_spans(text)
    return _replace(text, spans, [f"[{label}]" for label, _ in spans])


class IdentifierMasker:
    """Numbered placeholders for one request: the same value always gets the same one.

    An IBAN in groups and the same IBAN written compactly are one value. ``unmask``
    restores the spelling seen first and leaves unknown placeholders as they are.
    """

    def __init__(self) -> None:
        self._placeholders: dict[tuple[str, str], str] = {}  # (label, normalized) -> "[OIB_1]"
        self._values: dict[str, str] = {}  # "[OIB_1]" -> the original text
        self._counts: dict[str, int] = {}

    def mask(self, text: str) -> str:
        spans = _identifier_spans(text)
        return _replace(
            text, spans, [self._placeholder(label, text, span) for label, span in spans]
        )

    def unmask(self, text: str) -> str:
        return _PLACEHOLDER.sub(lambda match: self._values.get(match.group(), match.group()), text)

    def _placeholder(self, label: str, text: str, span: Span) -> str:
        value = text[span[0] : span[1]]
        key = (label, normalize_entity(label, value))
        if key not in self._placeholders:
            self._counts[label] = self._counts.get(label, 0) + 1
            placeholder = f"[{label}_{self._counts[label]}]"
            self._placeholders[key] = placeholder
            self._values[placeholder] = value
        return self._placeholders[key]
