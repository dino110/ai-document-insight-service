"""Fixed answers for routes that never reach the LLM, in Croatian and English."""

from typing import Literal

from docinsight.qa.guard import Language

# Every route except "rag" (answered by the LLM) and "meta" (answered from the database).
CannedRoute = Literal[
    "rejected_injection", "rejected_abuse", "legal_advice", "out_of_scope", "smalltalk"
]

_ANSWERS: dict[CannedRoute, dict[Language, str]] = {
    "rejected_injection": {
        "hr": "Ne mogu mijenjati svoje upute. Mogu odgovoriti na pitanja o tvojim dokumentima.",
        "en": "I can't change my instructions. I can answer questions about your documents.",
    },
    "rejected_abuse": {
        "hr": "Rado pomažem, ali bez uvreda. Pitaj me nešto o svojim dokumentima.",
        "en": "Happy to help, but without insults. Ask me something about your documents.",
    },
    "legal_advice": {
        "hr": (
            "Ne dajem pravne savjete. Mogu ti reći što piše u tvojim dokumentima "
            "(rokovi, iznosi, nadležni sud), a za savjet se obrati odvjetniku."
        ),
        "en": (
            "I don't give legal advice. I can tell you what your documents say "
            "(deadlines, amounts, the competent court); for advice, please consult a lawyer."
        ),
    },
    "out_of_scope": {
        "hr": "Odgovaram samo na pitanja o dokumentima koje si uploadao.",
        "en": "I only answer questions about the documents you have uploaded.",
    },
    "smalltalk": {
        "hr": (
            "Bok! Odgovaram na pitanja o tvojim uploadanim dokumentima, "
            "npr. „Koliki je ukupni iznos računa 2026-001?”"
        ),
        "en": (
            "Hi! I answer questions about your uploaded documents, "
            'e.g. "What is the total amount of invoice 2026-001?"'
        ),
    },
}


def canned_answer(route: CannedRoute, language: Language) -> str:
    return _ANSWERS[route][language]


_NOT_FOUND: dict[Language, str] = {
    "hr": (
        "U tvojim dokumentima nisam našao odgovor na to pitanje. "
        "Odgovaram samo na pitanja o sadržaju uploadanih dokumenata."
    ),
    "en": (
        "I couldn't find the answer to that in your documents. "
        "I only answer questions about the content of your uploaded documents."
    ),
}


def not_found_answer(language: Language) -> str:
    """Replaces the LLM's text when it found no answer (route "rag", found=false).

    One text for both cases the LLM cannot tell apart reliably: a question about the
    documents they do not answer, and a general question ("Who is the president?").
    """
    return _NOT_FOUND[language]


_STATUS = {
    "hr": {"pending": "u obradi", "processed": "obrađen", "failed": "greška pri obradi"},
    "en": {"pending": "processing", "processed": "processed", "failed": "failed"},
}


def meta_answer(documents: list[tuple[str, str]], language: Language) -> str:
    """The user's documents as (filename, status), oldest first, in one sentence."""
    if not documents:
        if language == "hr":
            return "Još nemaš uploadanih dokumenata."
        return "You have not uploaded any documents yet."
    listed = ", ".join(
        f"{filename} ({_STATUS[language].get(status, status)})" for filename, status in documents
    )
    if language == "hr":
        return f"Imaš {len(documents)} {_dokument(len(documents))}: {listed}."
    noun = "document" if len(documents) == 1 else "documents"
    return f"You have {len(documents)} {noun}: {listed}."


def _dokument(n: int) -> str:
    """Croatian plural: 1 dokument, 2-4 dokumenta, 5+ dokumenata (but 11-14 dokumenata)."""
    if n % 10 == 1 and n % 100 != 11:
        return "dokument"
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return "dokumenta"
    return "dokumenata"
