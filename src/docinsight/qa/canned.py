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
