"""Route a question before any search or LLM call, with local rules and no model.

Questions the service must not or need not send to the LLM (prompt injection, abuse,
requests for legal advice or code, small talk, questions about the document list) get a
canned answer or an answer from the database. Everything else goes to RAG.

The rules are deliberately narrow: when a rule is not sure, the question goes to RAG,
because refusing a real question about a document is worse than one API call.
"""

import re
from dataclasses import dataclass
from typing import Literal, Protocol

from docinsight.indexing.lexical import tokenize

Route = Literal[
    "rejected_injection",
    "rejected_abuse",
    "legal_advice",
    "out_of_scope",
    "smalltalk",
    "meta",
    "rag",
]
Language = Literal["hr", "en"]


@dataclass(frozen=True)
class RouteDecision:
    route: Route
    rule_id: str | None  # the rule that fired, for logs and tests; None for "rag"
    language: Language


class QueryRouter(Protocol):
    def route(self, question: str) -> RouteDecision: ...


# Patterns run on normalised text: lowercase words without diacritics, one space apart
# ("Ignoriraj sve prethodne upute!" -> "ignoriraj sve prethodne upute"), so a pattern
# never has to deal with punctuation, "č" vs "c" or upper case.
_INJECTION = [
    (
        "injection.ignore_instructions",
        r"\b(ignore|disregard|forget)\b.{0,40}\b(instructions?|rules|prompts?)\b",
    ),
    (
        "injection.ignoriraj_upute",
        r"\b(ignoriraj|zanemari|zaboravi)\w*\b.{0,40}\b(upute|uputu|pravila|instrukcije)\b",
    ),
    ("injection.system_prompt", r"\bsystem prompt\b|\bsistemsk\w* (prompt|upute|poruk\w*)\b"),
    (
        "injection.reveal",
        r"\b(reveal|print|show|otkrij|ispisi|pokazi)\b.{0,20}"
        r"\b(your|svoje|tvoje)\b.{0,10}\b(instructions|prompt|rules|upute|pravila)\b",
    ),
    (
        "injection.role_change",
        r"\b(you are now|pretend to be|ti si sada|pretvaraj se|jailbreak|developer mode)\b",
    ),
]

_ABUSE = [
    (
        "abuse.words",
        r"\b(fuck\w*|shit|bitch\w*|asshole\w*|idiot\w*|moron\w*|jeb\w*|kurac|kurc\w*"
        r"|picka|picku|picke|govn\w*|kreten\w*|debil\w*|budal\w*)\b",
    ),
]

# Legal advice needs both: someone asking what to do, and a legal subject. "Koji je sud
# nadležan prema ugovoru?" names a court but asks for a fact in the document: RAG.
_ADVICE = (
    r"\b(trebam li|trebao bih|trebala bih|mogu li|smijem li|isplati li se|vrijedi li"
    r"|sto da|kako da|sto mi je ciniti|savjetuj\w*|preporuci\w*"
    r"|should i|can i|could i|do i need|is it worth|how do i|how can i|advise|recommend)\b"
)
_LEGAL = (
    r"\b(tuz\w*|sud|suda|sudu|sudom|sudsk\w*|odvjetni\w*|ospor\w*|zalb\w*"
    r"|sue|suing|lawsuit|court|lawyer|attorney|litigat\w*|contest)\b"
)
_LEGAL_ADVICE = [
    ("legal.explicit", r"\b(pravn\w* savjet\w*|legal advice)\b"),
    ("legal.advice", rf"{_ADVICE}.*{_LEGAL}|{_LEGAL}.*{_ADVICE}"),
]

# Health advice works like legal advice: asking what to do about a medical subject, or
# asking for a cure. "Koji je doktor potpisao ugovor?" names a doctor but asks a fact: RAG.
_MEDICAL = (
    r"\b(bolest\w*|simptom\w*|lijek\w*|lijecnik\w*|doktor\w*|dijagnoz\w*|tablet[aeu]"
    r"|temperatur\w*|glavobolj\w*|gripa|gripu|gripe|prehlad\w*|terapij\w*"
    r"|illness|disease|symptoms?|medicine|medication|doctor|diagnos\w*|fever|headache"
    r"|flu|cold|therapy)\b"
)

# Requests that are not questions about the documents. Not the word "kod" alone for code:
# in Croatian it also means "at" ("koliko dugujem kod Hinića?").
_OUT_OF_SCOPE = [
    (
        "scope.code",
        r"\b(napisi|write|generiraj|generate|create)\b.{0,40}"
        r"\b(code|programski kod|funkcij\w*|function|skript\w*|script|sql|regex"
        r"|python|javascript|typescript|html|css)\b",
    ),
    (
        "scope.recipe",
        r"\b(recept\w*|recipes?)\b|\b(kako|how)\b.{0,15}\b(kuha\w*|ispec\w*|pece\w*|cook|bake)\b",
    ),
    (
        "scope.medical",
        rf"{_ADVICE}.*{_MEDICAL}|{_MEDICAL}.*{_ADVICE}"
        r"|\b(lijek\w* (za|protiv)|medicine for|cure for|treatment for)\b",
    ),
    # "sala/salu/sale" only: "reci mi saldo računa" is a question for RAG.
    ("scope.joke", r"\b(ispricaj|reci|tell)\b.{0,15}\b(vic\w*|sala|salu|sale|jokes?)\b"),
]

# The whole question must be small talk: "Bok, koliki je iznos?" is a question for RAG.
_SMALLTALK = [
    (
        "smalltalk.phrase",
        r"(bok|pozdrav|hej|cao|dobar dan|dobro jutro|dobra vecer|hvala( ti| vam| puno)?"
        r"|tko si( ti)?|sto si( ti)?|kako si|hi|hello|hey|thanks( a lot)?|thank you"
        r"|who are you|what are you|how are you)",
    ),
]

# Questions about the document list, not their content: they ask what the user has
# ("imam", "my", "uploaded"). "Koji dokument sadrži OIB …?" asks about content: RAG.
_META = [
    (
        "meta.documents",
        r"\b(koj\w*|koliko|popis\w*|lista|pokazi|prikazi|which|what|how many|list|show)\b"
        r".{0,30}\b(dokumen\w*|datotek\w*|documents|files)\b",  # "dokumenata": not "dokument"
    ),
]
_META_OWNER = (
    r"\b(imam|moj\w*|uploada\w*|have|my|uploaded"
    r"|sam\b.{0,20}\b(ucitao|ucitala|poslao|poslala|dodao|dodala))\b"
)

# One string reads better than a list of single words. Only words that are not English
# too ("me" is left out: "tell me").
_HR_WORDS = frozenset(
    "je li koji koja koje koliki kolika sto kako koliko tko kada kad gdje "  # noqa: SIM905
    "ti si mi za da ne se su sam nisam nije imam imas trebam treba mogu moze "
    "daj molim jesi ovo ovaj ova ili kao bi bih "
    "racun racuna ugovor ugovora iznos bok hvala dokument dokumente napisi".split()
)


def _compile(rules: list[tuple[str, str]]) -> list[tuple[str, re.Pattern[str]]]:
    return [(rule_id, re.compile(pattern)) for rule_id, pattern in rules]


class RuleBasedRouter:
    """Satisfies QueryRouter: checks the rule groups in order, the first match wins."""

    def __init__(self) -> None:
        self._groups: list[tuple[Route, list[tuple[str, re.Pattern[str]]]]] = [
            ("rejected_injection", _compile(_INJECTION)),
            ("rejected_abuse", _compile(_ABUSE)),
            ("legal_advice", _compile(_LEGAL_ADVICE)),
            ("out_of_scope", _compile(_OUT_OF_SCOPE)),
        ]
        self._smalltalk = _compile(_SMALLTALK)
        self._meta = _compile(_META)
        self._meta_owner = re.compile(_META_OWNER)

    def route(self, question: str) -> RouteDecision:
        text = normalize(question)
        language = detect_language(question)
        for route, rules in self._groups:
            for rule_id, pattern in rules:
                if pattern.search(text):
                    return RouteDecision(route, rule_id, language)
        for rule_id, pattern in self._smalltalk:
            if pattern.fullmatch(text):
                return RouteDecision("smalltalk", rule_id, language)
        for rule_id, pattern in self._meta:
            if pattern.search(text) and self._meta_owner.search(text):
                return RouteDecision("meta", rule_id, language)
        return RouteDecision("rag", None, language)


def normalize(question: str) -> str:
    """Words without case or diacritics, one space apart."""
    return " ".join(tokenize(question))


def detect_language(question: str) -> Language:
    """Croatian letters or common Croatian words -> "hr", otherwise "en"."""
    if any(char in "čćžšđ" for char in question.lower()):
        return "hr"
    return "hr" if _HR_WORDS.intersection(tokenize(question)) else "en"
