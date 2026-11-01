"""Named entities in answers: rules for fixed formats, a spaCy model for names.

Rules (``patterns.py``) find OIBs, IBANs, amounts and dates with check digits, and
companies by their legal form ("d.o.o."); a statistical model would only guess them.
spaCy finds people, other organisations and places, which no rule can. When both mark
the same text, the rule wins: an OIB is never "ORG".
"""

import logging
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from typing import Literal, Protocol

from docinsight.config import Settings
from docinsight.indexing.lexical import tokenize
from docinsight.nlp.patterns import (
    Span,
    find_amounts,
    find_companies,
    find_dates,
    find_ibans,
    find_oibs,
)
from docinsight.qa.guard import Language

logger = logging.getLogger(__name__)

EntityLabel = Literal["PERSON", "ORG", "LOC", "MISC", "OIB", "IBAN", "DATE", "MONEY"]


@dataclass(frozen=True)
class Entity:
    text: str
    label: EntityLabel
    start: int  # character offsets in the analysed text: text[start:end] == self.text
    end: int


class EntityExtractor(Protocol):
    # Backend and versions, stored with every document entity: like the index name of
    # chunks, it lets reindex find entities produced by an older extractor.
    name: str

    def extract(self, text: str, language: Language) -> list[Entity]:
        """Entities sorted by ``start``, without overlaps."""
        ...


# Bump when a rule changes what it finds, so reindex recomputes the stored entities.
RULES_VERSION = 3

# Rules in priority order: the first one to claim a span keeps it.
_RULES: list[tuple[EntityLabel, Callable[[str], list[Span]]]] = [
    ("IBAN", find_ibans),
    ("OIB", find_oibs),
    ("MONEY", find_amounts),
    ("DATE", find_dates),
    # A legal form makes a name a company; the models often call "Horvat d.o.o." a person.
    ("ORG", find_companies),
]

# Model labels we keep: people and places only. Measured on the sample documents (rules-1),
# the model's ORG was wrong 4 times out of 4 ("Naručitelja", "Mirogojska 1c") and MISC
# mostly wrong; companies come from the legal-form rule instead. Croatian: PER, DERIV_PER
# (adjectives from names, "Kovačev"), LOC. English: PERSON, GPE (countries, cities), LOC,
# FAC (buildings). Everything else is dropped; MISC stays an API label only because
# entities stored by older extractors may have it.
_MODEL_LABELS: dict[str, EntityLabel] = {
    "PER": "PERSON",
    "DERIV_PER": "PERSON",
    "PERSON": "PERSON",
    "LOC": "LOC",
    "GPE": "LOC",
    "FAC": "LOC",
}

# Words naming a role in a contract or invoice, in any case form ("Naručitelja",
# "Izvršitelju"): the models tag them as people, places or organisations.
_ROLE = re.compile(
    r"\b(naru[cč]itelj|izvr[sš]itelj|izdavatelj|prodavatelj|dobavlja[cč]|kupc|kupac"
    r"|client|contractor|buyer|seller|issuer|supplier)\w*",
    re.IGNORECASE,
)

# Citation markers such as "[1]" point to sources; they are never entities.
_CITATION = re.compile(r"\[\d+\]")


def rule_entities(text: str) -> list[Entity]:
    """OIBs, IBANs, amounts and dates, sorted by position."""
    candidates = [
        Entity(text[start:end], label, start, end)
        for label, find in _RULES
        for start, end in find(text)
    ]
    return _sorted(_without_overlaps(candidates, kept=[]))


class RuleEntityExtractor:
    """Satisfies EntityExtractor with rules only: no model, same result in every language."""

    name = f"rules-{RULES_VERSION}"

    def extract(self, text: str, language: Language) -> list[Entity]:
        return rule_entities(text)


# The small part of spaCy we use, as protocols: the extractor only needs a callable that
# returns something with ``.ents``, so tests can pass a fake pipeline without a model.
class ModelSpan(Protocol):
    @property
    def label_(self) -> str: ...
    @property
    def start_char(self) -> int: ...
    @property
    def end_char(self) -> int: ...


class ModelDoc(Protocol):
    @property
    def ents(self) -> Sequence[ModelSpan]: ...


class NerPipeline(Protocol):
    def __call__(self, text: str) -> ModelDoc: ...


class SpacyEntityExtractor:
    """Satisfies EntityExtractor: rules plus a spaCy pipeline chosen by the question's language."""

    def __init__(
        self, pipelines: Mapping[Language, NerPipeline], name: str = f"spacy+rules-{RULES_VERSION}"
    ) -> None:
        self._pipelines = pipelines
        self.name = name
        # The pipelines are shared by the threadpool and spaCy does not promise that one
        # pipeline can run in several threads at once; a call takes milliseconds.
        self._lock = threading.Lock()

    def extract(self, text: str, language: Language) -> list[Entity]:
        rules = rule_entities(text)
        citations = [match.span() for match in _CITATION.finditer(text)]
        with self._lock:
            doc = self._pipelines[language](text)
        model = [
            Entity(text[ent.start_char : ent.end_char], label, ent.start_char, ent.end_char)
            for ent in doc.ents
            if (label := _MODEL_LABELS.get(ent.label_)) is not None
        ]
        useful = [
            entity
            for entity in model
            # A name has letters; "2026-001" or a stray number tagged by the model is noise.
            if any(char.isalpha() for char in entity.text)
            # A person has a first and a last name: one word was a table header ("Kol."),
            # a town ("Popovača") or a street ("Gupčeva") in every measured case. A name
            # has no digits: "Demetrova 6c" is a street with a house number.
            and (entity.label != "PERSON" or _is_person_name(entity.text))
            and not _ROLE.search(entity.text)
            and not any(_overlaps((entity.start, entity.end), span) for span in citations)
        ]
        return _sorted(_without_overlaps(useful, kept=rules))


# Only the components NER needs; ``ner`` reads the vectors of ``tok2vec``. Excluded
# components are not even loaded (``disable`` would load them and switch them off).
SPACY_MODELS: dict[Language, str] = {"hr": "hr_core_news_md", "en": "en_core_web_sm"}
_UNUSED_COMPONENTS = ["tagger", "morphologizer", "parser", "lemmatizer", "attribute_ruler"]


def load_spacy_pipelines() -> dict[Language, NerPipeline]:
    """Load both models; raises ImportError or OSError when spaCy or a model is missing."""
    import spacy  # imported here: takes about a second and is not needed for rules

    return {
        language: spacy.load(model, exclude=_UNUSED_COMPONENTS)
        for language, model in SPACY_MODELS.items()
    }


def create_entity_extractor(settings: Settings) -> EntityExtractor:
    """spaCy plus rules when the models load (``auto`` falls back to rules), else rules."""
    if settings.ner_backend == "rules":
        logger.info("NER: rules only (DOCINSIGHT_NER_BACKEND=rules)")
        return RuleEntityExtractor()
    try:
        pipelines = load_spacy_pipelines()
    except (ImportError, OSError) as error:  # spacy.load raises OSError for a missing model
        if settings.ner_backend == "spacy":
            raise
        logger.warning("NER: spaCy models not available (%s), using rules only", error)
        return RuleEntityExtractor()
    # "spacy-hr_core_news_md-3.8.0-en_core_web_sm-3.8.0+rules-1": a new model version
    # (or rule version) is a different extractor.
    models = "-".join(f"{model}-{version(model)}" for model in SPACY_MODELS.values())
    extractor = SpacyEntityExtractor(pipelines, name=f"spacy-{models}+rules-{RULES_VERSION}")
    logger.info("NER: %s", extractor.name)
    return extractor


# Frequent words that are not names, after tokenize() (lowercase, no diacritics).
_HR_WORDS = frozenset(
    "i u na za da se je su od do s sa o po iz ili te kao koji koja koje sto nije "  # noqa: SIM905
    "ovaj ovog ovoga ovim taj biti bit ce clanak ugovor ugovora racun racuna iznos "
    "datum kupac izdavatelj narucitelj izvrsitelj ukupno osnovica placanje".split()
)
_EN_WORDS = frozenset(
    "the and of to in for is are be by with on as at an or this that from shall "  # noqa: SIM905
    "will any its it not agreement contract invoice article amount date total client".split()
)
_LANGUAGE_SAMPLE_CHARS = 5000


def detect_document_language(text: str) -> Language:
    """The language with more of its frequent words in the start of the text; ties are "hr".

    Unlike detect_language for questions, letters such as "č" do not count: an English
    contract between Croatian parties is full of them (Šalić, Madunić).
    """
    tokens = tokenize(text[:_LANGUAGE_SAMPLE_CHARS])
    croatian = sum(token in _HR_WORDS for token in tokens)
    english = sum(token in _EN_WORDS for token in tokens)
    return "en" if english > croatian else "hr"


def _is_person_name(text: str) -> bool:
    words = text.split()
    return len(words) >= 2 and not any(char.isdigit() for char in text)


def _overlaps(a: Span, b: Span) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _without_overlaps(candidates: list[Entity], kept: list[Entity]) -> list[Entity]:
    """``kept`` plus every candidate, in order, that overlaps nothing kept so far."""
    result = list(kept)
    for entity in candidates:
        if not any(_overlaps((entity.start, entity.end), (k.start, k.end)) for k in result):
            result.append(entity)
    return result


def _sorted(entities: list[Entity]) -> list[Entity]:
    return sorted(entities, key=lambda entity: entity.start)
