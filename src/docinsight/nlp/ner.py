"""Named entities in answers: rules for fixed formats, a spaCy model for names.

Rules (``patterns.py``) find OIBs, IBANs, amounts and dates with check digits; a
statistical model would only guess them. spaCy finds people, organisations and places,
which no rule can. When both mark the same text, the rule wins: an OIB is never "ORG".
"""

import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from docinsight.config import Settings
from docinsight.nlp.patterns import Span, find_amounts, find_dates, find_ibans, find_oibs
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
    name: str  # for logs: which backend produced the entities

    def extract(self, text: str, language: Language) -> list[Entity]:
        """Entities sorted by ``start``, without overlaps."""
        ...


# Rules in priority order: the first one to claim a span keeps it.
_RULES: list[tuple[EntityLabel, Callable[[str], list[Span]]]] = [
    ("IBAN", find_ibans),
    ("OIB", find_oibs),
    ("MONEY", find_amounts),
    ("DATE", find_dates),
]

# Model labels we keep. Croatian: PER, DERIV_PER (adjectives from names, "Kovačev"), ORG,
# LOC, MISC. English: PERSON, ORG, GPE (countries, cities), LOC, FAC (buildings). English
# DATE, MONEY and CARDINAL are dropped (the rules cover them), and so is everything else.
_MODEL_LABELS: dict[str, EntityLabel] = {
    "PER": "PERSON",
    "DERIV_PER": "PERSON",
    "PERSON": "PERSON",
    "ORG": "ORG",
    "LOC": "LOC",
    "GPE": "LOC",
    "FAC": "LOC",
    "MISC": "MISC",
}

# Both models tag these abbreviations as ORG; in our documents they never are one.
_NOT_ENTITIES = frozenset({"OIB", "IBAN", "PDV", "VAT", "EUR", "HRK", "USD", "GBP"})

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

    name = "rules"

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

    name = "spacy"

    def __init__(self, pipelines: Mapping[Language, NerPipeline]) -> None:
        self._pipelines = pipelines

    def extract(self, text: str, language: Language) -> list[Entity]:
        rules = rule_entities(text)
        citations = [match.span() for match in _CITATION.finditer(text)]
        model = [
            Entity(text[ent.start_char : ent.end_char], label, ent.start_char, ent.end_char)
            for ent in self._pipelines[language](text).ents
            if (label := _MODEL_LABELS.get(ent.label_)) is not None
        ]
        useful = [
            entity
            for entity in model
            # A name has letters; "2026-001" or a stray number tagged by the model is noise.
            if any(char.isalpha() for char in entity.text)
            and entity.text not in _NOT_ENTITIES
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
    logger.info("NER: spaCy %s plus rules", ", ".join(SPACY_MODELS.values()))
    return SpacyEntityExtractor(pipelines)


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
