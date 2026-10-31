"""Tests for entity extraction: rules, merging with a model, and the real spaCy models."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest

from docinsight.config import Settings
from docinsight.nlp import ner
from docinsight.nlp.ner import (
    Entity,
    EntityExtractor,
    ModelDoc,
    ModelSpan,
    NerPipeline,
    RuleEntityExtractor,
    SpacyEntityExtractor,
    create_entity_extractor,
    load_spacy_pipelines,
    rule_entities,
)
from docinsight.qa.guard import Language

ANSWER = (
    "Ukupno 1.575,00 EUR, kupac Horvat d.o.o., OIB 99083863799, "
    "IBAN HR72 2360 0004 2751 7979 0, dospijeće 29.03.2026. [1]"
)


def labelled(entities: list[Entity]) -> list[tuple[str, str]]:
    return [(entity.text, entity.label) for entity in entities]


def assert_offsets(text: str, entities: list[Entity]) -> None:
    assert all(text[e.start : e.end] == e.text for e in entities)
    assert [e.start for e in entities] == sorted(e.start for e in entities)


# --- Fake spaCy pipeline --------------------------------------------------------


@dataclass(frozen=True)
class FakeSpan:
    label_: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class FakeDoc:
    ents: Sequence[ModelSpan]


@dataclass
class FakePipeline:
    """Tags every occurrence of the given phrases, like a model that always says the same."""

    tags: dict[str, str]
    calls: list[str] = field(default_factory=list)

    def __call__(self, text: str) -> ModelDoc:
        self.calls.append(text)
        spans = []
        for phrase, label in self.tags.items():
            start = text.find(phrase)
            if start >= 0:
                spans.append(FakeSpan(label, start, start + len(phrase)))
        return FakeDoc(sorted(spans, key=lambda span: span.start_char))


def spacy_with(tags: dict[str, str]) -> SpacyEntityExtractor:
    pipeline = FakePipeline(tags)
    return SpacyEntityExtractor({"hr": pipeline, "en": pipeline})


# --- Rules ----------------------------------------------------------------------


def test_rules_find_amount_oib_iban_and_date() -> None:
    entities = rule_entities(ANSWER)

    assert labelled(entities) == [
        ("1.575,00 EUR", "MONEY"),
        ("Horvat d.o.o.", "ORG"),
        ("99083863799", "OIB"),
        ("HR72 2360 0004 2751 7979 0", "IBAN"),
        ("29.03.2026.", "DATE"),
    ]
    assert_offsets(ANSWER, entities)


def test_rule_extractor_ignores_the_language() -> None:
    extractor: EntityExtractor = RuleEntityExtractor()

    assert extractor.extract(ANSWER, "hr") == extractor.extract(ANSWER, "en")
    assert extractor.name == "rules"


def test_citation_marker_is_not_an_entity() -> None:
    assert rule_entities("Iznos je naveden u izvoru [1] i [12].") == []


def test_no_entities_in_plain_text() -> None:
    assert rule_entities("U tvojim dokumentima nisam našao odgovor.") == []


# --- Merging rules with the model -----------------------------------------------


def test_model_names_are_added_to_rule_entities() -> None:
    extractor = spacy_with({"Horvat d.o.o.": "ORG", "Ana Kovač": "PER", "Zagreb": "LOC"})
    text = "Kupac Horvat d.o.o. iz Zagreba plaća 1.575,00 EUR, potpisala Ana Kovač."

    entities = extractor.extract(text, "hr")

    assert labelled(entities) == [
        ("Horvat d.o.o.", "ORG"),
        ("Zagreb", "LOC"),
        ("1.575,00 EUR", "MONEY"),
        ("Ana Kovač", "PERSON"),
    ]
    assert_offsets(text, entities)


@pytest.mark.parametrize(
    "tags",
    [
        # The model claims a longer span, a part of an amount, or exactly a rule's span.
        pytest.param({"OIB 99083863799": "ORG", "EUR": "ORG"}, id="longer-and-partial"),
        pytest.param({"99083863799": "ORG", "1.575,00 EUR": "ORG"}, id="same-span"),
    ],
)
def test_rule_wins_when_the_model_tags_the_same_text(tags: dict[str, str]) -> None:
    extractor = spacy_with(tags)

    entities = extractor.extract(ANSWER, "hr")

    assert ("99083863799", "OIB") in labelled(entities)
    assert ("1.575,00 EUR", "MONEY") in labelled(entities)
    # The only organisation is the company found by its legal form, not the model's spans.
    assert [e.text for e in entities if e.label == "ORG"] == ["Horvat d.o.o."]


@pytest.mark.parametrize(
    "tags",
    [
        pytest.param({"[1]": "MISC"}, id="citation"),
        pytest.param({"u [1]": "MISC"}, id="span-with-letters-over-citation"),
        pytest.param({"OIB": "ORG"}, id="abbreviation"),
        pytest.param({"2026-001": "MISC"}, id="no-letters"),
        pytest.param({"Horvat": "CARDINAL"}, id="dropped-label"),
        pytest.param({"Hrvati": "NORP"}, id="dropped-english-label"),
    ],
)
def test_model_noise_is_dropped(tags: dict[str, str]) -> None:
    text = "Račun 2026-001 izdao je Horvat, OIB naveden u [1]. Hrvati."

    assert spacy_with(tags).extract(text, "hr") == []


@pytest.mark.parametrize(
    ("model_label", "label"),
    [("PER", "PERSON"), ("DERIV_PER", "PERSON"), ("PERSON", "PERSON"), ("GPE", "LOC"),
     ("FAC", "LOC"), ("LOC", "LOC"), ("ORG", "ORG"), ("MISC", "MISC")],
)  # fmt: skip
def test_model_labels_are_mapped(model_label: str, label: str) -> None:
    assert labelled(spacy_with({"Kovač": model_label}).extract("Ana Kovač", "hr")) == [
        ("Kovač", label)
    ]


def test_pipeline_is_chosen_by_language() -> None:
    croatian, english = FakePipeline({}), FakePipeline({})
    extractor = SpacyEntityExtractor({"hr": croatian, "en": english})

    extractor.extract("Tko je kupac?", "hr")
    extractor.extract("Who is the buyer?", "en")

    assert (croatian.calls, english.calls) == (["Tko je kupac?"], ["Who is the buyer?"])


# --- Factory --------------------------------------------------------------------


def fake_loader() -> dict[Language, NerPipeline]:
    return {"hr": FakePipeline({}), "en": FakePipeline({})}


def missing_model() -> dict[Language, NerPipeline]:
    raise OSError("[E050] Can't find model 'hr_core_news_md'")


def test_rules_backend_does_not_load_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ner, "load_spacy_pipelines", missing_model)

    extractor = create_entity_extractor(Settings(ner_backend="rules"))

    assert isinstance(extractor, RuleEntityExtractor)


def test_auto_uses_spacy_when_the_models_load(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ner, "load_spacy_pipelines", fake_loader)

    extractor = create_entity_extractor(Settings(ner_backend="auto"))

    assert isinstance(extractor, SpacyEntityExtractor)


def test_auto_falls_back_to_rules_with_a_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(ner, "load_spacy_pipelines", missing_model)

    with caplog.at_level(logging.WARNING, logger="docinsight.nlp.ner"):
        extractor = create_entity_extractor(Settings(ner_backend="auto"))

    assert isinstance(extractor, RuleEntityExtractor)
    assert "using rules only" in caplog.text


def test_spacy_backend_fails_without_the_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ner, "load_spacy_pipelines", missing_model)

    with pytest.raises(OSError, match="E050"):
        create_entity_extractor(Settings(ner_backend="spacy"))


# --- Real spaCy models ----------------------------------------------------------


@pytest.fixture(scope="module")
def spacy_extractor() -> SpacyEntityExtractor:
    pytest.importorskip("hr_core_news_md")
    pytest.importorskip("en_core_web_sm")
    return SpacyEntityExtractor(load_spacy_pipelines())  # about a second, once per module


@pytest.mark.slow
def test_real_croatian_model(spacy_extractor: SpacyEntityExtractor) -> None:
    text = (
        "Ukupno 1.575,00 EUR, kupac je Horvat d.o.o. iz Zagreba, OIB 99083863799 [1]. "
        "Račun je izdala Ana Kovač [2]."
    )

    entities = spacy_extractor.extract(text, "hr")

    found = labelled(entities)
    assert ("Zagreba", "LOC") in found
    assert ("Ana Kovač", "PERSON") in found
    assert ("99083863799", "OIB") in found
    assert ("1.575,00 EUR", "MONEY") in found
    # The model tags "EUR" and "OIB" as ORG; both must be gone.
    assert not {"EUR", "OIB", "[1]", "[2]"} & {text for text, _ in found}
    assert_offsets(text, entities)


@pytest.mark.slow
def test_real_english_model(spacy_extractor: SpacyEntityExtractor) -> None:
    text = (
        "The total is € 1,575.00 [1]; the buyer is Horvat Ltd. from Zagreb, signed by Ivan Kovač."
    )

    entities = spacy_extractor.extract(text, "en")

    found = labelled(entities)
    assert ("€ 1,575.00", "MONEY") in found
    assert ("Horvat Ltd.", "ORG") in found
    assert ("Zagreb", "LOC") in found
    assert ("Ivan Kovač", "PERSON") in found
    assert_offsets(text, entities)
