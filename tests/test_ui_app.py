"""Tests for the pure helpers of the Streamlit page (the page itself is checked by hand)."""

import pytest
from api_client import AskResponse, EntityLabel, EntityOut, Route
from app import (
    HISTORY_ANSWER_CHARS,
    ChatMessage,
    answer_caption,
    conversation_history,
    entity_legend,
    highlight,
)


def _response(route: Route, model: str | None, timings: dict[str, float]) -> AskResponse:
    return {
        "answer": "…",
        "found": route in ("rag", "meta"),
        "sources": [],
        "route": route,
        "model": model,
        "timings_ms": timings,
        "entities": [],
    }


def test_rag_caption_shows_model_and_timings() -> None:
    result = _response("rag", "gpt-6-luna", {"retrieval": 230.0, "generation": 1300.0})

    assert answer_caption(result) == "gpt-6-luna · pretraga 0.2 s · odgovor 1.3 s"


def test_document_list_caption() -> None:
    assert answer_caption(_response("meta", None, {})) == (
        "Popis iz baze · odgovoreno bez AI modela"
    )


@pytest.mark.parametrize(
    "route", ["rejected_injection", "rejected_abuse", "legal_advice", "out_of_scope", "smalltalk"]
)
def test_canned_caption_says_no_ai_model(route: Route) -> None:
    assert answer_caption(_response(route, None, {})) == (
        "Predefinirani odgovor · odgovoreno bez AI modela"
    )


def _pair(
    question: str, answer: str, route: str | None = "rag", error: bool = False
) -> list[ChatMessage]:
    return [
        ChatMessage(
            role="user",
            content=question,
            sources=[],
            caption="",
            error=False,
            route=None,
            entities=[],
        ),
        ChatMessage(
            role="assistant",
            content=answer,
            sources=[],
            caption="",
            error=error,
            route=route,
            entities=[],
        ),
    ]


def test_history_keeps_the_last_three_rag_pairs_oldest_first() -> None:
    messages = [
        *_pair("P1", "O1"),
        *_pair("P2", "O2"),
        *_pair("P3", "O3"),
        *_pair("P4", "O4"),
    ]

    history = conversation_history(messages)

    assert [m["text"] for m in history] == ["P2", "O2", "P3", "O3", "P4", "O4"]
    assert [m["role"] for m in history] == ["user", "assistant"] * 3


def test_history_skips_canned_answers_and_errors() -> None:
    messages = [
        *_pair("Koliki je iznos računa 2026-002?", "9.537,50 EUR [1]"),
        *_pair("Bok", "Bok! …", route="smalltalk"),
        *_pair("Koje dokumente imam?", "Imaš 2 dokumenta …", route="meta"),
        *_pair("A PDV?", "The answer service failed, try again.", route=None, error=True),
    ]

    assert [m["text"] for m in conversation_history(messages)] == [
        "Koliki je iznos računa 2026-002?",
        "9.537,50 EUR [1]",
    ]


def test_long_answers_are_cut_in_the_history() -> None:
    long_answer = "riječ " * 200

    [_, answer] = conversation_history(_pair("Pitanje?", long_answer))

    assert len(answer["text"]) <= HISTORY_ANSWER_CHARS + 2
    assert answer["text"].endswith(" …")


def test_empty_conversation_has_no_history() -> None:
    assert conversation_history([]) == []


def test_a_question_without_an_answer_does_not_shift_the_pairs() -> None:
    unanswered = _pair("Prekinuto?", "-")[:1]
    messages = [*_pair("P1", "O1"), *unanswered, *_pair("P2", "O2")]

    assert [m["text"] for m in conversation_history(messages)] == ["P1", "O1", "P2", "O2"]


# --- Entity highlighting --------------------------------------------------------


def entity(text: str, part: str, label: EntityLabel, occurrence: int = 0) -> EntityOut:
    start = -1
    for _ in range(occurrence + 1):
        start = text.index(part, start + 1)
    return {"text": part, "label": label, "start": start, "end": start + len(part)}


def marked(html: str) -> list[str]:
    """The texts inside <mark> tags, in order."""
    return [chunk.split(">", 1)[1] for chunk in html.split("<mark ")[1:]]


def test_entities_are_wrapped_in_marks_with_their_label() -> None:
    text = "Ukupno 1.575,00 EUR, kupac Horvat d.o.o. [1]"
    entities = [entity(text, "1.575,00 EUR", "MONEY"), entity(text, "Horvat d.o.o.", "ORG")]

    result = highlight(text, entities)

    assert [m.split("</mark>")[0] for m in marked(result)] == ["1.575,00 EUR", "Horvat d.o.o."]
    assert 'title="iznos"' in result
    assert 'title="organizacija"' in result
    assert result.startswith("Ukupno <mark ")
    assert result.endswith("</mark> [1]")


def test_html_in_the_text_is_escaped() -> None:
    text = '<script>alert("x")</script> Horvat d.o.o. <b>'
    result = highlight(text, [entity(text, "Horvat d.o.o.", "ORG")])

    assert "<script>" not in result
    assert "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;" in result
    assert result.endswith("</mark> &lt;b&gt;")
    # The only tags left are ours.
    assert result.count("<") == result.count("<mark ") + result.count("</mark>")


def test_html_inside_an_entity_is_escaped() -> None:
    text = "Tvrtka <i>Acme</i> Ltd. plaća"
    result = highlight(text, [entity(text, "<i>Acme</i> Ltd.", "ORG")])

    assert "&lt;i&gt;Acme&lt;/i&gt; Ltd.</mark>" in result


def test_entities_at_the_start_and_end_and_side_by_side() -> None:
    text = "29.03.2026.1.575,00 EUR"
    entities = [entity(text, "1.575,00 EUR", "MONEY"), entity(text, "29.03.2026.", "DATE")]

    result = highlight(text, entities)  # unsorted on purpose

    assert result.startswith("<mark ")
    assert result.endswith("</mark>")
    assert "</mark><mark " in result
    assert [m.split("</mark>")[0] for m in marked(result)] == ["29.03.2026.", "1.575,00 EUR"]


def test_dollar_signs_are_escaped_after_cutting() -> None:
    # "$" becomes "\$" only after the offsets were used, so the entity is still found.
    text = "Iznos $100 ili $200, kupac Acme Inc."
    result = highlight(text, [entity(text, "Acme Inc.", "ORG")])

    assert "Iznos \\$100 ili \\$200, kupac <mark " in result
    assert ">Acme Inc.</mark>" in result


def test_without_entities_the_text_is_only_escaped() -> None:
    text = "Odgovor s **naglaskom**, 5 < 6 i $10 [1]"

    assert highlight(text, []) == "Odgovor s **naglaskom**, 5 &lt; 6 i \\$10 [1]"


@pytest.mark.parametrize(
    "bad",
    [
        # After the first entity, but the text at the offsets is not the entity's text.
        pytest.param({"text": "Horvat", "label": "ORG", "start": 7, "end": 12}, id="stale-text"),
        # The right text at its offsets, but inside the first entity ("Ukupno").
        pytest.param({"text": "no", "label": "ORG", "start": 4, "end": 6}, id="overlap"),
    ],
)
def test_wrong_offsets_leave_the_text_unmarked(bad: EntityOut) -> None:
    text = "Ukupno kupac"
    first = entity(text, "Ukupno", "MISC")

    result = highlight(text, [first, bad])

    assert [m.split("</mark>")[0] for m in marked(result)] == ["Ukupno"]
    assert result.endswith("</mark> kupac")


def test_legend_lists_labels_present_in_a_fixed_order() -> None:
    text = "Acme Inc. 1.575,00 EUR Acme Inc."
    entities = [
        entity(text, "1.575,00 EUR", "MONEY"),
        entity(text, "Acme Inc.", "ORG"),
        entity(text, "Acme Inc.", "ORG", occurrence=1),
    ]

    assert [m.split("</mark>")[0] for m in marked(entity_legend(entities))] == [
        "organizacija",
        "iznos",
    ]
