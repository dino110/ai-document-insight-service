"""Tests for the pure helpers of the Streamlit page (the page itself is checked by hand)."""

import pytest
from api_client import AskResponse, Route
from app import HISTORY_ANSWER_CHARS, ChatMessage, answer_caption, conversation_history


def _response(route: Route, model: str | None, timings: dict[str, float]) -> AskResponse:
    return {
        "answer": "…",
        "found": route in ("rag", "meta"),
        "sources": [],
        "route": route,
        "model": model,
        "timings_ms": timings,
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
        ChatMessage(role="user", content=question, sources=[], caption="", error=False, route=None),
        ChatMessage(
            role="assistant", content=answer, sources=[], caption="", error=error, route=route
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
