"""Tests for the pure helpers of the Streamlit page (the page itself is checked by hand)."""

import pytest
from api_client import AskResponse, Route
from app import answer_caption


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
