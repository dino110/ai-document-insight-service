"""Tests for the pure helpers of the Streamlit page (the page itself is checked by hand)."""

import pytest
from api_client import (
    ApiError,
    AskResponse,
    ContractFields,
    EntityLabel,
    EntityOut,
    InvoiceFields,
    Route,
)
from app import (
    HISTORY_ANSWER_CHARS,
    ChatMessage,
    answer_caption,
    check_rows,
    contract_rows,
    conversation_history,
    entity_legend,
    fields_error,
    highlight,
    invoice_rows,
    page_html,
    summary_rows,
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


# --- Page text and entity summary -----------------------------------------------


def test_page_html_keeps_line_breaks_and_escapes_the_document() -> None:
    text = "# Ugovor\n<script>x()</script>\nOIB: 99083863799\nCijena: $100"
    result = page_html(text, [entity(text, "99083863799", "OIB")])

    assert result.startswith('<div style="white-space: pre-wrap;')
    assert result.endswith("</div>")
    # Line breaks stay real line breaks; "# " stays text because st.html has no Markdown.
    assert "# Ugovor\n&lt;script&gt;x()&lt;/script&gt;\nOIB: <mark " in result
    assert ">99083863799</mark>\nCijena: $100</div>" in result  # no "\$" in plain HTML


def test_highlight_for_html_does_not_escape_dollars() -> None:
    assert highlight("$5 & $6", [], markdown=False) == "$5 &amp; $6"
    assert highlight("$5 & $6", []) == "\\$5 &amp; \\$6"


def test_summary_rows_use_label_names_in_the_api_order() -> None:
    summary: dict[EntityLabel, list[str]] = {
        "ORG": ["Car d.o.o.", "Horvat d.o.o."],
        "OIB": ["28196001332"],
        "MONEY": ["1575.00"],
    }

    assert summary_rows(summary) == [
        {"Oznaka": "organizacija", "Vrijednosti": "Car d.o.o., Horvat d.o.o."},
        {"Oznaka": "OIB", "Vrijednosti": "28196001332"},
        {"Oznaka": "iznos", "Vrijednosti": "1575.00"},
    ]


def test_summary_rows_of_a_document_without_entities() -> None:
    assert summary_rows({}) == []


# --- Document fields -----------------------------------------------------------

FIELDS: InvoiceFields = {
    "number": "2026-001",
    "issuer": "Car d.o.o.",
    "issuer_oib": "28196001332",
    "issuer_iban": "HR7223600004275179790",
    "buyer": "Horvat d.o.o.",
    "buyer_oib": "99083863799",
    "issue_date": "2026-02-27",
    "due_date": "2026-03-29",
    "currency": "EUR",
    "net": "1260.00",
    "vat": None,
    "total": "1575.00",
}


def test_invoice_rows_label_every_field_and_mark_missing_ones() -> None:
    rows = invoice_rows(FIELDS)

    assert len(rows) == 12
    assert rows[0] == {"Polje": "Broj računa", "Vrijednost": "2026-001"}
    assert {"Polje": "PDV", "Vrijednost": "-"} in rows


def test_contract_rows_label_every_field() -> None:
    contract: ContractFields = {
        "client": "Načinović d.o.o.",
        "client_oib": None,
        "contractor": "Elvis Antić",
        "contractor_oib": None,
        "signed_date": "2026-06-25",
        "start_date": "2026-07-01",
        "end_date": "2026-10-29",
        "fee": "6300.00",
        "currency": "EUR",
    }

    rows = contract_rows(contract)

    assert len(rows) == 9
    assert rows[0] == {"Polje": "Naručitelj", "Vrijednost": "Načinović d.o.o."}
    assert {"Polje": "Izvršitelj", "Vrijednost": "Elvis Antić"} in rows
    assert {"Polje": "OIB naručitelja", "Vrijednost": "-"} in rows


def test_check_rows_keep_the_api_order_and_show_pass_fail_and_not_checked() -> None:
    checks: dict[str, bool | None] = {
        "client_oib_valid": True,
        "dates_consistent": False,
        "names_in_entities": None,
        "a_new_check": True,  # from a newer API: shown by its name
    }

    assert check_rows(checks) == [
        {"Provjera": "Kontrolna znamenka OIB-a naručitelja", "Rezultat": "✓"},
        {"Provjera": "Datumi su u ispravnom redoslijedu", "Rezultat": "✗"},
        {"Provjera": "Strane su među imenima u dokumentu (NER)", "Rezultat": "- (nije provjereno)"},
        {"Provjera": "a_new_check", "Rezultat": "✓"},
    ]


@pytest.mark.parametrize(
    ("status", "message"),
    [
        (409, "Dokument još nije obrađen."),
        (502, "Izvlačenje podataka nije uspjelo, pokušaj ponovno."),
        (503, "Izvlačenje podataka nije dostupno: API nema OpenAI ključ."),
        (404, "Document not found."),  # anything else: the API's own message
        (None, "API nije dostupan"),
    ],
)
def test_fields_error_messages(status: int | None, message: str) -> None:
    detail = "Document not found." if status == 404 else "API nije dostupan"

    assert fields_error(ApiError(status, detail)) == message
