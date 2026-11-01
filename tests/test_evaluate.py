"""Tests for the evaluation script and for the question set itself."""

import json
import time

import evaluate
import httpx2
import pytest
from evaluate import (
    SAMPLE_DIR,
    answer_matches,
    base_document,
    citation_matches,
    load_questions,
    normalize,
    percentile,
    run,
    send,
)
from factories import FakeGenerator
from fastapi.testclient import TestClient

from docinsight.extraction.pdf import extract_text_layer
from docinsight.qa.generator import LlmAnswer

QUESTIONS = load_questions(SAMPLE_DIR / "eval_questions.json")


def test_normalize_ignores_case_diacritics_and_whitespace() -> None:
    assert normalize("Fabijanić  Maloča D.D.") == normalize("fabijanic maloca d.d.")
    assert normalize("29. 10. 2026.") == "29.10.2026."


def test_answer_matches_any_accepted_spelling() -> None:
    assert answer_matches("Ukupno je 1.575,00 EUR [1].", ["1575.00", "1.575,00"])
    assert answer_matches("The total is 1,575.00 EUR.", ["1.575,00", "1,575.00"])
    assert not answer_matches("Ukupno je 1.260,00 EUR.", ["1.575,00", "1575.00"])


def test_scan_counts_as_its_original_for_citations() -> None:
    assert base_document("racun-2026-001-scan.jpg") == "racun-2026-001"
    assert citation_matches([("ugovor-01-scan.pdf", 2)], "ugovor-01.pdf", 2)
    assert not citation_matches([("ugovor-01.pdf", 1)], "ugovor-01.pdf", 2)
    assert not citation_matches([("racun-2026-002.pdf", 1)], "racun-2026-001.pdf", 1)


def test_percentile_nearest_rank() -> None:
    values = [float(v) for v in range(1, 101)]

    assert percentile(values, 50) == 50
    assert percentile(values, 95) == 95
    assert percentile([], 50) == 0.0


def test_question_set_shape() -> None:
    expected_docs = json.loads((SAMPLE_DIR / "expected.json").read_text(encoding="utf-8"))

    assert 16 <= len(QUESTIONS) <= 22
    assert len({q["id"] for q in QUESTIONS}) == len(QUESTIONS)
    unanswerable = [q for q in QUESTIONS if q["expected"] is None]
    assert 2 <= len(unanswerable) <= 3
    for q in QUESTIONS:
        if q["expected"] is not None:
            assert q["document"] in expected_docs
            assert q["page"] is not None


@pytest.mark.parametrize(
    "question", [q for q in QUESTIONS if q["expected"] is not None], ids=lambda q: q["id"]
)
def test_expected_answer_is_on_the_cited_page(question: dict[str, object]) -> None:
    """Guards the question set: a wrong amount or page here would look like a model error."""
    document, page, expected = question["document"], question["page"], question["expected"]
    assert isinstance(document, str) and isinstance(page, int) and isinstance(expected, list)
    pages = extract_text_layer((SAMPLE_DIR / document).read_bytes())

    assert answer_matches(pages[page - 1], expected)


def test_run_against_the_app_produces_a_report(
    client: TestClient, generator: FakeGenerator
) -> None:
    # The fake LLM always gives the same answer: the point is the flow, not the scores.
    generator.answer = LlmAnswer(answer="1.575,00 EUR [1]", found=True, citations=[1])

    text, results = run(client, QUESTIONS, label="test run", timeout_s=30)

    assert len(results) == len(QUESTIONS)
    assert {r.status_code for r in results} == {200}
    assert next(r for r in results if r.id == "inv001-total").answer_ok
    assert not next(r for r in results if r.id == "none-interest").answer_ok  # found=true
    assert "### Evaluation: test run" in text
    assert "| Answer accuracy | " in text
    assert "15 documents uploaded" in text  # more than one upload request allows


def test_follow_up_questions_depend_on_their_history() -> None:
    follow_ups = [q for q in QUESTIONS if "history" in q]

    assert len(follow_ups) == 2
    for q in follow_ups:
        [user, assistant] = q["history"]
        assert (user["role"], assistant["role"]) == ("user", "assistant")
        # The question alone does not name the document: only the history does.
        assert base_document(q["document"] or "") not in q["question"]


def _client(statuses: list[int]) -> httpx2.Client:
    """A client whose server answers with these statuses in turn (429s with Retry-After 7)."""
    replies = iter(statuses)

    def handler(request: httpx2.Request) -> httpx2.Response:
        status = next(replies)
        return httpx2.Response(status, headers={"Retry-After": "7"} if status == 429 else {})

    return httpx2.Client(transport=httpx2.MockTransport(handler), base_url="http://api")


def test_send_waits_out_a_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    client = _client([429, 429, 200])

    response = send(lambda: client.get("/ask"))

    assert response.status_code == 200
    assert sleeps == [7, 7]


def test_send_gives_up_after_the_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    client = _client([429] * 10)

    response = send(lambda: client.get("/ask"))

    assert response.status_code == 429
    assert len(sleeps) == evaluate.RATE_LIMIT_RETRIES
