"""Rate limits on /ask, /documents/{id}/fields and /upload."""

import logging

import pytest
from factories import make_text_pdf
from fastapi.testclient import TestClient
from httpx2 import Response

from docinsight.api.deps import get_rate_limiter
from docinsight.config import get_settings
from docinsight.core.rate_limit import InMemoryRateLimiter
from docinsight.main import app

# The router answers this without documents or an LLM, so only the limit is under test.
QUESTION = {"question": "Trebam li tužiti dobavljača?"}


@pytest.fixture
def limits(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """The monkeypatch for ``set_limits``, after ``client`` has started the app.

    conftest turns every limit off; each test turns on the ones it needs.
    """
    return monkeypatch


def set_limits(monkeypatch: pytest.MonkeyPatch, llm: int = 0, upload: int = 0, ip: int = 0) -> None:
    for name, value in (("LLM", llm), ("UPLOAD", upload), ("IP", ip)):
        monkeypatch.setenv(f"DOCINSIGHT_RATE_LIMIT_{name}_PER_MINUTE", str(value))
    get_settings.cache_clear()


def ask(client: TestClient, session: str) -> Response:
    return client.post("/ask", json=QUESTION, headers={"X-Session-ID": session})


def fields(client: TestClient, session: str) -> Response:
    return client.get("/documents/missing/fields", headers={"X-Session-ID": session})


def upload(client: TestClient, session: str | None) -> Response:
    headers = {"X-Session-ID": session} if session else {}
    pdf = ("files", ("a.pdf", make_text_pdf("A digital page with enough text on it."), "x"))
    return client.post("/upload", files=[pdf], headers=headers)


def test_the_request_over_the_session_limit_gets_429(
    client: TestClient, limits: pytest.MonkeyPatch
) -> None:
    set_limits(limits, llm=2)

    statuses = [ask(client, "s1").status_code for _ in range(3)]

    assert statuses == [200, 200, 429]
    response = ask(client, "s1")
    assert 1 <= int(response.headers["Retry-After"]) <= 60
    assert response.json()["detail"] == (
        f"Too many requests, try again in {response.headers['Retry-After']} s."
    )


def test_sessions_have_their_own_budgets(client: TestClient, limits: pytest.MonkeyPatch) -> None:
    set_limits(limits, llm=1)

    assert ask(client, "s1").status_code == 200
    assert ask(client, "s1").status_code == 429
    assert ask(client, "s2").status_code == 200


def test_ask_and_fields_share_the_llm_budget(
    client: TestClient, limits: pytest.MonkeyPatch
) -> None:
    set_limits(limits, llm=2)

    assert ask(client, "s1").status_code == 200
    assert fields(client, "s1").status_code == 404  # counted before the route runs
    assert fields(client, "s1").status_code == 429
    assert ask(client, "s1").status_code == 429


def test_uploads_have_their_own_budget(client: TestClient, limits: pytest.MonkeyPatch) -> None:
    set_limits(limits, llm=1, upload=1)
    assert ask(client, "s1").status_code == 200

    assert upload(client, "s1").status_code == 201
    assert upload(client, "s1").status_code == 429


def test_new_session_ids_do_not_escape_the_ip_limit(
    client: TestClient, limits: pytest.MonkeyPatch
) -> None:
    set_limits(limits, llm=100, upload=100, ip=3)

    statuses = [ask(client, f"s{i}").status_code for i in range(3)]
    statuses.append(upload(client, "s9").status_code)

    assert statuses == [200, 200, 200, 429]


def test_without_a_session_header_only_the_ip_counts(
    client: TestClient, limits: pytest.MonkeyPatch
) -> None:
    set_limits(limits, upload=1)

    # A new session is created for each upload, so the session limit cannot apply.
    assert [upload(client, None).status_code for _ in range(3)] == [201, 201, 201]


def test_zero_turns_the_limits_off(client: TestClient, limits: pytest.MonkeyPatch) -> None:
    set_limits(limits)

    assert {ask(client, "s1").status_code for _ in range(30)} == {200}


def test_reads_are_not_limited(client: TestClient, limits: pytest.MonkeyPatch) -> None:
    set_limits(limits, llm=1, upload=1, ip=1)
    assert ask(client, "s1").status_code == 200

    statuses = {
        client.get("/documents", headers={"X-Session-ID": "s1"}).status_code for _ in range(5)
    }

    assert statuses == {200}


def test_the_limit_resets_after_the_window(client: TestClient, limits: pytest.MonkeyPatch) -> None:
    now = [0.0]  # a list, so the clock below sees later changes
    limiter = InMemoryRateLimiter(window_s=60, clock=lambda: now[0])
    app.dependency_overrides[get_rate_limiter] = lambda: limiter
    set_limits(limits, llm=1)
    assert ask(client, "s1").status_code == 200
    now[0] = 30
    assert ask(client, "s1").headers["Retry-After"] == "30"

    now[0] = 60

    assert ask(client, "s1").status_code == 200


def test_log_names_the_budget_but_never_the_session(
    client: TestClient, limits: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    set_limits(limits, llm=1)
    ask(client, "secret-session-id")

    with caplog.at_level(logging.WARNING, logger="docinsight.api.deps"):
        ask(client, "secret-session-id")

    assert "Rate limit reached: llm budget, per session" in caplog.text
    assert "secret-session-id" not in caplog.text


def test_openapi_lists_429_on_the_limited_routes(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]

    for path, method in (
        ("/ask", "post"),
        ("/upload", "post"),
        ("/documents/{document_id}/fields", "get"),
    ):
        assert "429" in paths[path][method]["responses"], path
