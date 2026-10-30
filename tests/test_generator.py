"""Tests for the LLM generator. OpenAI is replaced by httpx2.MockTransport."""

import json
from collections.abc import Callable
from typing import Any

import httpx2
import pytest
from pydantic import SecretStr

from docinsight.config import Settings
from docinsight.qa.generator import (
    LlmAnswer,
    LlmRequestError,
    LlmUnavailableError,
    OpenAiGenerator,
    UnavailableGenerator,
    create_generator,
)
from docinsight.qa.prompts import ContextChunk

Handler = Callable[[httpx2.Request], httpx2.Response]
CONTEXTS = [ContextChunk(1, "racun.pdf", 1, "Ukupno za platiti: 1.575,00 EUR")]
SETTINGS = Settings(openai_api_key=SecretStr("sk-test"))


def _generator(handler: Handler) -> OpenAiGenerator:
    http_client = httpx2.Client(transport=httpx2.MockTransport(handler))
    return OpenAiGenerator(SETTINGS, http_client=http_client)


# Any: a JSON body of the Responses API; only the fields the SDK reads are filled in.
def _response(content: list[dict[str, Any]]) -> httpx2.Response:
    return httpx2.Response(
        200,
        json={
            "id": "resp_1",
            "object": "response",
            "created_at": 0,
            "model": SETTINGS.llm_model,
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg_1",
                    "status": "completed",
                    "role": "assistant",
                    "content": content,
                }
            ],
        },
    )


def _answer(answer: LlmAnswer) -> httpx2.Response:
    return _response([{"type": "output_text", "text": answer.model_dump_json(), "annotations": []}])


def test_parses_structured_answer_and_sends_settings() -> None:
    requests: list[httpx2.Request] = []
    expected = LlmAnswer(answer="Ukupno je 1.575,00 EUR [1].", found=True, citations=[1])

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return _answer(expected)

    answer = _generator(handler).generate("Koliki je iznos?", CONTEXTS)

    assert answer == expected
    [request] = requests
    assert request.url.path == "/v1/responses"
    body = json.loads(request.content)
    assert body["model"] == "gpt-6-luna"
    assert body["reasoning"] == {"effort": "low"}
    assert body["store"] is False
    assert [m["role"] for m in body["input"]] == ["system", "user"]
    assert body["text"]["format"]["type"] == "json_schema"
    assert body["text"]["format"]["strict"] is True


def test_server_error_becomes_llm_request_error() -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(500, json={"error": {"message": "boom"}})

    with pytest.raises(LlmRequestError):
        _generator(handler).generate("Pitanje?", CONTEXTS)
    assert len(requests) == 2  # first try + max_retries=1


def test_refusal_becomes_llm_request_error() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return _response([{"type": "refusal", "refusal": "I can't help with that."}])

    with pytest.raises(LlmRequestError):
        _generator(handler).generate("Pitanje?", CONTEXTS)


def test_without_key_the_generator_is_unavailable() -> None:
    generator = create_generator(Settings())

    assert isinstance(generator, UnavailableGenerator)
    assert generator.model == "gpt-6-luna"
    with pytest.raises(LlmUnavailableError):
        generator.generate("Pitanje?", CONTEXTS)


def test_with_key_the_generator_uses_openai() -> None:
    assert isinstance(create_generator(SETTINGS), OpenAiGenerator)


def test_openai_generator_needs_a_key() -> None:
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAiGenerator(Settings())
