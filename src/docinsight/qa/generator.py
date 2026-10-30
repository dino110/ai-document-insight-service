"""LLM answer generation behind a protocol, with structured output."""

import logging
from typing import Protocol

import httpx2
from openai import OpenAI, OpenAIError
from openai.types.responses import EasyInputMessageParam, ResponseInputParam
from pydantic import BaseModel, Field

from docinsight.config import Settings
from docinsight.qa.prompts import ContextChunk, build_messages

logger = logging.getLogger(__name__)


class LlmUnavailableError(RuntimeError):
    """No LLM is configured (no API key)."""


class LlmRequestError(RuntimeError):
    """The LLM call failed: timeout, HTTP error, or no usable answer."""


class LlmAnswer(BaseModel):
    """The JSON shape the model must return. Descriptions are part of the schema it sees."""

    answer: str = Field(description="Answer to the question, with [n] citations.")
    found: bool = Field(description="False when the excerpts do not contain the answer.")
    citations: list[int] = Field(description="Numbers of the excerpts the answer is based on.")


class AnswerGenerator(Protocol):
    model: str

    def generate(self, question: str, contexts: list[ContextChunk]) -> LlmAnswer:
        """Answer ``question`` from ``contexts`` only."""
        ...


class OpenAiGenerator:
    """OpenAI Responses API (or any compatible server via ``llm_base_url``)."""

    def __init__(self, settings: Settings, http_client: httpx2.Client | None = None) -> None:
        if settings.openai_api_key is None:
            raise ValueError("The LLM needs DOCINSIGHT_OPENAI_API_KEY")
        self._client = OpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            base_url=settings.llm_base_url,
            timeout=settings.llm_timeout_s,
            # One retry: the user is waiting for this answer.
            max_retries=1,
            http_client=http_client,
        )
        self.model = settings.llm_model
        self._reasoning_effort = settings.llm_reasoning_effort

    def generate(self, question: str, contexts: list[ContextChunk]) -> LlmAnswer:
        messages: ResponseInputParam = [
            EasyInputMessageParam(role=m["role"], content=m["content"])
            for m in build_messages(question, contexts)
        ]
        try:
            response = self._client.responses.parse(
                model=self.model,
                input=messages,
                text_format=LlmAnswer,  # JSON Schema with strict=True, parsed back by the SDK
                reasoning={"effort": self._reasoning_effort},
                store=False,  # do not keep the response (and the excerpts) on OpenAI's side
            )
        except OpenAIError as exc:  # also LengthFinishReasonError, which is not an APIError
            raise LlmRequestError("LLM request failed") from exc
        if response.output_parsed is None:  # e.g. the model refused
            raise LlmRequestError("LLM returned no parsable answer")
        return response.output_parsed


class UnavailableGenerator:
    """Placeholder without an API key: /ask answers 503 instead of failing at startup."""

    def __init__(self, model: str) -> None:
        self.model = model

    def generate(self, question: str, contexts: list[ContextChunk]) -> LlmAnswer:
        raise LlmUnavailableError("No LLM is configured")


def create_generator(settings: Settings) -> AnswerGenerator:
    if settings.openai_api_key is None:
        logger.warning("No OpenAI API key: /ask is disabled")
        return UnavailableGenerator(settings.llm_model)
    logger.info("LLM: %s (reasoning effort %s)", settings.llm_model, settings.llm_reasoning_effort)
    return OpenAiGenerator(settings)
