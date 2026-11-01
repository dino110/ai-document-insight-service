"""Tests for masking OIBs and IBANs before anything is sent to OpenAI."""

import numpy as np
import pytest
from factories import FakeGenerator
from numpy.typing import NDArray
from sqlmodel import Session

from docinsight.config import Settings
from docinsight.extraction.service import PageText
from docinsight.indexing.chunking import chunk_pages, document_intro
from docinsight.indexing.embeddings import HashingEmbedder
from docinsight.nlp.masking import IdentifierMasker, mask_identifiers
from docinsight.processing import build_chunks
from docinsight.qa.generator import LlmAnswer
from docinsight.qa.prompts import HistoryTurn
from docinsight.qa.rag import answer_question
from docinsight.storage.models import Document, DocumentPage

OIB = "99083863799"
OTHER_OIB = "28196001332"
IBAN = "HR7223600004275179790"
IBAN_GROUPED = "HR72 2360 0004 2751 7979 0"
IBAN_PARTS = ("2360", "0004", "2751", "7979")


class RecordingEmbedder(HashingEmbedder):
    """HashingEmbedder that keeps every text it was asked to embed."""

    def __init__(self) -> None:
        super().__init__()
        self.texts: list[str] = []

    def embed_documents(self, texts: list[str]) -> NDArray[np.float32]:
        self.texts.extend(texts)
        return super().embed_documents(texts)


def leaks(text: str) -> bool:
    """Whether any digits of the test identifiers are in ``text``."""
    return OIB in text or OTHER_OIB in text or any(part in text for part in IBAN_PARTS)


# --- Masking ------------------------------------------------------------------


def test_mask_identifiers_for_embeddings() -> None:
    text = f"OIB: {OIB}, IBAN: {IBAN_GROUPED}, uplata na {IBAN}."

    assert mask_identifiers(text) == "OIB: [OIB], IBAN: [IBAN], uplata na [IBAN]."


def test_numbers_that_are_not_valid_identifiers_stay() -> None:
    # A wrong check digit, an invoice number and an amount are not identifiers.
    text = "OIB 99083863790, račun 2026-001, iznos 1.575,00 EUR"

    assert mask_identifiers(text) == text
    assert IdentifierMasker().mask(text) == text


def test_same_value_gets_the_same_placeholder_in_every_text() -> None:
    masker = IdentifierMasker()

    first = masker.mask(f"Kupac OIB {OIB}, izdavatelj OIB {OTHER_OIB}, IBAN {IBAN_GROUPED}")
    second = masker.mask(f"Ponovno {OTHER_OIB} i {IBAN}")  # the IBAN written compactly

    assert first == "Kupac OIB [OIB_1], izdavatelj OIB [OIB_2], IBAN [IBAN_1]"
    assert second == "Ponovno [OIB_2] i [IBAN_1]"


def test_unmask_restores_the_values() -> None:
    masker = IdentifierMasker()
    text = f"OIB {OIB}, IBAN {IBAN_GROUPED}"

    assert masker.unmask(masker.mask(text)) == text
    # The LLM may use a placeholder anywhere: the first spelling seen comes back.
    assert masker.unmask("Uplata na [IBAN_1], OIB [OIB_1] [1]") == (
        f"Uplata na {IBAN_GROUPED}, OIB {OIB} [1]"
    )


def test_unknown_placeholders_are_left_as_they_are() -> None:
    masker = IdentifierMasker()
    masker.mask(f"OIB {OIB}")

    assert masker.unmask("[OIB_2] [IBAN_1] [OIB 1]") == "[OIB_2] [IBAN_1] [OIB 1]"


# --- No chunk or intro splits an IBAN -------------------------------------------


@pytest.mark.parametrize("filler", range(0, 230, 7))
def test_no_chunk_splits_an_iban(filler: int) -> None:
    # The IBAN moves along the text, so some windows would end between its groups.
    text = "riječ " * (filler // 6) + f"IBAN {IBAN_GROUPED} " + "kraj teksta " * 30

    chunks = chunk_pages([(1, text)], 200, 50)

    for chunk in chunks:
        masked = mask_identifiers(chunk.text)
        assert not leaks(masked), chunk.text  # whole IBAN (masked) or none of it


@pytest.mark.parametrize("filler", range(170, 200))
def test_intro_does_not_split_an_iban(filler: int) -> None:
    page = "x" * filler + f" {IBAN_GROUPED} kraj"

    intro = document_intro(page)

    assert not leaks(mask_identifiers(intro)), intro


# --- What goes out: embeddings and the LLM ------------------------------------


def test_embedded_text_is_masked_and_stored_text_is_not(session: Session) -> None:
    embedder = RecordingEmbedder()
    document = Document(
        owner_id="owner",
        filename=f"racun-{OIB}.pdf",
        content_type="application/pdf",
        size_bytes=1,
        sha256="0" * 64,
        stored_path="x",
    )
    page_text = f"Kupac Horvat d.o.o., OIB {OIB}\nIBAN: {IBAN_GROUPED}\nUkupno 1.575,00 EUR"
    chunks = build_chunks(document, [PageText(1, page_text, "text_layer")], embedder, Settings())

    assert embedder.texts and not any(leaks(text) for text in embedder.texts)
    assert "[OIB]" in embedder.texts[0] and "[IBAN]" in embedder.texts[0]
    assert [chunk.text for chunk in chunks] == [page_text]  # BM25 and the UI need the values


def add_invoice(session: Session, embedder: HashingEmbedder) -> None:
    """An indexed invoice whose page text holds an OIB and an IBAN."""
    document = Document(
        owner_id="owner",
        filename="racun.pdf",
        content_type="application/pdf",
        size_bytes=1,
        sha256="0" * 64,
        stored_path="x",
        status="processed",
    )
    text = (
        f"Račun 2026-001, kupac Horvat d.o.o., OIB {OIB}, "
        f"izdavatelj OIB {OTHER_OIB}, IBAN {IBAN_GROUPED}"
    )
    session.add(document)
    session.add(DocumentPage(document_id=document.id, page_number=1, text=text, method="ocr"))
    session.add_all(build_chunks(document, [PageText(1, text, "ocr")], embedder, Settings()))
    session.commit()


def test_llm_sees_placeholders_and_the_user_gets_the_values(session: Session) -> None:
    embedder = RecordingEmbedder()
    add_invoice(session, embedder)
    embedder.texts.clear()
    generator = FakeGenerator(
        LlmAnswer(
            answer="Izdavatelj ima OIB [OIB_2] i IBAN [IBAN_1] [1]", found=True, citations=[1]
        )
    )

    result = answer_question(
        f"Čiji je OIB {OTHER_OIB}?",
        "owner",
        session,
        embedder,
        generator,
        Settings(),
        history=[
            HistoryTurn("user", "Koji je IBAN?"),
            HistoryTurn("assistant", f"IBAN je {IBAN}."),
        ],
    )

    [(question, contexts)] = generator.calls
    [history] = generator.histories
    sent = [question, *(c.text for c in contexts), *(c.intro for c in contexts)]
    sent += [turn.text for turn in history]
    assert not any(leaks(text) for text in sent)
    # One masker for the whole request: the question's OIB is the second one in the excerpts.
    assert question == "Čiji je OIB [OIB_2]?"
    assert "OIB [OIB_1], izdavatelj OIB [OIB_2], IBAN [IBAN_1]" in contexts[0].text
    assert history[1].text == "IBAN je [IBAN_1]."  # compact in history, same placeholder
    # The question was embedded masked; BM25 (local) still found the invoice by the OIB.
    assert embedder.texts == ["Koji je IBAN? Čiji je OIB [OIB]?"]
    assert result.answer == f"Izdavatelj ima OIB {OTHER_OIB} i IBAN {IBAN_GROUPED} [1]"
    assert [source.filename for source in result.sources] == ["racun.pdf"]
