"""Tests for fixing legal forms in OCR text."""

import pytest

from docinsight.extraction.ocr_cleanup import clean_ocr_text
from docinsight.nlp.patterns import find_companies


@pytest.mark.parametrize(
    ("ocr", "canonical"),
    [
        ("d.0.0.", "d.o.o."),
        ("d.0.0", "d.o.o."),
        ("d.O.o.", "d.o.o."),
        ("d. o. o.", "d.o.o."),
        ("d,o,o.", "d.o.o."),
        ("d.o.o.", "d.o.o."),
        ("j.d.0.0", "j.d.o.o."),
        ("j. d. o. o.", "j.d.o.o."),
        ("d,d.", "d.d."),
        ("d.d", "d.d."),
        ("j.t,d.", "j.t.d."),
        ("k. d.", "k.d."),
    ],
)
def test_legal_forms_become_canonical(ocr: str, canonical: str) -> None:
    assert clean_ocr_text(f"Kupac: Horvat {ocr} Zagreb") == f"Kupac: Horvat {canonical} Zagreb"


@pytest.mark.parametrize(
    ("ocr", "clean"),
    [
        ("Car d.0.0. Mlinarska 3a", "Car d.o.o. Mlinarska 3a"),
        ("Horvat d.0.0 Kraljevecki odvojak", "Horvat d.o.o. Kraljevecki odvojak"),
        ("Nacinoviá d.0.0., Mihaljevac 3", "Nacinoviá d.o.o., Mihaljevac 3"),
        ("Car d.0.0.Mlinarska", "Car d.o.o.Mlinarska"),  # no second dot
    ],
)
def test_legal_forms_in_scanned_lines(ocr: str, clean: str) -> None:
    assert clean_ocr_text(ocr) == clean


@pytest.mark.parametrize(
    "text",
    [
        "UKUPNO 100,00 EUR",
        "OIB: 28196001332, IBAN: HR72 2360 0004 2751 7979 0",
        "Datum 01.02.2026.",
        "pod.0.0.kom",  # inside a word
        "add,d",  # a letter before the form
        "d.0.0k",  # a letter after the form
        "d,d0",  # a digit after the form
        "Ugovor 0 djelu i d.",  # zeros and lone letters outside a legal form
    ],
)
def test_other_text_is_left_alone(text: str) -> None:
    assert clean_ocr_text(text) == text


def test_cleaned_text_lets_the_company_rule_find_scanned_companies() -> None:
    ocr = "Izdavatelj: Car d.0.0. Mlinarska 3a\nKupac: Horvat d.0.0 Kraljevecki odvojak"
    assert find_companies(ocr) == []

    clean = clean_ocr_text(ocr)

    assert [clean[start:end] for start, end in find_companies(clean)] == [
        "Car d.o.o.",
        "Horvat d.o.o.",
    ]
