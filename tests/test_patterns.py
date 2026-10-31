"""Tests for the rule-based patterns: OIB, IBAN, dates, amounts and company names."""

import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from docinsight.nlp.patterns import (
    find_amounts,
    find_companies,
    find_dates,
    find_ibans,
    find_oibs,
    is_valid_iban,
    is_valid_oib,
    normalize_amount,
    normalize_date,
    normalize_entity,
)

EXPECTED = json.loads(
    (Path(__file__).parent.parent / "sample_docs" / "expected.json").read_text(encoding="utf-8")
)
INVOICES = [value for name, value in EXPECTED.items() if name.startswith("racun-")]
CONTRACTS = [value for name, value in EXPECTED.items() if name.startswith("ugovor-")]
OIBS = sorted({inv[key] for inv in INVOICES for key in ("issuer_oib", "buyer_oib")})
IBANS = sorted({inv["issuer_iban"] for inv in INVOICES})
AMOUNTS = sorted(
    {inv[key] for inv in INVOICES for key in ("net", "vat", "total")}
    | {contract["fee"] for contract in CONTRACTS}
)
DATES = sorted(
    {inv[key] for inv in INVOICES for key in ("issue_date", "due_date")}
    | {contract[key] for contract in CONTRACTS for key in ("start_date", "end_date")}
)


def found(text: str, spans: list[tuple[int, int]]) -> list[str]:
    return [text[start:end] for start, end in spans]


def change_digit(value: str, index: int) -> str:
    return value[:index] + str((int(value[index]) + 1) % 10) + value[index + 1 :]


def croatian_amount(amount: str) -> str:
    """'1575.00' -> '1.575,00', the format printed on the sample documents."""
    return f"{Decimal(amount):,.2f}".replace(",", " ").replace(".", ",").replace(" ", ".")


# --- OIB ------------------------------------------------------------------------


def test_sample_documents_have_the_expected_numbers_of_identifiers() -> None:
    assert (len(OIBS), len(IBANS)) == (6, 3)


@pytest.mark.parametrize("oib", OIBS)
def test_sample_oibs_are_valid(oib: str) -> None:
    assert is_valid_oib(oib)


@pytest.mark.parametrize("oib", OIBS)
def test_oib_with_a_changed_digit_is_invalid(oib: str) -> None:
    assert not is_valid_oib(change_digit(oib, 4))


@pytest.mark.parametrize("text", ["2819600133", "281960013320", "2819600133a", ""])
def test_oib_needs_exactly_eleven_digits(text: str) -> None:
    assert not is_valid_oib(text)


def test_find_oibs_in_a_sentence() -> None:
    text = "Kupac Horvat d.o.o., OIB 99083863799, PDV ID HR28196001332."

    assert found(text, find_oibs(text)) == ["99083863799", "28196001332"]


@pytest.mark.parametrize("number", ["990838637995", "12345678901" + "99083863799"])
def test_valid_oib_inside_a_longer_number_is_not_an_oib(number: str) -> None:
    # The valid OIB 99083863799 starts or ends the longer number.
    assert find_oibs(f"Broj {number} nije OIB.") == []


def test_find_oibs_skips_numbers_with_a_wrong_check_digit() -> None:
    assert find_oibs(f"OIB {change_digit('99083863799', 10)}") == []


# --- IBAN -----------------------------------------------------------------------


@pytest.mark.parametrize("iban", IBANS)
def test_sample_ibans_are_valid(iban: str) -> None:
    assert is_valid_iban(iban)


@pytest.mark.parametrize("iban", IBANS)
def test_iban_with_a_changed_digit_is_invalid(iban: str) -> None:
    assert not is_valid_iban(change_digit(iban, 10))


def test_iban_from_another_country_is_valid() -> None:
    # The standard example IBAN from the ISO 13616 / Wikipedia documentation.
    assert is_valid_iban("GB82 WEST 1234 5698 7654 32")


@pytest.mark.parametrize("text", ["HR72", "HR7223600004275179790X!", "7223600004275179790"])
def test_malformed_iban_is_invalid(text: str) -> None:
    assert not is_valid_iban(text)


def test_find_ibans_compact_and_grouped() -> None:
    text = "IBAN: HR72 2360 0004 2751 7979 0. Uplata na IBAN HR7223600004275179790 do petka."

    assert found(text, find_ibans(text)) == [
        "HR72 2360 0004 2751 7979 0",
        "HR7223600004275179790",
    ]


def test_find_ibans_skips_an_invalid_iban() -> None:
    assert find_ibans(f"IBAN {change_digit('HR7223600004275179790', 10)}") == []


def test_iban_inside_a_longer_word_is_not_an_iban() -> None:
    assert find_ibans("REF XHR7223600004275179790") == []


def test_digits_of_an_iban_are_not_an_oib() -> None:
    text = "HR7223600004275179790 i HR72 2360 0004 2751 7979 0"

    assert find_oibs(text) == []


# --- Dates ----------------------------------------------------------------------


def test_find_dates_croatian_and_iso() -> None:
    text = "Izdan 27.02.2026., dospijeće 29. 3. 2026. (ISO 2026-03-29)."

    assert found(text, find_dates(text)) == ["27.02.2026.", "29. 3. 2026.", "2026-03-29"]


def test_invoice_number_and_impossible_dates_are_not_dates() -> None:
    text = "Račun 2026-001 od 31.02.2026., 2026-13-01 i šifra 2026-03-291"

    assert find_dates(text) == []


@pytest.mark.parametrize("iso", DATES)
def test_normalize_sample_dates(iso: str) -> None:
    expected = date.fromisoformat(iso)

    assert normalize_date(iso) == expected
    assert normalize_date(expected.strftime("%d.%m.%Y.")) == expected


def test_normalize_date_without_a_date_raises() -> None:
    with pytest.raises(ValueError, match="no valid date"):
        normalize_date("31.02.2026.")


# --- Amounts --------------------------------------------------------------------


def test_find_amounts_croatian_and_english() -> None:
    text = "Ukupno 1.575,00 EUR (€ 1,575.00), PDV 315,00 €, ranije 2.000 kn."

    assert found(text, find_amounts(text)) == [
        "1.575,00 EUR",
        "€ 1,575.00",
        "315,00 €",
        "2.000 kn",
    ]


def test_amount_with_a_no_break_space_before_the_currency() -> None:
    text = "Ukupno 1.575,00\u00a0EUR"  # word processors often keep number and unit together

    assert found(text, find_amounts(text)) == ["1.575,00\u00a0EUR"]


def test_amount_and_currency_on_two_lines() -> None:
    # From the English sample contract: the line wraps between the number and "EUR".
    text = "a total fee of 8.800,00\nEUR (gross)"

    assert found(text, find_amounts(text)) == ["8.800,00\nEUR"]
    assert normalize_amount("8.800,00\nEUR") == Decimal("8800.00")


def test_amount_and_currency_are_not_joined_across_a_blank_line() -> None:
    assert find_amounts("Količina 3\n\nEUR je valuta") == []


def test_numbers_without_a_currency_are_not_amounts() -> None:
    assert find_amounts("PDV 25,00 %, količina 3, račun 2026-001, OIB 99083863799") == []


def test_currency_must_be_a_separate_word() -> None:
    assert find_amounts("Kod 100 EURIBOR i Neuro 5") == []


@pytest.mark.parametrize("amount", AMOUNTS)
def test_normalize_sample_amounts(amount: str) -> None:
    expected = Decimal(amount)

    assert normalize_amount(amount) == expected
    assert normalize_amount(f"{croatian_amount(amount)} EUR") == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("€ 1,575.00", "1575.00"),
        ("1.575 EUR", "1575.00"),
        ("1,575 USD", "1575.00"),
        ("12,50 kn", "12.50"),
        ("1.234.567,89 EUR", "1234567.89"),
    ],
)
def test_normalize_amount_formats(text: str, expected: str) -> None:
    assert normalize_amount(text) == Decimal(expected)


def test_normalize_amount_without_a_number_raises() -> None:
    with pytest.raises(ValueError, match="no amount"):
        normalize_amount("EUR")


# --- Companies ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "companies"),
    [
        # Answers from the real API (step 4), where the models got these wrong.
        ("Račun 2026-001 izdao je Car d.o.o.; OIB je 28196001332", ["Car d.o.o."]),
        ("Strane su Načinović d.o.o. (Naručitelj) i Elvis Antić", ["Načinović d.o.o."]),
        ("The parties are Šalić d.o.o., the Client", ["Šalić d.o.o."]),
        # Other forms, several words, connectors and a dropped final dot.
        ("Kupac je Horvat i Sinovi d.d. iz Splita", ["Horvat i Sinovi d.d."]),
        ("Izdavatelj: Kovač j.d.o.o., Zagreb", ["Kovač j.d.o.o."]),
        ("Ortaci su Babić j.t.d. i Perić k.d.", ["Babić j.t.d.", "Perić k.d."]),
        ("Izdao je Car d.o.o", ["Car d.o.o"]),
        ("AGRO-PLUS d.o.o. dostavlja", ["AGRO-PLUS d.o.o."]),
        ("Acme, Inc. and Foo & Bar GmbH, Baz Ltd and Qux LLC",
         ["Acme, Inc.", "Foo & Bar GmbH", "Baz Ltd", "Qux LLC"]),
    ],
)  # fmt: skip
def test_find_companies(text: str, companies: list[str]) -> None:
    assert found(text, find_companies(text)) == companies


@pytest.mark.parametrize(
    ("text", "companies"),
    [
        ("Kupac Horvat d.o.o. plaća", ["Horvat d.o.o."]),
        ("Tvrtka Horvat i Sinovi d.d.", ["Horvat i Sinovi d.d."]),
        ("The Client Acme Ltd. pays", ["Acme Ltd."]),
        ("Kupac i Prodavatelj Horvat d.o.o.", ["Horvat d.o.o."]),  # a connector between roles
    ],
)
def test_role_words_before_the_name_are_left_out(text: str, companies: list[str]) -> None:
    assert found(text, find_companies(text)) == companies


@pytest.mark.parametrize(
    "text",
    [
        "Tvrtka d.o.o. je osnovana",  # only a role word before the form
        "osnovano je društvo s ograničenom odgovornošću d.o.o.",  # lowercase words
        "horvat d.o.o.",  # a name starts with a capital letter
        "Horvat d.o.offline",  # the form must end the word
        "HorvatAG je",  # the form must be a separate word
        # A name must start a word: no "Kupi d.o.o." out of "eKupi d.o.o." (missed instead).
        "eKupi d.o.o.",
        "Iznos plaća Horvat.",  # no legal form
    ],
)
def test_not_a_company(text: str) -> None:
    assert find_companies(text) == []


# --- Normalised entities --------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "text", "normalized"),
    [
        ("OIB", "99083863799", "99083863799"),
        ("IBAN", "HR72 2360 0004 2751 7979 0", "HR7223600004275179790"),
        ("IBAN", "hr7223600004275179790", "HR7223600004275179790"),
        ("MONEY", "1.575,00 EUR", "1575.00"),
        ("MONEY", "€ 1,575.00", "1575.00"),
        ("MONEY", "8.800,00\nEUR", "8800.00"),
        ("DATE", "29.03.2026.", "2026-03-29"),
        ("DATE", "29. 3. 2026.", "2026-03-29"),
        ("ORG", "Horvat\nd.o.o.", "Horvat d.o.o."),
        ("PERSON", "  Ana   Kovač ", "Ana Kovač"),
    ],
)
def test_normalize_entity(label: str, text: str, normalized: str) -> None:
    assert normalize_entity(label, text) == normalized
