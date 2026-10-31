"""Rules for entities with a fixed format: OIB, IBAN, dates and amounts of money.

A regex only checks the shape, so OIBs and IBANs must also pass their check digits and a
date must exist in the calendar. Every ``find_*`` function returns ``(start, end)`` spans
in the given text, so ``text[start:end]`` is the match.
"""

import re
from datetime import date
from decimal import Decimal

Span = tuple[int, int]

# Lookarounds instead of \b: \b finds no boundary between "R" and "1" in "HR12345678901"
# (a VAT number is "HR" + OIB), and 11 digits inside a longer number are not an OIB.
_OIB = re.compile(r"(?<!\d)\d{11}(?!\d)")

# Croatian IBAN: HR, 2 check digits and 17 digits, compact or in groups of four
# ("HR7223600004275179790" or "HR72 2360 0004 2751 7979 0").
_IBAN = re.compile(r"(?<![^\W_])HR\d{2}(?: ?\d{4}){4} ?\d(?!\d)")
_IBAN_COMPACT = re.compile(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}")

# "29.03.2026.", "29. 3. 2026." and ISO "2026-03-29". The final dot belongs to a
# Croatian date. "2026-001" (an invoice number) is not a date.
_DATE_HR = re.compile(r"(?<![\d.])(\d{1,2})\. ?(\d{1,2})\. ?(\d{4})\.?(?!\d)")
_DATE_ISO = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")

# "1.575,00", "1,575.00", "1575.00", "1.575": groups of three, optional cents.
_AMOUNT = r"\d+(?:[.,]\d{3})*(?:[.,]\d{2})?"
# Only with a currency: "25,00 %" or a quantity is not money. [^\W\d_] means a letter.
_CURRENCY = r"(?:EUR|HRK|USD|GBP|[Ee]ur[ao]?|kn|€|\$|£)(?![^\W\d_])"
_MONEY = re.compile(
    rf"(?<![^\W\d_]){_CURRENCY}[ \u00a0]?{_AMOUNT}(?!\d)"  # "€ 1,575.00"
    rf"|(?<![\d.,]){_AMOUNT}[ \u00a0]?{_CURRENCY}"  # "1.575,00 EUR"
)
_AMOUNT_ONLY = re.compile(rf"(?<![\d.,]){_AMOUNT}(?!\d)")


def is_valid_oib(text: str) -> bool:
    """11 digits whose last one is the ISO 7064 MOD 11,10 check digit."""
    if not re.fullmatch(r"\d{11}", text):
        return False
    a = 10
    for char in text[:10]:
        a = (int(char) + a) % 10 or 10
        a = a * 2 % 11
    return (11 - a) % 10 == int(text[10])


def is_valid_iban(text: str) -> bool:
    """ISO 13616: move the first 4 characters to the end, letters become 10..35, mod 97 == 1.

    Spaces and lower case are allowed; any country is accepted here.
    """
    compact = text.replace(" ", "").upper()
    if not _IBAN_COMPACT.fullmatch(compact):
        return False
    rearranged = compact[4:] + compact[:4]
    # int(char, 36) turns "0".."9" into 0..9 and "A".."Z" into 10..35.
    return int("".join(str(int(char, 36)) for char in rearranged)) % 97 == 1


def find_oibs(text: str) -> list[Span]:
    return [m.span() for m in _OIB.finditer(text) if is_valid_oib(m.group())]


def find_ibans(text: str) -> list[Span]:
    """Croatian IBANs with valid check digits; a span covers the spaces between groups."""
    return [m.span() for m in _IBAN.finditer(text) if is_valid_iban(m.group())]


def find_dates(text: str) -> list[Span]:
    spans = [m.span() for m in _DATE_HR.finditer(text) if _to_date(m, day_first=True)]
    spans += [m.span() for m in _DATE_ISO.finditer(text) if _to_date(m, day_first=False)]
    return sorted(spans)


def find_amounts(text: str) -> list[Span]:
    """Amounts with a currency before or after them."""
    return [m.span() for m in _MONEY.finditer(text)]


def normalize_amount(text: str) -> Decimal:
    """'1.575,00 EUR', '€ 1,575.00' and '1575.00' -> Decimal('1575.00').

    A separator followed by exactly two digits at the end is the decimal one; every
    other separator groups thousands.
    """
    match = _AMOUNT_ONLY.search(text)
    if match is None:
        raise ValueError("no amount in text")
    number = match.group()
    if len(number) > 3 and number[-3] in ".,":
        whole, cents = number[:-3], number[-2:]
    else:
        whole, cents = number, "00"
    return Decimal(f"{whole.replace('.', '').replace(',', '')}.{cents}")


def normalize_date(text: str) -> date:
    """'29.03.2026.', '29. 3. 2026.' and '2026-03-29' -> date(2026, 3, 29)."""
    for pattern, day_first in ((_DATE_HR, True), (_DATE_ISO, False)):
        match = pattern.search(text)
        if match and (result := _to_date(match, day_first=day_first)):
            return result
    raise ValueError("no valid date in text")


def _to_date(match: re.Match[str], *, day_first: bool) -> date | None:
    """The date of a match, or None when it does not exist (31.02.2026.)."""
    first, second, third = (int(group) for group in match.groups())
    day, month, year = (first, second, third) if day_first else (third, second, first)
    try:
        return date(year, month, day)
    except ValueError:
        return None
