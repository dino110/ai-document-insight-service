"""Generate fictional invoices and contracts for manual tests, demos and evaluation.

All data is invented (Faker, random OIBs with a valid check digit). The same seed
always produces the same documents and the same ``expected.json``.

Usage:
    uv run python scripts/generate_sample_docs.py [--out sample_docs] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from io import BytesIO
from pathlib import Path
from typing import Literal

import numpy as np
import pymupdf
from faker import Faker
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from PIL import Image, ImageFilter

FONTS = Path(__file__).parent / "fonts"
VAT_RATE = Decimal("0.25")
CENT = Decimal("0.01")
# Fixed PDF creation date, otherwise every run produces different bytes.
CREATION_DATE = datetime(2026, 1, 1, tzinfo=UTC)

SERVICES = [
    ("Izrada web stranice", Decimal("1200.00")),
    ("Održavanje poslužitelja (mjesečno)", Decimal("180.00")),
    ("Savjetovanje - sat rada", Decimal("65.00")),
    ("Licenca za softver (godišnja)", Decimal("349.90")),
    ("Grafički dizajn logotipa", Decimal("450.00")),
    ("Prijevod dokumentacije - kartica", Decimal("14.50")),
    ("Izrada mobilne aplikacije - faza 1", Decimal("2500.00")),
    ("Obuka zaposlenika - dan", Decimal("390.00")),
]

TASKS_HR = [
    "analiza postojećih poslovnih procesa naručitelja",
    "izrada tehničke specifikacije sustava",
    "dizajn korisničkog sučelja",
    "programiranje i testiranje aplikacije",
    "migracija postojećih podataka",
    "izrada korisničke dokumentacije",
    "obuka zaposlenika naručitelja",
    "podrška u prvih trideset dana rada sustava",
]

TASKS_EN = [
    "analysis of the client's existing business processes",
    "preparation of the technical specification",
    "user interface design",
    "development and testing of the application",
    "migration of existing data",
    "preparation of user documentation",
    "training of the client's staff",
    "support during the first thirty days of operation",
]


@dataclass(frozen=True)
class Party:
    name: str
    oib: str
    address: str


@dataclass(frozen=True)
class LineItem:
    description: str
    quantity: int
    unit_price: Decimal

    @property
    def amount(self) -> Decimal:
        return (self.unit_price * self.quantity).quantize(CENT, ROUND_HALF_UP)


@dataclass(frozen=True)
class Invoice:
    number: str
    issuer: Party
    buyer: Party
    issue_date: date
    due_date: date
    items: list[LineItem]

    @property
    def net(self) -> Decimal:
        return sum((item.amount for item in self.items), Decimal("0"))

    @property
    def vat(self) -> Decimal:
        return (self.net * VAT_RATE).quantize(CENT, ROUND_HALF_UP)

    @property
    def total(self) -> Decimal:
        return self.net + self.vat


@dataclass(frozen=True)
class Contract:
    language: Literal["hr", "en"]
    client: Party
    contractor: Party
    fee: Decimal
    signed_date: date
    start_date: date
    end_date: date
    tasks: list[str]


# --- Formatting helpers ---------------------------------------------------------


def oib_check_digit(first10: str) -> str:
    """ISO 7064 MOD 11,10 check digit of a Croatian OIB."""
    a = 10
    for ch in first10:
        a = (a + int(ch)) % 10 or 10
        a = (a * 2) % 11
    return str((11 - a) % 10)


def make_oib(rng: random.Random) -> str:
    first10 = str(rng.randint(1, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(9))
    return first10 + oib_check_digit(first10)


def format_money(amount: Decimal) -> str:
    """Croatian format: 1314.6 -> '1.314,60 EUR'."""
    us = f"{amount:,.2f}"  # '1,314.60'
    return us.replace(",", " ").replace(".", ",").replace(" ", ".") + " EUR"


def format_date(d: date) -> str:
    return d.strftime("%d.%m.%Y.")


def random_date(rng: random.Random, start: date, max_days: int) -> date:
    return start + timedelta(days=rng.randint(0, max_days))


# --- Data generation ------------------------------------------------------------


def make_company(fake: Faker, rng: random.Random) -> Party:
    name = fake.company()
    # hr_HR sometimes returns a bare surname ("Car") without a legal form.
    if not name.endswith(("d.o.o.", "d.d.")):
        name += " d.o.o."
    address = f"{fake.street_address()}, {fake.postcode()} {fake.city()}"
    return Party(name=name, oib=make_oib(rng), address=address)


def make_person(fake: Faker, rng: random.Random) -> Party:
    address = f"{fake.street_address()}, {fake.postcode()} {fake.city()}"
    return Party(name=fake.name(), oib=make_oib(rng), address=address)


def make_invoice(index: int, fake: Faker, rng: random.Random) -> Invoice:
    issue = random_date(rng, date(2026, 1, 1), 270)
    services = rng.sample(SERVICES, rng.randint(2, 4))
    items = [LineItem(desc, rng.randint(1, 5), price) for desc, price in services]
    return Invoice(
        number=f"2026-{index:03d}",
        issuer=make_company(fake, rng),
        buyer=make_company(fake, rng),
        issue_date=issue,
        due_date=issue + timedelta(days=rng.choice([15, 30])),
        items=items,
    )


def make_contract(language: Literal["hr", "en"], fake: Faker, rng: random.Random) -> Contract:
    signed = random_date(rng, date(2026, 1, 1), 200)
    start = signed + timedelta(days=rng.randint(1, 14))
    tasks = TASKS_HR if language == "hr" else TASKS_EN
    return Contract(
        language=language,
        client=make_company(fake, rng),
        contractor=make_person(fake, rng),
        fee=Decimal(rng.randint(30, 150) * 100),
        signed_date=signed,
        start_date=start,
        end_date=start + timedelta(days=rng.choice([60, 90, 120])),
        tasks=rng.sample(tasks, rng.randint(5, len(tasks))),
    )


# --- PDF rendering --------------------------------------------------------------


def new_pdf() -> FPDF:
    pdf = FPDF(format="A4")
    pdf.add_font("DejaVu", fname=str(FONTS / "DejaVuSans.ttf"))
    pdf.add_font("DejaVu", style="B", fname=str(FONTS / "DejaVuSans-Bold.ttf"))
    pdf.set_creation_date(CREATION_DATE)
    pdf.set_margins(20, 20, 20)
    pdf.add_page()
    return pdf


def line(pdf: FPDF, text: str, size: int = 10, bold: bool = False, height: float = 5.5) -> None:
    pdf.set_font("DejaVu", style="B" if bold else "", size=size)
    pdf.multi_cell(0, height, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def render_invoice(inv: Invoice) -> bytes:
    pdf = new_pdf()
    line(pdf, f"RAČUN br. {inv.number}", size=18, bold=True, height=10)
    pdf.ln(4)

    line(pdf, "Izdavatelj:", bold=True)
    line(pdf, f"{inv.issuer.name}\n{inv.issuer.address}\nOIB: {inv.issuer.oib}")
    pdf.ln(3)
    line(pdf, "Kupac:", bold=True)
    line(pdf, f"{inv.buyer.name}\n{inv.buyer.address}\nOIB: {inv.buyer.oib}")
    pdf.ln(3)
    line(pdf, f"Datum izdavanja: {format_date(inv.issue_date)}")
    line(pdf, f"Datum dospijeća: {format_date(inv.due_date)}")
    pdf.ln(6)

    pdf.set_font("DejaVu", size=10)
    with pdf.table(
        col_widths=(80, 20, 35, 35),
        text_align=("LEFT", "RIGHT", "RIGHT", "RIGHT"),
    ) as table:
        header = table.row()
        for title in ("Opis", "Kol.", "Cijena", "Iznos"):
            header.cell(title)
        for item in inv.items:
            row = table.row()
            row.cell(item.description)
            row.cell(str(item.quantity))
            row.cell(format_money(item.unit_price))
            row.cell(format_money(item.amount))
    pdf.ln(6)

    for label, amount, bold in (
        ("Osnovica", inv.net, False),
        ("PDV 25 %", inv.vat, False),
        ("UKUPNO ZA PLATITI", inv.total, True),
    ):
        pdf.set_font("DejaVu", style="B" if bold else "", size=11)
        pdf.cell(120, 7, label, align="R")
        pdf.cell(50, 7, format_money(amount), align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(10)
    line(pdf, f"Molimo uplatu do {format_date(inv.due_date)} uz poziv na broj {inv.number}.")
    return bytes(pdf.output())  # output() returns a bytearray


def contract_articles(c: Contract) -> tuple[str, list[tuple[str, str]]]:
    """Return the title and (heading, body) pairs in the contract's language."""
    fee = format_money(c.fee)
    start, end, signed = (format_date(d) for d in (c.start_date, c.end_date, c.signed_date))
    tasks = "\n".join(f"  {i}. {task}" for i, task in enumerate(c.tasks, start=1))

    if c.language == "hr":
        intro = (
            f"Sklopljen dana {signed} u Zagrebu između:\n\n"
            f"{c.client.name}, {c.client.address}, OIB: {c.client.oib} "
            f"(u daljnjem tekstu: Naručitelj)\n\ni\n\n"
            f"{c.contractor.name}, {c.contractor.address}, OIB: {c.contractor.oib} "
            f"(u daljnjem tekstu: Izvršitelj)"
        )
        bodies = [
            (
                "Predmet ugovora",
                "Ovim ugovorom Izvršitelj se obvezuje za Naručitelja izvršiti sljedeće "
                f"poslove:\n{tasks}\n\nIzvršitelj će poslove obaviti samostalno, "
                "savjesno i u skladu s pravilima struke.",
            ),
            (
                "Naknada",
                f"Za izvršeno djelo Naručitelj se obvezuje Izvršitelju isplatiti ukupnu "
                f"naknadu u iznosu od {fee} (bruto). Naknada se isplaćuje u roku od "
                "15 dana od primopredaje djela, na račun Izvršitelja.",
            ),
            (
                "Rok izvršenja",
                f"Izvršitelj započinje s radom {start}, a djelo je dužan predati "
                f"najkasnije do {end}. Rok se može produžiti samo pisanim dodatkom "
                "ovom ugovoru.",
            ),
            (
                "Obveze Naručitelja",
                "Naručitelj se obvezuje Izvršitelju pravodobno staviti na raspolaganje "
                "sve podatke, dokumentaciju i pristupe potrebne za izvršenje djela te "
                "imenovati osobu za kontakt koja je ovlaštena davati upute i potvrđivati "
                "isporuke. Kašnjenje Naručitelja s ispunjenjem ovih obveza na odgovarajući "
                "način produžuje rok izvršenja.",
            ),
            (
                "Primopredaja",
                "O primopredaji djela sastavlja se zapisnik koji potpisuju obje ugovorne "
                "strane. Naručitelj je dužan djelo pregledati u roku od osam dana od "
                "primitka i o uočenim nedostacima bez odgađanja obavijestiti Izvršitelja. "
                "Izvršitelj je dužan uočene nedostatke otkloniti u primjerenom roku, "
                "bez prava na dodatnu naknadu.",
            ),
            (
                "Povjerljivost",
                "Izvršitelj se obvezuje kao poslovnu tajnu čuvati sve podatke i "
                "informacije za koje sazna tijekom izvršenja ovog ugovora, osobito "
                "podatke o poslovanju, klijentima i zaposlenicima Naručitelja. Obveza "
                "čuvanja povjerljivosti traje i dvije godine nakon prestanka ovog ugovora.",
            ),
            (
                "Autorska prava",
                "Izvršitelj na Naručitelja prenosi isključiva, vremenski i prostorno "
                "neograničena imovinska autorska prava na djelu nastalom izvršenjem "
                "ovog ugovora. Naknada za prijenos prava sadržana je u naknadi iz "
                "članka 2. ovog ugovora.",
            ),
            (
                "Raskid ugovora",
                "Svaka ugovorna strana može raskinuti ovaj ugovor pisanom obavijesti "
                "ako druga strana ne ispunjava svoje obveze ni u roku od petnaest dana "
                "od primitka pisane opomene. U slučaju raskida Izvršitelj ima pravo na "
                "razmjerni dio naknade za do tada izvršene poslove.",
            ),
            (
                "Rješavanje sporova",
                "Ugovorne strane će sporove nastojati riješiti sporazumno. Ako to ne "
                "bude moguće, nadležan je stvarno nadležni sud u Zagrebu.",
            ),
            (
                "Završne odredbe",
                "Na odnose koji nisu uređeni ovim ugovorom primjenjuju se odredbe Zakona "
                "o obveznim odnosima. Ugovor je sastavljen u dva istovjetna primjerka, "
                "od kojih svaka strana zadržava po jedan, a stupa na snagu danom "
                "potpisa obiju strana.",
            ),
        ]
        articles = [(f"Članak {i}. - {title}", body) for i, (title, body) in enumerate(bodies, 1)]
        return "UGOVOR O DJELU", [("", intro), *articles]

    intro = (
        f"Concluded on {signed} in Zagreb between:\n\n"
        f"{c.client.name}, {c.client.address}, OIB: {c.client.oib} "
        f"(hereinafter: the Client)\n\nand\n\n"
        f"{c.contractor.name}, {c.contractor.address}, OIB: {c.contractor.oib} "
        f"(hereinafter: the Contractor)"
    )
    bodies = [
        (
            "Subject of the Agreement",
            "By this Agreement the Contractor undertakes to perform the following work "
            f"for the Client:\n{tasks}\n\nThe Contractor shall perform the work "
            "independently, diligently and in line with professional standards.",
        ),
        (
            "Fee",
            f"For the completed work the Client shall pay the Contractor a total fee "
            f"of {fee} (gross). The fee is payable within 15 days of the handover of "
            "the work, to the Contractor's bank account.",
        ),
        (
            "Term",
            f"The Contractor shall start the work on {start} and deliver it no later "
            f"than {end}. The term may only be extended by a written annex to this "
            "Agreement.",
        ),
        (
            "Obligations of the Client",
            "The Client shall provide the Contractor in due time with all data, "
            "documentation and access required for the work, and shall appoint a "
            "contact person authorised to give instructions and approve deliveries. "
            "Any delay by the Client in meeting these obligations extends the term "
            "accordingly.",
        ),
        (
            "Handover",
            "A handover report shall be signed by both parties. The Client shall "
            "inspect the work within eight days of receipt and notify the Contractor "
            "of any defects without delay. The Contractor shall remedy such defects "
            "within a reasonable time and without additional fee.",
        ),
        (
            "Confidentiality",
            "The Contractor shall keep confidential all data and information obtained "
            "while performing this Agreement, in particular information about the "
            "Client's business, customers and employees. This obligation survives "
            "for two years after the termination of this Agreement.",
        ),
        (
            "Copyright",
            "The Contractor transfers to the Client the exclusive economic copyright "
            "in the work, unlimited in time and territory. The consideration for the "
            "transfer is included in the fee under Article 2 of this Agreement.",
        ),
        (
            "Termination",
            "Either party may terminate this Agreement by written notice if the other "
            "party fails to meet its obligations within fifteen days of receiving a "
            "written reminder. In case of termination the Contractor is entitled to a "
            "proportionate part of the fee for the work performed until then.",
        ),
        (
            "Disputes",
            "The parties shall try to resolve any dispute amicably. Failing that, the "
            "competent court in Zagreb shall have jurisdiction.",
        ),
        (
            "Final Provisions",
            "Matters not regulated by this Agreement are governed by the Croatian "
            "Civil Obligations Act. The Agreement is made in two identical copies, one "
            "for each party, and enters into force on the day it is signed by both "
            "parties.",
        ),
    ]
    articles = [(f"Article {i} - {title}", body) for i, (title, body) in enumerate(bodies, 1)]
    return "SERVICE AGREEMENT", [("", intro), *articles]


def render_contract(c: Contract) -> bytes:
    pdf = new_pdf()
    title, sections = contract_articles(c)
    line(pdf, title, size=16, bold=True, height=10)
    pdf.ln(4)
    for heading, body in sections:
        if heading:
            pdf.ln(3)
            line(pdf, heading, size=11, bold=True, height=7)
        line(pdf, body, size=11, height=6)

    pdf.ln(16)
    client_label, contractor_label = (
        ("Za Naručitelja", "Izvršitelj") if c.language == "hr" else ("For the Client", "Contractor")
    )
    pdf.set_font("DejaVu", size=11)
    pdf.cell(85, 6, client_label)
    pdf.cell(85, 6, contractor_label, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(12)
    pdf.cell(85, 6, "______________________")
    pdf.cell(85, 6, "______________________", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(85, 6, c.client.name)
    pdf.cell(85, 6, c.contractor.name, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    return bytes(pdf.output())


# --- Fake scans -----------------------------------------------------------------


def scan_page(pdf_bytes: bytes, index: int, rng: random.Random, seed: int) -> bytes:
    """Render one page as a 'scan': grey, slightly rotated, noisy, blurred JPEG."""
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        pix = doc[index].get_pixmap(dpi=150)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    img = img.convert("L").rotate(rng.uniform(-1.5, 1.5), expand=True, fillcolor=255)
    noise = np.random.default_rng(seed + index).normal(0, 12, (img.height, img.width))
    arr = np.clip(np.asarray(img, dtype=np.float32) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(0.6))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return buf.getvalue()


def make_scans(pdf_bytes: bytes, rng: random.Random, seed: int) -> tuple[bytes, bytes]:
    """Return (JPEG of the first page, image-only PDF with every page scanned)."""
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        page_count = doc.page_count
    jpgs = [scan_page(pdf_bytes, i, rng, seed) for i in range(page_count)]

    out = pymupdf.open()
    for jpg in jpgs:
        page = out.new_page()  # A4 by default
        page.insert_image(page.rect, stream=jpg)
    scan_pdf = out.tobytes(no_new_id=True)  # a random /ID would break reproducibility
    out.close()
    return jpgs[0], scan_pdf


# --- Expected values ------------------------------------------------------------


def invoice_expected(inv: Invoice) -> dict[str, str]:
    return {
        "number": inv.number,
        "issuer": inv.issuer.name,
        "issuer_oib": inv.issuer.oib,
        "buyer": inv.buyer.name,
        "buyer_oib": inv.buyer.oib,
        "issue_date": inv.issue_date.isoformat(),
        "due_date": inv.due_date.isoformat(),
        "net": str(inv.net),
        "vat": str(inv.vat),
        "total": str(inv.total),
    }


def contract_expected(c: Contract) -> dict[str, str | list[str]]:
    return {
        "parties": [c.client.name, c.contractor.name],
        "fee": str(c.fee.quantize(CENT)),
        "start_date": c.start_date.isoformat(),
        "end_date": c.end_date.isoformat(),
        "language": c.language,
    }


# --- Main -----------------------------------------------------------------------


def generate(out: Path, seed: int) -> list[str]:
    rng = random.Random(seed)
    fake = Faker("hr_HR")
    fake.seed_instance(seed)
    out.mkdir(parents=True, exist_ok=True)

    files: dict[str, bytes] = {}
    expected: dict[str, dict[str, str] | dict[str, str | list[str]]] = {}

    for i in range(1, 4):
        inv = make_invoice(i, fake, rng)
        name = f"racun-{inv.number}"
        files[f"{name}.pdf"] = render_invoice(inv)
        expected[f"{name}.pdf"] = invoice_expected(inv)

    languages: tuple[Literal["hr", "en"], ...] = ("hr", "en")
    for i, language in enumerate(languages, start=1):
        contract = make_contract(language, fake, rng)
        name = f"ugovor-{i:02d}"
        files[f"{name}.pdf"] = render_contract(contract)
        expected[f"{name}.pdf"] = contract_expected(contract)

    for name in ("racun-2026-001", "ugovor-01"):
        jpg, scan_pdf = make_scans(files[f"{name}.pdf"], rng, seed)
        files[f"{name}-scan.jpg"] = jpg
        files[f"{name}-scan.pdf"] = scan_pdf
        expected[f"{name}-scan.jpg"] = expected[f"{name}-scan.pdf"] = expected[f"{name}.pdf"]

    for filename, content in files.items():
        (out / filename).write_bytes(content)
    (out / "expected.json").write_text(
        json.dumps(expected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return [*files, "expected.json"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--out", type=Path, default=Path("sample_docs"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for filename in generate(args.out, args.seed):
        size_kb = (args.out / filename).stat().st_size / 1024
        print(f"{filename:28} {size_kb:8.1f} KB")


if __name__ == "__main__":
    main()
