"""Evaluate GET /documents/{id}/fields on the sample documents against a running API.

    uv run python scripts/evaluate_fields.py --api http://localhost:8000 --label "openai"

Uploads the sample invoices and contracts (digital and scanned) into a fresh session, waits
until they are processed and compares the type and every extracted field with
sample_docs/expected.json: amounts as numbers, dates as dates, OIBs and IBANs without
spaces, names exactly. It also reports the checks that failed, whether a summary came back,
and whether local NER found the identifiers and amounts among the document's entities.
Each run calls the LLM once per document (new session, nothing cached).
"""

import argparse
import json
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Literal
from uuid import uuid4

import httpx2
from evaluate import SAMPLE_DIR, percentile, send, upload_and_wait

Kind = Literal["invoice", "contract"]

FIELDS: dict[Kind, list[str]] = {
    "invoice": [
        "number",
        "issuer",
        "issuer_oib",
        "issuer_iban",
        "buyer",
        "buyer_oib",
        "issue_date",
        "due_date",
        "net",
        "vat",
        "total",
    ],
    # The contract fields expected.json knows.
    "contract": ["client", "contractor", "start_date", "end_date", "fee"],
}
AMOUNTS = {"net", "vat", "total", "fee"}
DATES = {"issue_date", "due_date", "signed_date", "start_date", "end_date"}
IDENTIFIERS = {"issuer_oib", "buyer_oib", "issuer_iban", "client_oib", "contractor_oib"}
# Field -> label of the entity local NER should have found for its expected value.
NER_TARGETS: dict[Kind, dict[str, str]] = {
    "invoice": {"issuer_oib": "OIB", "buyer_oib": "OIB", "issuer_iban": "IBAN", "total": "MONEY"},
    "contract": {"fee": "MONEY"},
}


@dataclass(frozen=True)
class DocumentResult:
    filename: str
    kind: Kind
    status_code: int
    type_ok: bool
    summary_chars: int
    fields_ok: dict[str, bool]  # field -> extracted value matches expected.json
    failed_checks: list[str]
    ner_found: dict[str, bool]  # field -> its expected value is among the entities
    ms: float


def field_matches(name: str, got: str | None, expected: str) -> bool:
    """Compare as the field's type: "1575" == "1575.00", grouped IBAN == compact IBAN."""
    if got is None:
        return False
    try:
        if name in AMOUNTS:
            return Decimal(got) == Decimal(expected)
        if name in DATES:
            return date.fromisoformat(got) == date.fromisoformat(expected)
    except (InvalidOperation, ValueError):
        return False
    if name in IDENTIFIERS:
        return got.replace(" ", "").upper() == expected
    return got.strip() == expected


def expected_values(kind: Kind, expected: dict[str, object]) -> dict[str, str]:
    """expected.json in field names: a contract lists its parties as [client, contractor]."""
    values = {k: v for k, v in expected.items() if isinstance(v, str)}
    if kind == "contract":
        parties = expected["parties"]
        if not isinstance(parties, list) or len(parties) != 2:
            raise ValueError("a contract in expected.json needs two parties")
        values |= {"client": parties[0], "contractor": parties[1]}
    return values


def _entity_value(label: str, expected: str) -> str:
    """The expected value in the form of the API's ``normalized`` entities."""
    return str(Decimal(expected).quantize(Decimal("0.01"))) if label == "MONEY" else expected


def evaluate_document(
    client: httpx2.Client,
    session_id: str,
    filename: str,
    document_id: str,
    expected: dict[str, object],
) -> DocumentResult:
    kind: Kind = "invoice" if filename.startswith("racun-") else "contract"
    values = expected_values(kind, expected)
    headers = {"X-Session-ID": session_id}
    response = send(lambda: client.get(f"/documents/{document_id}/fields", headers=headers))
    ms = response.elapsed.total_seconds() * 1000
    if response.status_code != 200:
        return DocumentResult(filename, kind, response.status_code, False, 0, {}, [], {}, ms)
    body = response.json()
    fields = body[kind] or {}
    detail = client.get(f"/documents/{document_id}", headers=headers).json()
    entities = {(e["label"], e["normalized"]) for p in detail["pages"] for e in p["entities"]}
    return DocumentResult(
        filename=filename,
        kind=kind,
        status_code=200,
        type_ok=body["document_type"] == kind,
        summary_chars=len(body["summary"]),
        fields_ok={f: field_matches(f, fields.get(f), values[f]) for f in FIELDS[kind]},
        failed_checks=[name for name, passed in body["checks"].items() if passed is False],
        ner_found={
            f: (label, _entity_value(label, values[f])) in entities
            for f, label in NER_TARGETS[kind].items()
        },
        ms=ms,
    )


def _ratio(ok: int, total: int) -> str:
    return f"{ok}/{total} ({ok / total:.0%})" if total else "-"


def report(label: str, results: list[DocumentResult]) -> str:
    done = [r for r in results if r.status_code == 200]
    lines = [
        f"### Field extraction: {label}",
        "",
        f"{len(results)} documents, {len(done)} answered with 200.",
        "",
        "| Field | Correct |",
        "| --- | --- |",
    ]
    for kind, names in FIELDS.items():
        of_kind = [r for r in done if r.kind == kind]
        lines += [
            f"| {kind}: {f} | {_ratio(sum(r.fields_ok[f] for r in of_kind), len(of_kind))} |"
            for f in names
        ]
    all_fields = sum(sum(r.fields_ok.values()) for r in done)
    field_count = sum(len(r.fields_ok) for r in done)
    ner = sum(sum(r.ner_found.values()) for r in done)
    ner_count = sum(len(r.ner_found) for r in done)
    lines += [
        f"| **all fields** | {_ratio(all_fields, field_count)} |",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Document type | {_ratio(sum(r.type_ok for r in done), len(done))} |",
        f"| Summary returned | {_ratio(sum(r.summary_chars > 0 for r in done), len(done))} |",
        f"| Documents with a failed check | {sum(bool(r.failed_checks) for r in done)} |",
        f"| Identifiers and amounts among the entities (local NER) | {_ratio(ner, ner_count)} |",
        f"| Extraction p50 / max | {percentile([r.ms for r in done], 50):.0f} / "
        f"{max((r.ms for r in done), default=0):.0f} ms |",
        "",
        "| Document | Fields | Failed checks | Wrong fields |",
        "| --- | --- | --- | --- |",
    ]
    for r in results:
        if r.status_code != 200:
            lines.append(f"| {r.filename} | HTTP {r.status_code} | - | - |")
            continue
        wrong = ", ".join(f for f, ok in r.fields_ok.items() if not ok) or "-"
        if not r.type_ok:
            wrong = f"type, {wrong}"
        lines.append(
            f"| {r.filename} | {sum(r.fields_ok.values())}/{len(r.fields_ok)} "
            f"| {', '.join(r.failed_checks) or '-'} | {wrong} |"
        )
    return "\n".join(lines)


def documents(include_scans: bool) -> dict[str, dict[str, object]]:
    expected: dict[str, dict[str, object]] = json.loads(
        (SAMPLE_DIR / "expected.json").read_text(encoding="utf-8")
    )
    return {
        name: values for name, values in expected.items() if include_scans or "-scan" not in name
    }


def run(
    client: httpx2.Client, *, label: str, include_scans: bool = True, timeout_s: float = 900
) -> tuple[str, list[DocumentResult]]:
    """Upload the documents, extract their fields and return the report and the results."""
    session_id = f"eval-fields-{uuid4().hex[:12]}"
    expected = documents(include_scans)
    upload_and_wait(client, session_id, list(expected), timeout_s)
    listed = client.get("/documents", headers={"X-Session-ID": session_id}).json()
    ids = {d["filename"]: d["id"] for d in listed}
    results = [
        evaluate_document(client, session_id, name, ids[name], values)
        for name, values in expected.items()
    ]
    return report(label, results), results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate document field extraction.")
    parser.add_argument("--api", default="http://localhost:8000", help="base URL of the API")
    parser.add_argument("--label", default="", help="name of this run in the report")
    parser.add_argument("--no-scans", action="store_true", help="upload digital documents only")
    parser.add_argument("--out", type=Path, help="append the report to this file")
    args = parser.parse_args(argv)

    with httpx2.Client(base_url=args.api, timeout=180) as client:
        text, _ = run(client, label=args.label or args.api, include_scans=not args.no_scans)
    print(text)
    if args.out:
        with args.out.open("a", encoding="utf-8") as out:
            out.write("\n" + text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
