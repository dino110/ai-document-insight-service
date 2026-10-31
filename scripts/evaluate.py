"""Evaluate POST /ask on sample_docs/eval_questions.json against a running API.

    uv run python scripts/evaluate.py --api http://localhost:8000 --label "openai, hybrid"

Uploads the sample documents (digital and scanned) into a fresh session, waits until they
are processed, asks every question and prints a markdown report: answer accuracy,
citation accuracy (expected file and page among the sources), correct "not found" for
questions the documents cannot answer, and latency. The embedder and retrieval mode are
settings of the server, so compare runs against servers started with different settings.
"""

import argparse
import json
import math
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict
from uuid import uuid4

import httpx2
from pydantic import TypeAdapter

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_docs"
CONTENT_TYPES = {".pdf": "application/pdf", ".jpg": "image/jpeg", ".png": "image/png"}


class EvalQuestion(TypedDict):
    id: str
    question: str
    expected: list[str] | None  # acceptable spellings of the answer; None = not in the documents
    document: str | None
    page: int | None


@dataclass(frozen=True)
class QuestionResult:
    id: str
    answerable: bool
    status_code: int
    answer: str
    found: bool
    sources: list[tuple[str, int]]
    answer_ok: bool  # unanswerable: the API said found=false
    citation_ok: bool | None  # None for unanswerable questions
    retrieval_ms: float
    generation_ms: float
    total_ms: float
    model: str


# --- Matching -----------------------------------------------------------------------


def normalize(text: str) -> str:
    """Lowercase, no diacritics, no whitespace: "1.575, 00 EUR" ~ "1.575,00 eur"."""
    folded = unicodedata.normalize("NFKD", text.lower().replace("đ", "d"))
    return "".join(c for c in folded if not unicodedata.combining(c) and not c.isspace())


def base_document(filename: str) -> str:
    """A scan answers like its original: "racun-2026-001-scan.jpg" -> "racun-2026-001"."""
    return Path(filename).stem.removesuffix("-scan")


def answer_matches(answer: str, expected: list[str]) -> bool:
    return any(normalize(option) in normalize(answer) for option in expected)


def citation_matches(sources: list[tuple[str, int]], document: str, page: int) -> bool:
    return (base_document(document), page) in {(base_document(f), p) for f, p in sources}


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile; 0.0 for no values."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q / 100 * len(ordered)) - 1)]


# --- Talking to the API -------------------------------------------------------------


def load_questions(path: Path) -> list[EvalQuestion]:
    return TypeAdapter(list[EvalQuestion]).validate_json(path.read_bytes())


def sample_documents(include_scans: bool) -> list[str]:
    names = list(json.loads((SAMPLE_DIR / "expected.json").read_text(encoding="utf-8")))
    return [n for n in names if include_scans or "-scan" not in n]


def upload_and_wait(
    client: httpx2.Client, session_id: str, filenames: list[str], timeout_s: float
) -> dict[str, str]:
    """Upload the files and poll until none is pending; return filename -> status."""
    headers = {"X-Session-ID": session_id}
    files = [
        ("files", (n, (SAMPLE_DIR / n).read_bytes(), CONTENT_TYPES[Path(n).suffix]))
        for n in filenames
    ]
    response = client.post("/upload", files=files, headers=headers)
    response.raise_for_status()
    deadline = time.monotonic() + timeout_s
    while True:
        documents = client.get("/documents", headers=headers).json()
        if all(d["status"] != "pending" for d in documents) or time.monotonic() > deadline:
            return {d["filename"]: d["status"] for d in documents}
        time.sleep(1)


def ask(client: httpx2.Client, session_id: str, question: EvalQuestion) -> QuestionResult:
    started = time.perf_counter()
    response = client.post(
        "/ask", json={"question": question["question"]}, headers={"X-Session-ID": session_id}
    )
    total_ms = (time.perf_counter() - started) * 1000
    body = response.json() if response.status_code == 200 else {}
    answer = str(body.get("answer", f"HTTP {response.status_code}"))
    found = bool(body.get("found", False))
    sources = [(s["filename"], s["page_number"]) for s in body.get("sources", [])]
    timings = body.get("timings_ms", {})

    expected, document, page = question["expected"], question["document"], question["page"]
    answerable = expected is not None
    if expected is None:
        answer_ok, citation_ok = response.status_code == 200 and not found, None
    else:
        answer_ok = found and answer_matches(answer, expected)
        citation_ok = (
            document is not None
            and page is not None
            and (citation_matches(sources, document, page))
        )
    return QuestionResult(
        id=question["id"],
        answerable=answerable,
        status_code=response.status_code,
        answer=answer,
        found=found,
        sources=sources,
        answer_ok=answer_ok,
        citation_ok=citation_ok,
        retrieval_ms=float(timings.get("retrieval", 0.0)),
        generation_ms=float(timings.get("generation", 0.0)),
        total_ms=total_ms,
        model=str(body.get("model", "")),
    )


# --- Report -------------------------------------------------------------------------


def _ratio(ok: int, total: int) -> str:
    return f"{ok}/{total} ({ok / total:.0%})" if total else "-"


def _mark(value: bool | None) -> str:
    return "-" if value is None else ("✓" if value else "✗")


def report(label: str, statuses: dict[str, str], results: list[QuestionResult]) -> str:
    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]
    models = sorted({r.model for r in results if r.model}) or ["-"]
    not_processed = sorted(n for n, s in statuses.items() if s != "processed")

    answers = _ratio(sum(r.answer_ok for r in answerable), len(answerable))
    citations = _ratio(sum(bool(r.citation_ok) for r in answerable), len(answerable))
    not_found = _ratio(sum(r.answer_ok for r in unanswerable), len(unanswerable))

    def latency(values: list[float]) -> str:
        return f"{percentile(values, 50):.0f} / {percentile(values, 95):.0f} ms"

    lines = [
        f"### Evaluation: {label}",
        "",
        f"{len(results)} questions ({len(answerable)} answerable, {len(unanswerable)} not), "
        f"model {', '.join(models)}, {len(statuses)} documents uploaded"
        + (f", not processed: {', '.join(not_processed)}" if not_processed else "")
        + ".",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Answer accuracy | {answers} |",
        f"| Citation accuracy (expected file and page among sources) | {citations} |",
        f'| Correct "not found" | {not_found} |',
        f"| Retrieval p50 / p95 | {latency([r.retrieval_ms for r in results])} |",
        f"| Generation p50 / p95 | {latency([r.generation_ms for r in results])} |",
        f"| Total (client) p50 / p95 | {latency([r.total_ms for r in results])} |",
        "",
        "| Question | Answer | Citation | Sources | Response |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in results:
        sources = ", ".join(f"{f} p{p}" for f, p in r.sources) or "-"
        text = r.answer.replace("|", "\\|").replace("\n", " ")
        text = text if len(text) <= 90 else text[:87] + "..."
        lines.append(
            f"| {r.id} | {_mark(r.answer_ok)} | {_mark(r.citation_ok)} | {sources} | {text} |"
        )
    return "\n".join(lines)


def run(
    client: httpx2.Client,
    questions: list[EvalQuestion],
    *,
    label: str,
    include_scans: bool = True,
    timeout_s: float = 900,
) -> tuple[str, list[QuestionResult]]:
    """Upload, ask everything and return the markdown report and the raw results."""
    session_id = f"eval-{uuid4().hex[:12]}"
    statuses = upload_and_wait(client, session_id, sample_documents(include_scans), timeout_s)
    results = [ask(client, session_id, question) for question in questions]
    return report(label, statuses, results), results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate /ask on the sample documents.")
    parser.add_argument("--api", default="http://localhost:8000", help="base URL of the API")
    parser.add_argument("--label", default="", help="name of this run in the report")
    parser.add_argument(
        "--questions", type=Path, default=SAMPLE_DIR / "eval_questions.json", help="question set"
    )
    parser.add_argument("--no-scans", action="store_true", help="upload digital documents only")
    parser.add_argument("--out", type=Path, help="also write the report to this file")
    args = parser.parse_args(argv)

    questions = load_questions(args.questions)
    # Long timeout: scans go through OCR on CPU before the first question can be asked.
    with httpx2.Client(base_url=args.api, timeout=180) as client:
        text, _ = run(
            client, questions, label=args.label or args.api, include_scans=not args.no_scans
        )
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
