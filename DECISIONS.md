# Decision log

Short records of non-trivial decisions: context, decision, alternatives, consequences.

## 001 – uv for packaging and environments

- **Context:** Need reproducible installs on macOS, Linux CI and Docker.
- **Decision:** uv with `pyproject.toml` and `uv.lock`.
- **Alternatives:** pip + venv + requirements.txt, Poetry.
- **Consequences:** One fast tool for Python versions, venvs and locking; CI and Docker use `uv sync --frozen`.

## 002 – src layout, package `docinsight`

- **Context:** Tests must import the installed package, not files from the working directory by accident.
- **Decision:** `src/docinsight/`, created with `uv init --package`.
- **Alternatives:** Flat layout.
- **Consequences:** Imports are always `from docinsight...`; tests run against the same code that ships.

## 003 – Strict quality gates

- **Context:** Code is partly AI-assisted and must be verifiable.
- **Decision:** ruff (lint + format), mypy strict and pytest must pass before every commit (run manually, no pre-commit hooks for now) and in CI.
- **Alternatives:** Lint only, or no type checking.
- **Consequences:** Slightly slower iteration, much higher confidence in generated code.

## 004 – No LangChain (for now)

- **Context:** The pipeline (extract → chunk → embed → retrieve → generate) is small.
- **Decision:** Implement it directly with sentence-transformers, FAISS and the OpenAI SDK.
- **Alternatives:** LangChain / LlamaIndex.
- **Consequences:** Every step is visible and testable; fewer dependencies. Can be swapped later behind the same protocols.

## 005 – SQLite + SQLModel for metadata

- **Context:** Need to store document metadata (owner, filename, type, size, hash, path) with minimal setup.
- **Decision:** SQLite file in `data/app.db`, accessed through SQLModel (SQLAlchemy + Pydantic). Engine is created once in `lifespan`; one session per request via a `yield` dependency.
- **Alternatives:** PostgreSQL from day one; raw `sqlite3`; JSON files.
- **Consequences:** Zero infrastructure and easy tests (a fresh DB per test in `tmp_path`). SQLite allows one writer at a time and drops timezones, which is fine for a single-instance demo. Moving to Postgres means turning `database_url` into a configurable setting and adding migrations (Alembic); models and endpoints stay the same.

## 006 – File type from magic bytes

- **Context:** Only PDF, PNG and JPEG are accepted. Filename and `Content-Type` are both chosen by the client and trivial to fake.
- **Decision:** Detect the type from the first bytes of the content (`%PDF-`, PNG signature, `\xff\xd8\xff`) in `core/files.py`. Files are stored under a generated name; the client filename is kept only in the database.
- **Alternatives:** Trust the extension or header; `python-magic` / `filetype` packages.
- **Consequences:** No extra dependency and no path traversal through filenames. A valid signature does not prove the file is well formed; the extraction step (phase 2) still has to handle broken files.

## 007 – `X-Session-ID` as a temporary owner

- **Context:** Documents must be isolated per user before authentication exists (phase 6).
- **Decision:** The owner is the `X-Session-ID` header. `/upload` creates a new id if it is missing and returns it; `/documents` requires it. The header is restricted to `[A-Za-z0-9_-]{1,64}` because it becomes a directory name. All of this lives in `api/deps.py`.
- **Alternatives:** Implement JWT immediately; cookies.
- **Consequences:** Anyone who knows a session id can read its documents, so this is not security, only isolation. Switching to JWT replaces the dependency, not the endpoints.

## 008 – Sync `def` endpoints for blocking I/O

- **Context:** Upload writes files to disk and uses the synchronous SQLite driver.
- **Decision:** `/upload` and `/documents` are plain `def`; FastAPI runs them in its threadpool. Uploaded content is read with `file.file.read(limit + 1)`.
- **Alternatives:** `async def` with `aiofiles` and an async DB driver (`aiosqlite`); `async def` with `run_in_threadpool` around blocking calls.
- **Consequences:** Blocking calls never stall the event loop. Concurrency is bounded by the threadpool size (about 40), which is plenty here. Ruff's `ASYNC` rules only catch blocking calls they can recognise by type, e.g. `Path.write_bytes` on a variable annotated as `Path`. Checked on this code: with these endpoints switched to `async def`, ruff reports nothing, because it cannot infer the type of `settings.upload_dir / ...` and does not know `UploadFile.file.read()` or a sync DB commit block. The rule is a safety net, not a guarantee.

## 009 – PyMuPDF pinned to `>=1.28,<2`

- **Context:** PDF text extraction and page rendering need a fast, well-maintained library. Development happens on an Intel Mac (x86_64); PyMuPDF 2.0 ships no wheel for that platform.
- **Decision:** `pymupdf>=1.28,<2` as a runtime dependency, the same version locally and in Docker. Code that calls it stays in `extraction/` (and the sample generator); mypy uses `untyped_calls_exclude = ["pymupdf"]` and results are converted to typed values at the boundary (`str(page.get_text())`).
- **Alternatives:** pypdf (pure Python, but no page rendering for OCR); pdfplumber (built on pdfminer, slower); building PyMuPDF 2.x from source.
- **Consequences:** One library covers both the text layer and rendering pages to images. Upgrading to 2.x is blocked until there is an Intel macOS wheel or development moves off this machine. PyMuPDF is AGPL; fine for this assignment, a commercial product would need a licence or a different library.

## 010 – ML packages only in the Linux container

- **Context:** PyTorch has no Intel macOS wheel after 2.2.2, while current `transformers`/`easyocr` need `torch>=2.6`. PyTorch from PyPI on Linux is a CUDA build that pulls several GB of `nvidia-*` wheels.
- **Decision:** torch, torchvision and easyocr live in a `ml` dependency group with the marker `sys_platform == 'linux'`. torch and torchvision come from the CPU-only PyTorch index (`[[tool.uv.index]]` with `explicit = true`, mapped in `[tool.uv.sources]`). OCR sits behind the `OcrEngine` protocol: locally `UnavailableOcrEngine` (and `FakeOcrEngine` in tests), in Docker `EasyOcrEngine`, chosen by `create_ocr_engine` from `ocr_enabled` and whether `easyocr` imports. Slow tests with the real model are marked `slow` and run in the `test` image stage.
- **Alternatives:** Old torch 2.2.2 natively (blocks newer libraries); a cloud OCR API (cost, data leaves the service); Tesseract (system dependency, weaker on noisy scans); developing entirely inside a dev container.
- **Consequences:** One `uv.lock` serves macOS and Linux. Local development and the fast test suite need no ML packages. The image is about 1.75 GB (CUDA build would be 5+ GB), models are downloaded at build time. Real OCR can only be exercised in Docker; local scans end as `failed` with "OCR is not available".

## 011 – Background processing with `BackgroundTasks` and a status field

- **Context:** OCR takes tens of seconds per page on CPU (measured 17–50 s), far too long to run inside the upload request.
- **Decision:** `/upload` stores the files, commits, returns `201` with `status="pending"` and schedules one `process_document` task per document after the commit. The task opens its own database session (the request's session is closed by then), stores text per page and sets `processed` or `failed`. Any exception is logged and turned into a short, user-facing `error`; raw exception text is never returned. Clients poll `GET /documents/{id}`.
- **Alternatives:** Synchronous processing (HTTP timeouts); Celery/RQ/arq with Redis (broker and worker process to run and monitor); a separate worker reading `pending` rows from the database.
- **Consequences:** No extra infrastructure and simple tests (`TestClient` runs background tasks before returning). Tasks live in memory: a crash or restart during processing leaves documents `pending` forever. Tasks of one request run one after another, so a fast digital PDF can wait behind slow scans. Both are reasons to move to a real queue with retries if this goes beyond a demo.

## 012 – OCR per page, decided by a character threshold

- **Context:** PDFs can mix digital pages with scanned ones (e.g. a signed last page). The text layer is exact and fast; OCR is slow and loses diacritics.
- **Decision:** For every PDF page, count non-whitespace characters in the text layer. Fewer than `ocr_min_chars` (default 30) → render the page at `ocr_dpi` (default 200) and run OCR; otherwise keep the text layer. PNG/JPEG always go through OCR as one page. Each stored page records its `method` (`text_layer` or `ocr`).
- **Alternatives:** Decide once per document; always OCR; compare image area with text area on the page.
- **Consequences:** Mixed documents get the best source for each page, and the stored method makes results explainable. A mostly-image page with a short text header passes the threshold and skips OCR; a PDF with a poor embedded OCR layer is trusted as is. Threshold and DPI are settings, so they can be tuned against `sample_docs/expected.json`.
