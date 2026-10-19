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
