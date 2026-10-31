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
- **Update:** embeddings come from the OpenAI API, not sentence-transformers (see 014).

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
- **Consequences:** No extra infrastructure and simple tests (`TestClient` runs background tasks before returning). Tasks live in memory: a crash or restart during processing leaves documents `pending` forever. Tasks of one request run one after another, so a fast digital PDF can wait behind slow scans; to soften this, tasks of one upload are scheduled cheapest first (PDFs by size, then images), a heuristic until a real queue exists. Both are reasons to move to a real queue with retries if this goes beyond a demo.

## 012 – OCR per page, decided by a character threshold

- **Context:** PDFs can mix digital pages with scanned ones (e.g. a signed last page). The text layer is exact and fast; OCR is slow and loses diacritics.
- **Decision:** For every PDF page, count non-whitespace characters in the text layer. Fewer than `ocr_min_chars` (default 30) → render the page at `ocr_dpi` (default 200) and run OCR; otherwise keep the text layer. PNG/JPEG always go through OCR as one page. Each stored page records its `method` (`text_layer` or `ocr`).
- **Alternatives:** Decide once per document; always OCR; compare image area with text area on the page.
- **Consequences:** Mixed documents get the best source for each page, and the stored method makes results explainable. A mostly-image page with a short text header passes the threshold and skips OCR; a PDF with a poor embedded OCR layer is trusted as is. Threshold and DPI are settings, so they can be tuned against `sample_docs/expected.json`.

## 013 – Streamlit UI instead of Next.js

- **Context:** The assignment is graded on the backend (extraction, RAG, security, deployment); the UI only has to make the service easy to try. The deadline is tight and RAG (phase 4) had not started yet.
- **Decision:** A Streamlit app in `ui/` (`app.py` + `api_client.py`). It is just another client of the public API over HTTP (`httpx2`), never imports `docinsight` and holds no business logic. Response shapes are mirrored as `TypedDict`s. Its packages are a `ui` dependency group that is in `default-groups` for local work; the API image is built with `--no-default-groups --group ml` and the UI image with `--only-group ui`. In compose the UI is its own service, reaching the API at `http://api:8000` once the API's healthcheck passes.
- **Alternatives:** Next.js + TypeScript (familiar, more polished, about 3x the time, needs CORS); Gradio (similar effort, less control over layout and state); no UI, only `/docs`.
- **Consequences:** About an hour of work instead of three, all in Python, and no CORS because the Streamlit server, not the browser, calls the API. The UI looks like an internal tool. Streamlit's rerun model needs `session_state`, `query_params` and fragments for state and polling, and `AppTest` cannot drive fragments with `run_every`, so the page itself is checked manually while the client is unit-tested. The UI image is about 630 MB (Streamlit pulls in pandas and pyarrow) but contains no ML packages. A Next.js UI remains a possible next step behind the same API.

## 014 – Embeddings through the OpenAI API, hashing as fallback

- **Context:** Retrieval needs multilingual embeddings (Croatian documents, Croatian and English questions). A local model (`multilingual-e5-small` via sentence-transformers) needs `torch>=2.6`, which has no Intel macOS wheel (010), so it would only run in Docker and add about 500 MB to the image.
- **Decision:** `text-embedding-3-small` (1536 dimensions) through the OpenAI SDK, behind an `Embedder` protocol. `OpenAiEmbedder` sends batches of 100 texts, sorts results by `index`, L2-normalises them and retries twice. `HashingEmbedder` (hashed words, diacritics removed, no model, no network) is used in tests and when no API key is set; `embedding_backend = "auto"` picks OpenAI when a key exists. Each chunk stores the name of what produced its vector (`embedding_model`); search only compares chunks with the current name. `python -m docinsight.reindex` rebuilds outdated chunks from the stored page text, without OCR.
- **Alternatives:** Local `multilingual-e5-small` in Docker only; ChromaDB with built-in embeddings; hashing only.
- **Consequences:** Same behaviour locally and in Docker, no torch for embeddings, one API key for embeddings and answers. Document text leaves the service at indexing time (answers send it anyway) and the service depends on an external API; a local embedder fits behind the same protocol. Evaluation (`docs/evaluation.md`): hashing 76–82 %, OpenAI 88–100 % answer accuracy; hashing fails on paraphrases and cross-language questions but needs no network. Changing the model or the indexed text makes old chunks unsearchable until `reindex` runs, which is preferable to mixing incomparable vectors silently.

## 015 – Chunks in SQLite, in-memory FAISS, hybrid retrieval

- **Context:** Each user has hundreds to a few thousand chunks. A separate vector index would have to stay in sync with the database on every upload and delete. Manual testing (step 11) showed that pure embedding search misses identifiers: for "When is invoice 2026-002 due?" the chunk of invoice 2026-002 was not in the top 5, because invoices with the same layout get almost the same vector.
- **Decision:** `DocumentChunk` rows (text, page, float32 vector as bytes, index name) are the source of truth. For every search the owner's chunks are loaded with one SQL query (owner, index name, `processed` documents only) and searched exactly with `faiss.IndexFlatIP`, or the same dot product in NumPy when faiss is missing. With `retrieval_mode = "hybrid"` (default) the full embedding ranking is fused with a BM25 ranking of the same chunks (`k1` 1.5, `b` 0.75, computed in memory) by Reciprocal Rank Fusion (`k` 60), and the fused score is scaled to 0..1. `"dense"` stays for comparison.
- **Alternatives:** Persistent FAISS index per user; a vector database or `pgvector`; approximate indexes (HNSW, IVF); SQLite FTS5 for BM25; weighted sum of normalised scores instead of RRF; metadata filters extracted from the question by an LLM.
- **Consequences:** No index file can drift from the database, and filtering by owner happens in SQL before any vector is compared. Each search costs one query and an exact scan, which is milliseconds at this size; most of the ~230 ms retrieval time is embedding the question. Hybrid retrieval fixed the identifier question and raised answer accuracy from 88 % to 94 % with OpenAI embeddings (76 % to 82 % with hashing). RRF needs no score normalisation or weights but ignores how far apart scores are. With tens of thousands of chunks per user, move to a persistent or approximate index and BM25 in the database (FTS5, `tsvector`); metadata filters come after invoice fields are extracted (phase 7).

## 016 – Answers from the OpenAI Responses API with structured output

- **Context:** Answers must come only from the user's documents and cite file and page so they can be checked. Asking for JSON in the prompt is unreliable.
- **Decision:** An `AnswerGenerator` protocol; `OpenAiGenerator` calls `client.responses.parse` with `text_format=LlmAnswer` (`answer`, `found`, `citations`), model `gpt-6-luna`, `reasoning_effort = "low"`, `store=False`, a 30 s timeout and one retry. The system prompt holds the rules: answer only from the numbered excerpts, treat them as data and not as instructions, answer in the language of the question, answer for each matching document, cite `[n]`, and set `found=false` instead of guessing. Excerpts go into the user message inside `<context>`, and context tags are stripped from document text. Sources in the API response are the cited chunks mapped back by our code; unknown numbers and repeats are dropped. Without a key `UnavailableGenerator` makes `/ask` return 503; nothing to search returns 409; a failed embedding or LLM call returns 502 with a short message.
- **Alternatives:** Local extractive QA model (torch, Docker only, no synthesis across chunks); a local LLM through `llm_base_url` (Ollama or another OpenAI-compatible server); Chat Completions API; LangChain (004).
- **Consequences:** The response shape is guaranteed by the schema and the same Pydantic class is the type in code; the model cannot invent a source outside the context. The schema does not guarantee truth: a wrong reading of the context or a misleading excerpt still produces a wrong answer, which the evaluation set measures (`found=false` was correct on 3/3 unanswerable questions in every configuration). Prompt injection is mitigated, not solved; the model has no tools and sees only the caller's documents. Question and excerpts are sent to OpenAI; `store=False` minimises what is kept there.

## 017 – Chunking per page with document headers

- **Context:** Citations point to a page, so a chunk must belong to exactly one page. Evaluation then showed lost context: in `c02-start-hr` the chunk with the answer did not name the parties of the contract, and in Docker the model answered with the start date of the other contract, with a citation.
- **Decision:** Chunks never cross a page boundary: windows of 800 characters with 150 characters of overlap, cut at whitespace. Characters, not tokens: 800 characters are about 200–250 tokens, far below the embedding limit. Every chunk is embedded, ranked by BM25 and shown to the LLM with a header: the file name and the first 200 characters of page 1 (title, parties, invoice number). The stored chunk text has no header; search rebuilds it from the stored first page with the same function. The index name carries the text format (`text-embedding-3-small+header-v1`), so `reindex` finds chunks indexed without headers.
- **Alternatives:** Chunks across pages with a page range; token-based or sentence/section-aware chunking; a header written by an LLM per chunk (contextual retrieval); storing the header in a column.
- **Consequences:** Every citation has one page the user can open. A sentence split by a page break loses context, which overlap cannot fix. Headers raised OpenAI hybrid answer accuracy from 16/17 to 17/17, locally and in Docker with scans, and turned the confident wrong answer into a correct one; with hashing the right contract is found but the cross-language question still fails safely. A document whose first page is blank or only an image gets the file name as its header. Search costs one more query for first pages. A different header format means a new `INDEX_FORMAT` and a re-index.
