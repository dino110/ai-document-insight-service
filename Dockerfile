# syntax=docker/dockerfile:1
# API image with real OCR (EasyOCR + CPU-only PyTorch).
#   docker build --target runtime -t docinsight-api .   production image (default: last stage)
#   docker build --target test -t docinsight-test .     same image plus dev tools and tests

FROM python:3.12-slim AS base

COPY --from=ghcr.io/astral-sh/uv:0.12.6 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    EASYOCR_MODULE_PATH=/opt/easyocr

WORKDIR /app

# 1. Dependencies only: this heavy layer is rebuilt only when pyproject.toml or uv.lock change.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --group ml --no-install-project

# 2. OCR models (detector + Latin recognizer), so the first request does not download them.
RUN python -c "import easyocr; easyocr.Reader(['hr', 'en'], gpu=False)"

# 3. Application code.
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --group ml


# Tests inside the container, including the slow ones that need real OCR.
# Runs as root: it is never deployed and pytest writes caches into /app.
FROM base AS test

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --group dev --group ml
COPY tests ./tests
COPY sample_docs ./sample_docs
# pytest directly, not `uv run`: that would re-sync to the default groups and drop `ml`.
CMD ["pytest", "-m", "slow"]


FROM base AS runtime

# Run as an unprivileged user that can only write to the data directory.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/data \
    && chown app:app /app/data
USER app

EXPOSE 8000
CMD ["uvicorn", "docinsight.main:app", "--host", "0.0.0.0", "--port", "8000"]
