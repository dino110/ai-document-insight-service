# syntax=docker/dockerfile:1
# Streamlit UI: a small image with only the `ui` dependency group (no torch, no docinsight).
# Build context is the repository root (it needs pyproject.toml and uv.lock).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.6 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# The UI only talks to the API over HTTP: install the `ui` group alone, without the
# docinsight package or its dependencies (FastAPI, PyMuPDF, SQLModel...).
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --only-group ui

COPY ui ./ui

RUN useradd --create-home --uid 1000 app
USER app

EXPOSE 8501
# 0.0.0.0: reachable from outside the container; gatherUsageStats=false: no telemetry.
CMD ["streamlit", "run", "ui/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--browser.gatherUsageStats=false"]
