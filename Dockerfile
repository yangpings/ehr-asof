# syntax=docker/dockerfile:1
# --- build stage: resolve and install dependencies into a virtualenv ---
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app
# Dependency layer first: it only rebuilds when the lock file changes.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project --extra postgres
COPY src ./src
RUN uv sync --locked --no-dev --extra postgres

# --- runtime stage: only the virtualenv and source, as a non-root user ---
FROM python:3.12-slim AS runtime
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=build --chown=app:app /app /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live')"
CMD ["uvicorn", "ehr_asof.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]