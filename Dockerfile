# The delphi-rag API (api/app.py) for Google Cloud Run.
# Build context excludes secrets and data: see .dockerignore.
FROM python:3.13-slim

# uv, pinned (same version as development).
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1

# Dependencies first, so code changes don't reinstall them.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY agent/ agent/
COPY api/ api/
COPY core/ core/

# Run as a non-root user.
RUN useradd --create-home app
USER app

# Cloud Run sets PORT.
CMD ["sh", "-c", "exec /app/.venv/bin/uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
