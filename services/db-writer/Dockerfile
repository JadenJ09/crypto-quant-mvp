# ==============================================================================
# File: services/db-writer/Dockerfile
# ==============================================================================
# Containerizes the db-writer application.

# --- Stage 1: Builder ---
FROM python:3.11 AS builder
WORKDIR /app
RUN pip install uv
COPY pyproject.toml .
RUN uv venv .venv && uv sync

# --- Stage 2: Final Image ---
FROM python:3.11-slim
WORKDIR /app
RUN useradd --create-home --shell /bin/bash appuser
USER appuser
COPY --from=builder /app/.venv ./.venv
ENV PATH="/app/.venv/bin:$PATH"
COPY ./app ./app
CMD ["python", "app/main.py"]