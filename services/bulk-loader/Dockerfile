FROM python:3.11-slim AS builder

WORKDIR /app

# Install uv for fast dependency management
RUN pip install uv

# Copy dependency files
COPY pyproject.toml .

# Install dependencies in virtual environment
RUN uv venv .venv && uv sync

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Copy virtual environment from builder
COPY --from=builder /app/.venv ./.venv

# Copy application code
COPY ./app ./app

# Set ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

# Run the application
CMD ["python", "app/main.py"]
