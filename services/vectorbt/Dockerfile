# Multi-stage build for technical indicators service
FROM python:3.11.11-slim AS builder

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libffi-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy dependency files AND the readme
# The readme is required by hatchling during the build process as specified in pyproject.toml
COPY pyproject.toml ./
COPY README.md ./

# Install dependencies
# This step builds the local project, which is why README.md is needed
RUN uv venv && uv sync

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libopenblas0 \
    liblapack3 \
    libgfortran5 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY app/ ./app/

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=15s --start-period=5s --retries=3 \
    CMD python -c "from app.database import db_manager; print('Health check passed')" || exit 1

# Run the application
CMD ["python", "-m", "app.main"]
