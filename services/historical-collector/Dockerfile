FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY app/ ./app/

# Install dependencies using uv
RUN uv pip install --system --no-cache -e .

# Set the entry point
CMD ["python", "-m", "app.main"]
