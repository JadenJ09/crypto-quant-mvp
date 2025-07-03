# Dockerfile for Custom Backtesting Engine
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY src/ ./src/
COPY tests/ ./tests/
COPY examples/ ./examples/
COPY *.md ./

# Set Python path
ENV PYTHONPATH=/app/src

# Expose the API port
EXPOSE 8003

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8003/health/ || exit 1

# Default command to run the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8003"]
