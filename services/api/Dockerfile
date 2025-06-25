# services/api/Dockerfile

# --- Stage 1: Builder ---
# This stage installs dependencies into a clean virtual environment.
FROM python:3.11 as builder
WORKDIR /app
RUN pip install uv
COPY pyproject.toml .
# Create venv and install dependencies defined in pyproject.toml
RUN uv venv && uv sync

# --- Stage 2: Final Image ---
# This is the lean, final image that will run in production.
FROM python:3.11-slim
WORKDIR /app

# Create a non-root user for enhanced security.
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# Copy the pre-built virtual environment from the builder stage.
COPY --from=builder /app/.venv ./.venv
# Add the venv to the system's PATH.
ENV PATH="/app/.venv/bin:$PATH"

# Copy the application source code into the final image.
# The app/app structure is a best practice for clean Python imports.
COPY ./app ./app

# Expose the port the Uvicorn server will run on.
EXPOSE 8000

# The command to run the Uvicorn server when the container starts.
# --host 0.0.0.0 makes the server accessible from outside the container.
# --port 8000 is the port inside the container.
# app.main:app tells Uvicorn to look for an object named 'app' in the 'main.py' file inside the 'app' module.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
