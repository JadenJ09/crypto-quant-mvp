# services/ingestor/Dockerfile

# --- Stage 1: Builder ---
# This stage's only purpose is to install our Python dependencies into a clean venv.
# We use a full python image here because it contains the build tools needed by some libraries.
ARG BASE_IMAGE=python:3.11
FROM ${BASE_IMAGE}-slim AS builder

# # Set the working directory
WORKDIR /app

# Install uv, the fast package manager
RUN pip install uv

# Copy only the dependency file first. This leverages Docker's layer caching.
# If pyproject.toml doesn't change, Docker will re-use the cached layer below,
# making subsequent builds much faster.
COPY pyproject.toml .

# Use uv to create a virtual environment and install dependencies into it.
RUN uv venv .venv && uv sync


# --- Stage 2: Final Image ---
# This is the actual image we will run in production.
# We use the slim image because it's much smaller and has a reduced attack surface.
FROM ${BASE_IMAGE}-slim AS final

# # Set the working directory
WORKDIR /app

# Add a non-root user for security. Running as root in a container is a bad practice.
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# Copy the virtual environment with all its installed packages from the 'builder' stage.
COPY --from=builder /app/.venv .venv

# Set the PATH to include the virtual environment's bin directory.
# This makes our installed packages available on the command line.
ENV PATH="/app/.venv/bin:$PATH"

# Copy the application source code.
COPY ./app ./app

# The command to run when the container starts.
CMD ["python", "app/main.py"]
