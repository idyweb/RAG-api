# Stage 1: Builder
FROM python:3.12-slim AS builder

# Install uv from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app

# Configure uv to use /opt/venv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv

# Install dependencies
COPY pyproject.toml uv.lock ./
# We use --frozen to ensure strict reproducible builds based on uv.lock
RUN uv sync --frozen --no-install-project --no-dev

# Stage 2: Final
FROM python:3.12-slim
WORKDIR /app

# Point Python to use the virtual environment
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy the dependencies from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application source code
COPY . .

# Expose port
EXPOSE 8000

# Run FastAPI using uvicorn with hot reload enabled
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
