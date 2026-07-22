FROM python:3.11-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Copy project files
COPY . .

# Install project in editable mode with all dependencies
RUN uv pip install --system -e .

# Install FastAPI server dependencies (not part of the research library itself)
RUN uv pip install --system fastapi uvicorn

# Expose FastAPI port
EXPOSE 2024

# Start FastAPI server via uvicorn
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "2024"]
