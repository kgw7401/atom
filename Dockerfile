FROM python:3.11-slim

WORKDIR /app

# Install system deps for pydub (audio processing)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python package
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir .

# Copy alembic for migrations
COPY alembic/ alembic/
COPY alembic.ini .

# Pre-create data directories
RUN mkdir -p data/audio/chunks data/audio/rounds data/audio/samples

ENV PYTHONUNBUFFERED=1
ENV ATOM_DATABASE_URL=sqlite+aiosqlite:///data/atom.db

# Render provides PORT env var; default 8000 for local Docker
CMD uvicorn atom.api.app:app --host 0.0.0.0 --port ${PORT:-8000}
