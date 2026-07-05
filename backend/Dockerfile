FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps: ffmpeg for yt-dlp audio extraction (ASR fallback)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src
COPY alembic.ini ./
COPY alembic ./alembic

RUN pip install --upgrade pip && pip install -e .

EXPOSE 8000

CMD ["uvicorn", "ytkb.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
