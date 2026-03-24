# ── Dockerfile ─────────────────────────────────────────────────────
# Multi-stage build: builder installs deps, runtime copies the venv.
# BASE_IMAGE задаётся через --build-arg для переключения CPU/GPU.
# ───────────────────────────────────────────────────────────────────

ARG BASE_IMAGE=python:3.11-slim

# ─── Stage 1: builder ────────────────────────────────────────────
FROM ${BASE_IMAGE} AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_NO_INTERACTION=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && pip install poetry==1.8.*

WORKDIR /app

COPY pyproject.toml poetry.lock* ./

RUN poetry install --no-root --only main || poetry install --no-root --only main --no-cache

COPY . .

RUN poetry install --only main


# ─── Stage 2: runtime ───────────────────────────────────────────
FROM ${BASE_IMAGE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app /app

ENV PATH="/app/.venv/bin:$PATH"

ENV CLIP_CACHE_DIR=/app/.cache/clip
ENV EASYOCR_MODULE_PATH=/app/.cache/easyocr

RUN mkdir -p ${CLIP_CACHE_DIR} ${EASYOCR_MODULE_PATH}

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

CMD ["python", "-m", "uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
