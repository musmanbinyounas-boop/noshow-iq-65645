# ---------- Stage 1: builder ----------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --prefix=/install -r requirements.txt \
    # Strip bytecode caches and bundled tests from installed packages
    && find /install -depth -type d -name "__pycache__" -exec rm -rf {} + \
    && find /install -depth -type d -name "tests" -exec rm -rf {} + \
    && find /install -depth -type d -name "test" -exec rm -rf {} + \
    && find /install -name "*.pyc" -delete \
    && find /install -name "*.pyo" -delete

# ---------- Stage 2: runtime ----------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN useradd --create-home --shell /bin/bash app

WORKDIR /app

COPY --from=builder /install /usr/local

COPY noshow_iq/ ./noshow_iq/
COPY models/ ./models/

RUN chown -R app:app /app

USER app

EXPOSE 7860

CMD ["uvicorn", "noshow_iq.api:app", "--host", "0.0.0.0", "--port", "7860"]
