# ── Build stage ─────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app
COPY pyproject.toml .

RUN pip install --no-cache-dir build && \
    pip install --no-cache-dir \
        fastapi uvicorn[standard] psycopg2-binary \
        anthropic openai google-generativeai \
        pygithub GitPython chromadb pydantic httpx \
        python-dotenv sentence-transformers

# ── Runtime stage ──────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY . .

EXPOSE 8000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRACE_EMITTER=json

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
