FROM python:3.12-slim AS builder

WORKDIR /app

ENV UV_NO_CACHE=1

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ src/
RUN uv sync --frozen --no-dev


FROM python:3.12-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"
ENV PORT=8000

RUN addgroup --system app && adduser --system --ingroup app app

COPY --from=builder /app /app
COPY alembic.ini ./
COPY migrations/ migrations/
COPY docker-entrypoint.sh /app/docker-entrypoint.sh

RUN mkdir -p /app/data && chown -R app:app /app/data

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/livez')" || exit 1

ENTRYPOINT ["/app/docker-entrypoint.sh"]
