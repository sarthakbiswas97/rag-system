#!/bin/sh
set -e

echo "Running database migrations..."
uv run alembic upgrade head

# Default workers: CPU count * 2 + 1, clamped between 2 and 8
WORKERS="${WEB_CONCURRENCY:-$(python -c 'import os; print(min(max(os.cpu_count() * 2 + 1, 2), 8))')}"

echo "Starting Gunicorn with ${WORKERS} workers..."
exec uv run gunicorn rag.main:app \
    -k uvicorn.workers.UvicornWorker \
    --bind "0.0.0.0:${PORT:-8000}" \
    --workers "${WORKERS}" \
    --access-logfile - \
    --error-logfile - \
    --capture-output \
    --enable-stdio-inheritance \
    --graceful-timeout 30 \
    --timeout 60 \
    --keep-alive 5
