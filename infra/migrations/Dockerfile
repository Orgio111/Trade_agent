FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de

WORKDIR /app

RUN pip install --no-cache-dir asyncpg==0.31.0

COPY scripts/apply_migrations.py /app/scripts/apply_migrations.py
COPY db/migrations /app/db/migrations

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN useradd --create-home --uid 10002 quantex-migrate \
    && chown -R quantex-migrate:quantex-migrate /app
USER quantex-migrate

CMD ["python", "/app/scripts/apply_migrations.py"]
