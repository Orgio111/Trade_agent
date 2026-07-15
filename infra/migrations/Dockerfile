FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir asyncpg==0.31.0

COPY scripts/apply_migrations.py /app/scripts/apply_migrations.py
COPY db/migrations /app/db/migrations

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

CMD ["python", "/app/scripts/apply_migrations.py"]
