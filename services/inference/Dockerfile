FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY services/inference/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY services/inference/ /app/services/inference/
COPY orchestrator/ /app/orchestrator/

ENV PYTHONPATH=/app

EXPOSE 8085

CMD ["python", "-m", "services.inference"]