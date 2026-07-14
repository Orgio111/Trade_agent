FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY services/feature_engine/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY services/feature_engine/ /app/services/feature_engine/
COPY services/market_data/ /app/services/market_data/
COPY orchestrator/ /app/orchestrator/

ENV PYTHONPATH=/app

EXPOSE 8082

CMD ["python", "-m", "services.feature_engine"]