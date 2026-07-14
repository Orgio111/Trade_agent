FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY services/moe_router/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY services/moe_router/ /app/services/moe_router/
COPY services/feature_engine/ /app/services/feature_engine/
COPY services/orderbook/ /app/services/orderbook/
COPY services/market_data/ /app/services/market_data/
COPY orchestrator/ /app/orchestrator/

ENV PYTHONPATH=/app

EXPOSE 8084

CMD ["python", "-m", "services.moe_router"]