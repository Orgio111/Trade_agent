FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY services/orderbook/requirements.txt /app/requirements.txt
COPY services/market_data/requirements.txt /app/market_data_requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt -r /app/market_data_requirements.txt

COPY services/orderbook/ /app/services/orderbook/
COPY services/market_data/ /app/services/market_data/
COPY orchestrator/ /app/orchestrator/

ENV PYTHONPATH=/app

EXPOSE 8083

CMD ["python", "-m", "services.orderbook"]