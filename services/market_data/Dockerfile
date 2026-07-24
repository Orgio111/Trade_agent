FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY services/market_data/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy service code
COPY services/market_data/ /app/services/market_data/
COPY orchestrator/ /app/orchestrator/

# Set Python path
ENV PYTHONPATH=/app

EXPOSE 8081

CMD ["python", "-m", "services.market_data"]
