FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY services/rl_learning/requirements.txt /app/requirements.txt
# Install CPU-only torch first (saves ~1.5GB vs full CUDA wheel)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY services/rl_learning/ /app/services/rl_learning/
COPY orchestrator/ /app/orchestrator/

ENV PYTHONPATH=/app

EXPOSE 8088

CMD ["python", "-m", "services.rl_learning"]