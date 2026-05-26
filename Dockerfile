# ═══════════════════════════════════════════════════════════════════════════════
#  Trade Agent — Multi‑Stage Docker Build
# ═══════════════════════════════════════════════════════════════════════════════

# ── Stage 1: Build dependencies (compilers, headers, protobuf codegen) ────────
FROM python:3.11-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install Python dependencies into a venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# If gRPC protobuf codegen is needed, uncomment:
# COPY proto/ proto/
# RUN pip install grpcio-tools && \
#     python -m grpc_tools.protoc -Iproto --python_out=proto --grpc_python_out=proto \
#         proto/agents.proto proto/orders.proto proto/risk.proto

# ── Stage 2: GPU runtime (CUDA + PyTorch) — optional, tag separately ─────────
# Uncomment for GPU‑enabled image:
# FROM nvidia/cuda:12.4-runtime-ubuntu22.04 AS gpu
# RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*
# COPY --from=builder /opt/venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"
# ENV CUDA_VISIBLE_DEVICES=0

# ── Stage 3: Production runtime ──────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS production

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the virtualenv with all dependencies from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy application code
COPY . .

# Non-root user for security
RUN useradd -m -u 1000 trader && chown -R trader:trader /app
USER trader

EXPOSE 3000 8000

# Default: start the main trading loop (also starts Prometheus + dashboard)
CMD ["python", "-m", "main"]
