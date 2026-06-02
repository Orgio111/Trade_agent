# ═══════════════════════════════════════════════════════════
# QUANTEX Orchestrator — Multi-stage Python Build
# ═══════════════════════════════════════════════════════════

# ── Stage 1: Builder ────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast pip (pinned version)
COPY --from=ghcr.io/astral-sh/uv:0.4.23 /uv /usr/local/bin/uv

# Copy and install dependencies
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# ── Stage 2: Runner ─────────────────────────────────────
FROM python:3.12-slim AS runner

WORKDIR /app

# Create non-root user
RUN groupadd -r quantex && useradd -r -g quantex -u 1001 quantex

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy all Python package files into the orchestrator/ subdirectory
# (build context is ./orchestrator, so . contains all the package files)
COPY . ./orchestrator/

# Switch to non-root user
USER quantex

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8001

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request, json; r = urllib.request.urlopen('http://localhost:8001/health'); assert json.loads(r.read())['status'] == 'ok'" || exit 1

CMD ["uvicorn", "orchestrator.main:app", "--host", "0.0.0.0", "--port", "8001"]
