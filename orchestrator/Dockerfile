# ═══════════════════════════════════════════════════════════
# QUANTEX Orchestrator — Multi-stage Python Build (optimized)
# ═══════════════════════════════════════════════════════════

# ── Stage 1: Builder ────────────────────────────────────
FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de AS builder

WORKDIR /app

# Install build dependencies (gcc/g++ for asyncpg, aiohttp C extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast pip (pinned version)
COPY --from=ghcr.io/astral-sh/uv:0.4.23@sha256:ce5a4a9b4c8f2c5658cb8eb63cf5b4ac3e11d80c00780aa79178314b87685445 /uv /usr/local/bin/uv

# Copy and install dependencies
COPY orchestrator/requirements.txt .
RUN uv pip install --system --no-cache --no-compile -r requirements.txt \
    && find /usr/local/lib/python3.12/site-packages -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true

# ── Stage 2: Runner ─────────────────────────────────────
FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de AS runner

WORKDIR /app

# Copy installed packages from builder (only .dist-info + pure python + compiled)
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy orchestrator package + inference package from project root
# (build context is project root, configured in docker-compose.yml)
COPY orchestrator/ ./orchestrator/
COPY inference/ ./inference/
COPY pyproject.toml ./

# Cleanup: remove test dirs, __pycache__, .pyc bloat from site-packages
# Keep .dist-info (needed by importlib.metadata for version checks at runtime)
RUN find /usr/local/lib/python3.12/site-packages -type d -name 'tests' -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib/python3.12/site-packages -type d -name 'test' -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib/python3.12/site-packages -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib/python3.12/site-packages -name '*.pyc' -delete 2>/dev/null || true \
    && find /usr/local/lib/python3.12/site-packages -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true \
    && mkdir -p /app/models /app/logs

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8001

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request, json; r = urllib.request.urlopen('http://localhost:8001/health'); assert json.loads(r.read())['status'] == 'ok'" || exit 1

CMD ["uvicorn", "orchestrator.main:app", "--host", "0.0.0.0", "--port", "8001"]
