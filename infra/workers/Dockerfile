FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de

COPY --from=ghcr.io/astral-sh/uv:0.11.24@sha256:99ea34acedc870ba4ad11a1f540a1c04267c9f30aadc465a94406f52dfda2c36 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Resolve exactly the repository lock before copying source so dependency
# layers remain reproducible and cacheable.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --only-group workers --no-install-project --no-cache \
    && rm -f /bin/uv /bin/uvx

COPY packages ./packages
COPY workers ./workers

RUN useradd --create-home --uid 10001 quantex \
    && chown -R quantex:quantex /app
USER quantex

ENV PATH="/app/.venv/bin:$PATH"

CMD ["python", "-m", "packages.control_plane"]
