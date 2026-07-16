FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.24 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Resolve exactly the repository lock before copying source so dependency
# layers remain reproducible and cacheable.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --only-group workers --no-install-project --no-cache

COPY packages ./packages
COPY workers ./workers

RUN useradd --create-home --uid 10001 quantex \
    && chown -R quantex:quantex /app
USER quantex

ENV PATH="/app/.venv/bin:$PATH"

CMD ["python", "-m", "packages.control_plane"]
