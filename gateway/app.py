"""
API Gateway — JWT-authenticated, rate-limited entry point for external clients.

Provides:
  - ``POST /auth/login``       — issue a JWT token (requires admin API key)
  - ``GET  /health``           — public health check
  - ``GET  /api/v1/*``         — JWT-protected proxy to the dashboard API

Rate limiting is applied per client IP using an in-memory sliding-window counter.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from core.config import get_settings
from gateway.auth import (
    TokenValidationError,
    create_access_token,
    verify_admin_api_key,
    verify_token,
)

logger = logging.getLogger(__name__)

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Aegis Trader API Gateway",
    version="0.1.0",
    description=__doc__,
)


# ── Rate limiter (in-memory sliding window) ──────────────────────────────────

class _RateLimiter:
    """Simple in-memory sliding-window rate limiter keyed by client IP.

    ``max_requests`` is re-read from the application config on every
    check so that a live config update (e.g. via env reload) takes effect
    without a restart.
    """

    def __init__(self, window_s: float = 60.0) -> None:
        self._window_s = window_s
        self._buckets: dict[str, deque[float]] = defaultdict(deque)

    def check(self, client_ip: str) -> bool:
        max_reqs = get_settings().gateway_rate_limit_per_minute

        now = time.monotonic()
        bucket = self._buckets[client_ip]
        # Prune timestamps outside the window
        cutoff = now - self._window_s
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= max_reqs:
            return False
        bucket.append(now)
        return True


_rate_limiter = _RateLimiter()


# ── Middleware: rate limiting ─────────────────────────────────────────────────

@app.middleware("http")
async def _rate_limit_middleware(request: Request, call_next: Any) -> Any:
    cfg = get_settings()
    client_ip = request.client.host if request.client else "unknown"
    # Skip rate limiting for auth endpoint
    if request.url.path == "/auth/login":
        return await call_next(request)

    if not _rate_limiter.check(client_ip):
        logger.warning("Rate limit exceeded for %s on %s", client_ip, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": "Rate limit exceeded. Try again later."},
            headers={"Retry-After": "60"},
        )

    return await call_next(request)


# ── Auth dependencies ─────────────────────────────────────────────────────────

async def _require_jwt(request: Request) -> dict[str, Any]:
    """FastAPI dependency: extract and verify the JWT from the Authorization header.

    Returns the decoded token payload (``sub``, ``role``, …).
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header. Use: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = auth_header.removeprefix("Bearer ").strip()
    try:
        payload = verify_token(token)
    except TokenValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload


async def _require_admin(payload: dict[str, Any] = Depends(_require_jwt)) -> dict[str, Any]:
    """FastAPI dependency: require the ``admin`` role in the JWT payload."""
    if payload.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    return payload


# ── Public endpoints ──────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Public health-check endpoint (no auth required)."""
    return {"status": "ok", "service": "api-gateway", "timestamp": time.time()}


@app.post("/auth/login")
async def login(request: Request):
    """Issue a JWT token.

    Accept the admin API key either via the ``X-API-Key`` header or as
    ``{"api_key": "..."}`` in the request body.

    On success returns ``{"access_token": "<jwt>", "token_type": "bearer"}``.
    """
    cfg = get_settings()
    api_key: str | None = None

    # Try header first
    api_key = request.headers.get("X-API-Key")

    # Fall back to JSON body
    if not api_key:
        try:
            body = await request.json()
            api_key = body.get("api_key")
        except Exception:
            pass

    if not api_key or not verify_admin_api_key(api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Provide via X-API-Key header or {\"api_key\": \"…\"} body.",
        )

    # Determine role — if the key matches we treat as admin
    token = create_access_token(subject="gateway-client", role="admin")
    return {"access_token": token, "token_type": "bearer"}


# ── Protected proxy endpoints ────────────────────────────────────────────────

def _get_state():
    """Lazy import the dashboard state to avoid circular imports at module level."""
    try:
        from web.server import get_dashboard_state
        return get_dashboard_state()
    except ImportError:
        logger.warning("web.server not importable — dashboard state unavailable")
        return None


def _require_state():
    """FastAPI dependency: ensure the dashboard state is available."""
    ds = _get_state()
    if ds is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dashboard state not available — dashboard server may not be running.",
        )
    return ds


@app.get("/api/v1/snapshot")
async def proxy_snapshot(
    payload: dict[str, Any] = Depends(_require_jwt),
    ds=Depends(_require_state),
):
    """Return a full dashboard snapshot (JWT required)."""
    return await ds.snapshot()


@app.get("/api/v1/health")
async def proxy_health(
    payload: dict[str, Any] = Depends(_require_jwt),
):
    """Return system health (JWT required)."""
    from datetime import datetime
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/api/v1/paper-status")
async def proxy_paper_status(
    payload: dict[str, Any] = Depends(_require_jwt),
    ds=Depends(_require_state),
):
    """Return PaperAccount snapshot (JWT required)."""
    async with ds._lock:
        return JSONResponse(
            ds.paper_state or {"mode": "live", "message": "Paper trading not active"},
        )


@app.get("/api/v1/serve-status")
async def proxy_serve_status(
    payload: dict[str, Any] = Depends(_require_jwt),
):
    """Return Ray Serve health (JWT required)."""
    from core.config import get_settings
    cfg = get_settings()

    try:
        import aiohttp
    except ImportError:
        return JSONResponse({
            "configured": bool(cfg.ray_serve_url),
            "error": "aiohttp not installed",
        })

    result = {
        "configured": bool(cfg.ray_serve_url),
        "serve_url": cfg.ray_serve_url,
        "reachable": False,
        "deployments": {},
    }

    if not cfg.ray_serve_url:
        return JSONResponse(result)

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            async with session.get(f"{cfg.ray_serve_url}/health") as resp:
                if resp.ok:
                    result["reachable"] = True
                    result["cluster_health"] = await resp.json()

            for dep_name in ("ppo", "llm", "embed", "market"):
                try:
                    async with session.get(f"{cfg.ray_serve_url}/{dep_name}/health") as dep_resp:
                        if dep_resp.ok:
                            result["deployments"][dep_name] = await dep_resp.json()
                        else:
                            result["deployments"][dep_name] = {"status": "error", "code": dep_resp.status}
                except Exception as exc:
                    result["deployments"][dep_name] = {"status": "unreachable", "error": str(exc)}
    except Exception as exc:
        result["error"] = str(exc)

    return JSONResponse(result)


@app.get("/api/v1/models")
async def proxy_models(
    payload: dict[str, Any] = Depends(_require_jwt),
):
    """List registered models from the model registry (JWT required)."""
    try:
        from mlops.model_registry import list_models
        models = list_models()
        return [{
            "version": m.version,
            "created_at": m.created_at,
            "model_type": m.model_type,
            "train_sharpe": m.train_sharpe,
            "test_sharpe": m.test_sharpe,
            "test_max_drawdown": m.test_max_drawdown,
            "test_win_rate": m.test_win_rate,
            "total_timesteps": m.total_timesteps,
            "test_information_ratio": m.test_information_ratio,
            "test_calmar_ratio": m.test_calmar_ratio,
        } for m in models]
    except ImportError:
        return JSONResponse(
            {"error": "Model registry not available"},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )


@app.get("/api/v1/config")
async def proxy_config(
    payload: dict[str, Any] = Depends(_require_jwt),
):
    """Expose non-sensitive config (JWT required)."""
    cfg = get_settings()
    return {
        "exchange": cfg.exchange,
        "exchanges": cfg.exchanges,
        "symbols": cfg.symbols,
        "paper_trading": cfg.paper_trading,
        "initial_capital": cfg.initial_capital,
        "min_consensus_score": cfg.min_consensus_score,
        "max_daily_drawdown_pct": cfg.max_daily_drawdown_pct,
        "kelly_fraction": cfg.kelly_fraction,
    }


# ── Admin-only endpoints ─────────────────────────────────────────────────────

@app.get("/api/v1/admin/rate-limits")
async def admin_rate_limits(
    payload: dict[str, Any] = Depends(_require_admin),
):
    """Show current rate-limit buckets (admin only)."""
    return {
        "max_requests_per_minute": get_settings().gateway_rate_limit_per_minute,
        "active_buckets": len(_rate_limiter._buckets),
    }


# ── Startup / shutdown ───────────────────────────────────────────────────────

@app.on_event("startup")
async def _on_startup():
    logger.info("API Gateway started — JWT auth + rate limiting enabled")


@app.on_event("shutdown")
async def _on_shutdown():
    logger.info("API Gateway stopped")
