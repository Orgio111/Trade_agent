"""
API Gateway — JWT-authenticated, rate-limited entry point for external clients.

Provides:
  - JWT token issuance (/auth/login)
  - Token-protected proxy routes (/api/v1/*)
  - IP-based rate limiting
  - RBAC scaffolding (admin / user roles)
  - Health-check endpoint

Start with ``GATEWAY_ENABLED=True`` in the env, or import ``app`` directly
for embedding in another FastAPI server.
"""
from __future__ import annotations

from gateway.app import app, create_access_token, verify_token

__all__ = ["app", "create_access_token", "verify_token"]
