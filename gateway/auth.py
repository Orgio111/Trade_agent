"""
JWT authentication helpers for the API Gateway.

Uses PyJWT for token creation and verification.
Token payload includes:
  - ``sub`` — user identifier (e.g. ``"user"`` or ``"admin"``)
  - ``role`` — ``"admin"`` | ``"user"``
  - ``exp`` — expiration timestamp (UTC)
  - ``iat`` — issued-at timestamp
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt as _jwt

from core.config import get_settings

logger = logging.getLogger(__name__)


# ── Token creation ────────────────────────────────────────────────────────────


def create_access_token(
    subject: str = "user",
    role: str = "user",
    expires_delta: timedelta | None = None,
) -> str:
    """Create a signed JWT access token.

    Parameters
    ----------
    subject:
        Token subject — typically a username or client ID.
    role:
        RBAC role (``"admin"`` or ``"user"``).
    expires_delta:
        Token TTL.  Defaults to ``gateway_jwt_expiry_minutes`` from config.

    Returns
    -------
    str:
        Encoded JWT string ready to send to the client.
    """
    cfg = get_settings()
    now = datetime.now(timezone.utc)
    delta = expires_delta or timedelta(minutes=cfg.gateway_jwt_expiry_minutes)

    payload: dict[str, Any] = {
        "sub": subject,
        "role": role,
        "iat": now,
        "exp": now + delta,
    }
    token = _jwt.encode(
        payload,
        cfg.gateway_jwt_secret,
        algorithm=cfg.gateway_jwt_algorithm,
    )
    return token


# ── Token verification ────────────────────────────────────────────────────────


class TokenValidationError(Exception):
    """Raised when a token is invalid, expired, or malformed."""


def verify_token(token: str) -> dict[str, Any]:
    """Decode and validate a JWT token.

    Parameters
    ----------
    token:
        The raw JWT string from the ``Authorization: Bearer <token>`` header.

    Returns
    -------
    dict:
        Decoded payload (``sub``, ``role``, ``exp``, ``iat``).

    Raises
    ------
    TokenValidationError:
        If the token is expired, malformed, or the signature is invalid.
    """
    cfg = get_settings()
    try:
        payload = _jwt.decode(
            token,
            cfg.gateway_jwt_secret,
            algorithms=[cfg.gateway_jwt_algorithm],
        )
        return payload
    except _jwt.ExpiredSignatureError:
        raise TokenValidationError("Token has expired")
    except _jwt.InvalidTokenError as exc:
        raise TokenValidationError(f"Invalid token: {exc}")


# ── Admin API key check ───────────────────────────────────────────────────────


def verify_admin_api_key(api_key: str) -> bool:
    """Check a static admin API key from config.

    Useful for monitoring / health-check endpoints that should not require
    a full JWT flow.
    """
    cfg = get_settings()
    if not cfg.gateway_admin_api_key:
        return False
    return api_key == cfg.gateway_admin_api_key
