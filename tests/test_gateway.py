"""Unit tests for the API Gateway — auth, rate limiter, and HTTP endpoints."""

from __future__ import annotations

import time
from unittest.mock import patch

import jwt as _jwt
import pytest
from fastapi.testclient import TestClient

from gateway.app import _RateLimiter, app
from gateway.auth import (
    TokenValidationError,
    create_access_token,
    verify_admin_api_key,
    verify_token,
)


# =============================================================================
# Fixtures
# =============================================================================

# NOTE on patching strategy
# --------------------------
# gateway/auth.py and gateway/app.py both do ``from core.config import
# get_settings`` at module-import time, creating *local* references.
# Patching ``core.config.get_settings`` has no effect on those local
# references, so we patch the module-level names directly.

_PATCH_TARGETS = [
    "gateway.auth.get_settings",
    "gateway.app.get_settings",
    # ``core.config.get_settings`` covers function-local imports like the
    # ``from core.config import get_settings`` inside ``proxy_serve_status``.
    "core.config.get_settings",
]
_MOCKED_CONFIG = {
    # JWT / auth
    "gateway_jwt_secret": "test-secret-0123456789",
    "gateway_jwt_algorithm": "HS256",
    "gateway_jwt_expiry_minutes": 60,
    "gateway_admin_api_key": "super-secret-admin-key",
    # Rate limiter
    "gateway_rate_limit_per_minute": 10,
    # Proxy endpoints (config)
    "exchange": "binance",
    "paper_trading": True,
    "initial_capital": 100_000.0,
    "min_consensus_score": 0.65,
    "max_daily_drawdown_pct": 0.05,
    "kelly_fraction": 0.25,
    "ray_serve_url": "",
    # Exchanges / symbols
    "exchanges": ["binance"],
    "symbols": ["BTC/USDT", "ETH/USDT"],
}


def _make_mock_settings():
    """Build a mock object with all the attributes that gateway code reads."""
    return type("MockSettings", (), _MOCKED_CONFIG)()


@pytest.fixture(autouse=True)
def _mock_settings():
    """Patch *both* module-level ``get_settings`` references.

    Also resets the global rate-limiter bucket so that ``TestClient``
    tests don't accumulate entries across tests.

    Each yielded value can be mutated by tests to simulate live config
    changes (e.g. changing the rate-limit value mid-test).
    """
    from gateway.app import _rate_limiter as _global_limiter
    _global_limiter._buckets.clear()

    mock_cfg = _make_mock_settings()

    patchers = [patch(target, return_value=mock_cfg) for target in _PATCH_TARGETS]
    for p in patchers:
        p.start()
    yield mock_cfg
    for p in patchers:
        p.stop()


@pytest.fixture
def admin_token() -> str:
    """A valid JWT with the ``admin`` role."""
    return create_access_token(subject="tester", role="admin")


@pytest.fixture
def user_token() -> str:
    """A valid JWT with the ``user`` role."""
    return create_access_token(subject="tester", role="user")


@pytest.fixture
def expired_token() -> str:
    """A JWT that is already expired (issued 2 hours ago, valid for 1 minute)."""
    from datetime import timedelta

    return create_access_token(
        subject="tester",
        role="user",
        expires_delta=timedelta(minutes=-120),  # negative → issued in the past
    )


# =============================================================================
# Auth unit tests — gateway/auth.py
# =============================================================================


class TestCreateAccessToken:
    def test_returns_string(self) -> None:
        token = create_access_token()
        assert isinstance(token, str)
        assert token.count(".") == 2  # header.payload.signature

    def test_contains_correct_claims(self) -> None:
        token = create_access_token(subject="alice", role="admin")
        payload = _jwt.decode(
            token, "test-secret-0123456789", algorithms=["HS256"]
        )
        assert payload["sub"] == "alice"
        assert payload["role"] == "admin"
        assert "iat" in payload
        assert "exp" in payload

    def test_expiry_uses_config_by_default(self) -> None:
        """Default expiry should honour ``gateway_jwt_expiry_minutes`` (60)."""
        token = create_access_token()
        payload = _jwt.decode(
            token, "test-secret-0123456789", algorithms=["HS256"]
        )
        elapsed = payload["exp"] - payload["iat"]
        assert elapsed == pytest.approx(3600, abs=5)  # 60 min ± 5s slop


class TestVerifyToken:
    def test_valid_token_returns_payload(self) -> None:
        token = create_access_token(subject="bob", role="user")
        payload = verify_token(token)
        assert payload["sub"] == "bob"
        assert payload["role"] == "user"

    def test_expired_token_raises(self, expired_token: str) -> None:
        with pytest.raises(TokenValidationError, match="expired"):
            verify_token(expired_token)

    def test_malformed_token_raises(self) -> None:
        with pytest.raises(TokenValidationError, match="Invalid token"):
            verify_token("not-a-valid-jwt")

    def test_wrong_secret_raises(self) -> None:
        """Token signed with a different secret must be rejected."""
        different = _jwt.encode(
            {"sub": "hacker", "role": "admin"},
            "wrong-secret",
            algorithm="HS256",
        )
        with pytest.raises(TokenValidationError, match="Invalid token"):
            verify_token(different)

    def test_none_algorithm_rejected(self) -> None:
        """JWT signed with ``none`` algorithm — known attack vector."""
        none_token = _jwt.encode(
            {"sub": "hacker", "role": "admin"},
            key="",
            algorithm="none",
        )
        with pytest.raises(TokenValidationError, match="Invalid token"):
            verify_token(none_token)


class TestVerifyAdminApiKey:
    def test_valid_key_returns_true(self) -> None:
        assert verify_admin_api_key("super-secret-admin-key") is True

    def test_invalid_key_returns_false(self) -> None:
        assert verify_admin_api_key("wrong-key") is False

    def test_empty_key_returns_false(self) -> None:
        assert verify_admin_api_key("") is False

    def test_when_config_is_empty_returns_false(self, _mock_settings) -> None:
        _mock_settings.gateway_admin_api_key = ""
        assert verify_admin_api_key("super-secret-admin-key") is False


# =============================================================================
# Rate limiter unit tests — gateway/app.py
# =============================================================================


class TestRateLimiter:
    def test_allows_requests_under_limit(self) -> None:
        limiter = _RateLimiter(window_s=60.0)
        for _ in range(10):
            assert limiter.check("127.0.0.1") is True

    def test_blocks_requests_over_limit(self) -> None:
        limiter = _RateLimiter(window_s=60.0)
        for _ in range(10):
            limiter.check("127.0.0.1")
        assert limiter.check("127.0.0.1") is False

    def test_different_ips_have_independent_buckets(self) -> None:
        limiter = _RateLimiter(window_s=60.0)
        for _ in range(10):
            limiter.check("IP_A")
        assert limiter.check("IP_A") is False
        assert limiter.check("IP_B") is True  # untouched

    def test_sliding_window_expires_old_entries(self) -> None:
        """After the window passes, old entries are pruned and new requests pass."""
        limiter = _RateLimiter(window_s=0.05)  # 50 ms window
        for _ in range(10):
            limiter.check("test-ip")
        assert limiter.check("test-ip") is False

        time.sleep(0.06)
        assert limiter.check("test-ip") is True

    def test_re_reads_max_requests_from_config(self, _mock_settings) -> None:
        """The limit is re-read from settings on every check()."""
        limiter = _RateLimiter(window_s=60.0)
        _mock_settings.gateway_rate_limit_per_minute = 2

        assert limiter.check("ip") is True   # 1/2
        assert limiter.check("ip") is True   # 2/2
        assert limiter.check("ip") is False  # 3rd blocked

        # Raise limit — the *new* limit applies immediately on next check
        _mock_settings.gateway_rate_limit_per_minute = 5
        # Now 2 entries < 5 limit, so request should be allowed
        assert limiter.check("ip") is True   # 3/5 — config change took effect

    def test_increased_limit_applies_after_window_rolls(self, _mock_settings) -> None:
        """After entries expire, the new (higher) limit is used."""
        limiter = _RateLimiter(window_s=0.05)
        _mock_settings.gateway_rate_limit_per_minute = 2

        assert limiter.check("ip") is True
        assert limiter.check("ip") is True
        assert limiter.check("ip") is False  # hit old limit

        # Raise limit and wait for window to slide
        _mock_settings.gateway_rate_limit_per_minute = 5
        time.sleep(0.06)

        # Now the bucket is empty — should be able to make 5 requests
        for _ in range(5):
            assert limiter.check("ip") is True, "new limit should apply"
        assert limiter.check("ip") is False  # hit new limit

    def test_window_is_configurable(self) -> None:
        limiter = _RateLimiter(window_s=0.0)  # zero → never prunes
        for _ in range(10):
            limiter.check("ip")
        assert limiter.check("ip") is False


# =============================================================================
# HTTP endpoint integration tests — gateway/app.py
# =============================================================================


class TestHealthEndpoint:
    """``GET /health`` — public, no auth required."""

    def test_returns_ok(self) -> None:
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "api-gateway"
        assert "timestamp" in data

    def test_no_auth_required(self) -> None:
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200


class TestLoginEndpoint:
    """``POST /auth/login`` — issue JWT with valid API key."""

    ADMIN_KEY = "super-secret-admin-key"

    def test_valid_api_key_header_returns_token(self) -> None:
        client = TestClient(app)
        resp = client.post("/auth/login", headers={"X-API-Key": self.ADMIN_KEY})
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        payload = verify_token(data["access_token"])
        assert payload["role"] == "admin"

    def test_valid_api_key_body_returns_token(self) -> None:
        client = TestClient(app)
        resp = client.post("/auth/login", json={"api_key": self.ADMIN_KEY})
        assert resp.status_code == 200
        assert "access_token" in resp.json()

    def test_invalid_api_key_returns_401(self) -> None:
        client = TestClient(app)
        resp = client.post("/auth/login", json={"api_key": "wrong-key"})
        assert resp.status_code == 401

    def test_missing_api_key_returns_401(self) -> None:
        client = TestClient(app)
        resp = client.post("/auth/login")
        assert resp.status_code == 401


class TestProxyEndpointsAuth:
    """All ``/api/v1/*`` endpoints require a valid JWT."""

    ENDPOINTS = [
        "/api/v1/snapshot",
        "/api/v1/health",
        "/api/v1/config",
        "/api/v1/models",
        "/api/v1/serve-status",
        "/api/v1/paper-status",
        "/api/v1/admin/rate-limits",
    ]

    @pytest.mark.parametrize("endpoint", ENDPOINTS)
    def test_missing_auth_returns_401(self, endpoint: str) -> None:
        client = TestClient(app)
        resp = client.get(endpoint)
        assert resp.status_code == 401, f"{endpoint} should require auth"

    @pytest.mark.parametrize("endpoint", ENDPOINTS)
    def test_bad_token_returns_401(self, endpoint: str) -> None:
        client = TestClient(app)
        resp = client.get(
            endpoint,
            headers={"Authorization": "Bearer invalid-jwt"},
        )
        assert resp.status_code == 401, f"{endpoint} should reject bad tokens"

    @pytest.mark.parametrize("endpoint", ENDPOINTS)
    def test_expired_token_returns_401(self, endpoint: str, expired_token: str) -> None:
        client = TestClient(app)
        resp = client.get(
            endpoint,
            headers={"Authorization": f"Bearer {expired_token}"},
        )
        assert resp.status_code == 401, f"{endpoint} should reject expired tokens"

    def test_malformed_auth_header_returns_401(self) -> None:
        client = TestClient(app)
        resp = client.get("/api/v1/health", headers={"Authorization": "NotBearer abc"})
        assert resp.status_code == 401


class TestProxyHealth:
    """``GET /api/v1/health`` — JWT required, returns basic health info."""

    def test_valid_token_returns_ok(self, user_token: str) -> None:
        client = TestClient(app)
        resp = client.get(
            "/api/v1/health",
            headers={"Authorization": f"Bearer {user_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "timestamp" in data


class TestProxyConfig:
    """``GET /api/v1/config`` — JWT required, returns non-sensitive config."""

    def test_returns_expected_fields(self, user_token: str) -> None:
        client = TestClient(app)
        resp = client.get(
            "/api/v1/config",
            headers={"Authorization": f"Bearer {user_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["exchange"] == "binance"
        assert "BTC/USDT" in data["symbols"]
        assert data["paper_trading"] is True
        assert data["initial_capital"] == 100_000.0


class TestProxySnapshot:
    """``GET /api/v1/snapshot`` — JWT + dashboard state required."""

    def test_no_dashboard_state_returns_503(self, user_token: str) -> None:
        """When ``web.server.get_dashboard_state`` returns None, return 503."""
        with patch("gateway.app._get_state", return_value=None):
            client = TestClient(app)
            resp = client.get(
                "/api/v1/snapshot",
                headers={"Authorization": f"Bearer {user_token}"},
            )
        assert resp.status_code == 503
        assert "Dashboard state not available" in resp.text


class TestProxyModels:
    """``GET /api/v1/models`` — JWT required, lists registered models."""

    def test_registry_unavailable_returns_503(self, user_token: str) -> None:
        """When ``mlops.model_registry`` isn't importable, return 503."""
        import sys
        orig = sys.modules.pop("mlops.model_registry", None)
        try:
            sys.modules["mlops.model_registry"] = None  # type: ignore[assignment]
            client = TestClient(app)
            resp = client.get(
                "/api/v1/models",
                headers={"Authorization": f"Bearer {user_token}"},
            )
            assert resp.status_code == 503
            assert "Model registry not available" in resp.text
        finally:
            if orig:
                sys.modules["mlops.model_registry"] = orig
            else:
                del sys.modules["mlops.model_registry"]


class TestProxyServeStatus:
    """``GET /api/v1/serve-status`` — JWT required, checks Ray Serve."""

    def test_returns_configured_false(self, user_token: str) -> None:
        """When ray_serve_url is empty, returns configured=False."""
        client = TestClient(app)
        resp = client.get(
            "/api/v1/serve-status",
            headers={"Authorization": f"Bearer {user_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["configured"] is False


class TestAdminRateLimits:
    """``GET /api/v1/admin/rate-limits`` — admin role required."""

    def test_user_token_returns_403(self, user_token: str) -> None:
        client = TestClient(app)
        resp = client.get(
            "/api/v1/admin/rate-limits",
            headers={"Authorization": f"Bearer {user_token}"},
        )
        assert resp.status_code == 403

    def test_admin_token_returns_buckets(self, admin_token: str) -> None:
        client = TestClient(app)
        client.get(
            "/api/v1/health",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        resp = client.get(
            "/api/v1/admin/rate-limits",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["max_requests_per_minute"] == 10
        assert data["active_buckets"] >= 1


class TestRateLimitingMiddleware:
    """The rate limiter applies to all endpoints except ``/auth/login``."""

    def test_rate_limit_blocks_excess_requests(self, user_token: str) -> None:
        client = TestClient(app)
        for _ in range(10):
            resp = client.get(
                "/api/v1/health",
                headers={"Authorization": f"Bearer {user_token}"},
            )
            assert resp.status_code == 200

        # 11th request → 429
        resp = client.get(
            "/api/v1/health",
            headers={"Authorization": f"Bearer {user_token}"},
        )
        assert resp.status_code == 429
        assert "Rate limit exceeded" in resp.text
        assert "Retry-After" in resp.headers

    def test_auth_login_exempt_from_rate_limiting(self) -> None:
        """``/auth/login`` bypasses rate limiting."""
        client = TestClient(app)
        for _ in range(20):
            resp = client.post(
                "/auth/login",
                headers={"X-API-Key": "super-secret-admin-key"},
            )
            assert resp.status_code == 200
