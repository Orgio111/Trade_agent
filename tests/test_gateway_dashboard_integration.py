"""Integration tests: API Gateway × Dashboard running together.

Both FastAPI apps run in the same process (via ``TestClient``), mirroring the
production deployment where ``main.py`` starts both as uvicorn tasks and the
gateway's proxy endpoints call ``web.server.get_dashboard_state()`` directly.

These tests verify:
  - Full auth flow (login → JWT → proxy endpoint)
  - Dashboard state propagation through the gateway
  - Data consistency between dashboard and gateway APIs
  - Rate limiting across proxy endpoints
  - Dashboard state lifecycle (created on demand, reset between tests)
  - Error scenarios (state unavailable, expired tokens, etc.)
"""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from gateway.app import app as gateway_app
from gateway.auth import create_access_token, verify_admin_api_key

# =============================================================================
# Fixtures
# =============================================================================

# Patch targets — all modules that import ``get_settings`` at module level.
# This is required because ``from core.config import get_settings`` creates a
# local reference that ``patch("core.config.get_settings")`` doesn't reach.
_PATCH_TARGETS = [
    "gateway.auth.get_settings",
    "gateway.app.get_settings",
    "web.server.get_settings",
    "core.config.get_settings",
]

_MOCKED_CONFIG = {
    # JWT / auth
    "gateway_jwt_secret": "int-test-secret-0123456789",
    "gateway_jwt_algorithm": "HS256",
    "gateway_jwt_expiry_minutes": 60,
    "gateway_admin_api_key": "int-test-admin-key",
    # Rate limiter
    "gateway_rate_limit_per_minute": 100,  # high enough not to interfere
    # Dashboard / proxy config
    "exchange": "binance",
    "exchanges": ["binance"],
    "symbols": ["BTC/USDT", "ETH/USDT"],
    "paper_trading": True,
    "initial_capital": 100_000.0,
    "min_consensus_score": 0.65,
    "max_daily_drawdown_pct": 0.05,
    "kelly_fraction": 0.25,
    "ray_serve_url": "",
    # Misc
    "gateway_enabled": True,
    "gateway_port": 8000,
    "prometheus_port": 9999,
    "supervisor_cadence_s": 60.0,
    "backtest_start": "2024-01-01",
    "backtest_end": "2024-12-31",
    "backtest_timeframe": "1h",
    "backtest_slippage_bps": 5,
    "backtest_max_cycles": 1000,
    "nim_api_key": "",
    "nim_base_url": "http://localhost:8000/v1",
    "nim_chat_model": "meta/llama-3.1-8b-instruct",
    "nim_json_model": "meta/llama-3.1-8b-instruct",
    "nim_embed_model": "nvidia/nv-embed-qa-4",
}


def _make_mock_settings():
    """Build a plain object with all ``_MOCKED_CONFIG`` attributes."""
    return type("MockSettings", (), _MOCKED_CONFIG)()


@pytest.fixture(autouse=True)
def _mock_settings():
    """Patch all module-level ``get_settings`` references and reset shared state.

    Also clears the global rate-limiter buckets and resets the
    ``DashboardState`` singleton so each test starts clean.
    """
    # Reset rate-limiter buckets
    from gateway.app import _rate_limiter as _global_limiter
    _global_limiter._buckets.clear()

    # Reset DashboardState singleton
    import web.server as _ws_mod
    _ws_mod._state = None

    mock_cfg = _make_mock_settings()

    patchers = [patch(target, return_value=mock_cfg) for target in _PATCH_TARGETS]
    for p in patchers:
        p.start()
    yield mock_cfg
    for p in patchers:
        p.stop()


@pytest.fixture
def gateway_client() -> TestClient:
    """``TestClient`` for the API Gateway FastAPI app."""
    return TestClient(gateway_app)


@pytest.fixture
def dashboard_client() -> TestClient:
    """``TestClient`` for the Dashboard FastAPI app."""
    from web.server import app as dashboard_app
    return TestClient(dashboard_app)


@pytest.fixture
def admin_token() -> str:
    """Valid JWT with ``admin`` role."""
    return create_access_token(subject="integ-test", role="admin")


@pytest.fixture
def user_token() -> str:
    """Valid JWT with ``user`` role."""
    return create_access_token(subject="integ-test", role="user")


@pytest.fixture
def expired_token() -> str:
    """JWT that has already expired."""
    return create_access_token(
        subject="integ-test",
        role="user",
        expires_delta=timedelta(minutes=-120),
    )


def _obtain_token(gateway_client: TestClient) -> str:
    """Helper: login via the gateway and return a fresh JWT."""
    resp = gateway_client.post(
        "/auth/login",
        headers={"X-API-Key": "int-test-admin-key"},
    )
    assert resp.status_code == 200, f"Login failed: {resp.text}"
    return resp.json()["access_token"]


def _populate_dashboard_state(**overrides) -> None:
    """Populate the shared ``DashboardState`` singleton with sample data.

    Uses the overriding kwargs for portfolio fields where provided; falls
    back to defaults for all other fields.
    """
    import web.server as ws
    from datetime import datetime
    from core.models import PortfolioState

    ds = ws.get_dashboard_state()

    # Populate portfolio
    import asyncio
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(ds.update_portfolio(PortfolioState(
            equity=overrides.get("equity", 105_000.0),
            cash=overrides.get("cash", 55_000.0),
            daily_pnl=overrides.get("daily_pnl", 1_500.0),
            daily_pnl_pct=overrides.get("daily_pnl_pct", 0.0145),
            peak_equity=110_000.0,
            current_drawdown_pct=0.025,
            kill_switch_active=False,
            positions={"BTC/USDT": 0.5},
        )))
        # Populate agent signals
        loop.run_until_complete(ds.update_agent_signal("technical", {
            "signal": "bullish", "confidence": 0.72,
        }))
        loop.run_until_complete(ds.update_agent_signal("sentiment", {
            "signal": "neutral", "confidence": 0.55,
        }))
        # Populate risk metrics
        loop.run_until_complete(ds.update_risk("BTC/USDT", {
            "var_99": 0.032, "sharpe": 1.45, "max_drawdown": 0.12,
        }))
        # Populate prices
        loop.run_until_complete(ds.update_price("BTC/USDT", 69_500.0))
        loop.run_until_complete(ds.update_price("ETH/USDT", 3_450.0))
        # Populate paper state
        loop.run_until_complete(ds.update_paper_state({
            "equity": 105_000.0,
            "cash": 55_000.0,
            "open_positions": 1,
            "total_trades": 14,
            "win_rate": 0.64,
        }))
    finally:
        loop.close()



# =============================================================================
# Tests: Full Auth Flow
# =============================================================================


class TestFullAuthFlow:
    """End-to-end: login via gateway, use JWT to call proxy endpoints."""

    def test_login_issue_token(self, gateway_client: TestClient) -> None:
        """``POST /auth/login`` with valid API key returns a JWT."""
        token = _obtain_token(gateway_client)
        assert isinstance(token, str)
        assert token.count(".") == 2  # header.payload.signature

    def test_token_works_for_proxy_endpoint(
        self, gateway_client: TestClient,
    ) -> None:
        """Token obtained via login can be used to call a proxy endpoint."""
        token = _obtain_token(gateway_client)
        resp = gateway_client.get(
            "/api/v1/health",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_rejected_token_returns_401(
        self, gateway_client: TestClient, expired_token: str,
    ) -> None:
        """Expired/malformed tokens are rejected by proxy endpoints."""
        resp = gateway_client.get(
            "/api/v1/health",
            headers={"Authorization": f"Bearer {expired_token}"},
        )
        assert resp.status_code == 401

    def test_full_round_trip(
        self, gateway_client: TestClient,
    ) -> None:
        """Login → token → proxy_health → 200 OK."""
        token = _obtain_token(gateway_client)
        resp = gateway_client.get(
            "/api/v1/config",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["exchange"] == "binance"
        assert data["paper_trading"] is True


# =============================================================================
# Tests: Dashboard State Propagation
# =============================================================================


class TestDashboardStatePropagation:
    """Data written to the dashboard flows through to the gateway proxy."""

    def test_snapshot_contains_portfolio_data(
        self, gateway_client: TestClient,
    ) -> None:
        """``/api/v1/snapshot`` returns portfolio data from dashboard state."""
        _populate_dashboard_state()
        token = _obtain_token(gateway_client)
        resp = gateway_client.get(
            "/api/v1/snapshot",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "portfolio" in data
        assert data["portfolio"]["equity"] == 105_000.0
        assert data["portfolio"]["cash"] == 55_000.0
        assert "orders" in data
        assert "agents" in data
        assert "risk" in data
        assert "prices" in data

    def test_snapshot_includes_agent_signals(
        self, gateway_client: TestClient,
    ) -> None:
        """Agent signals written to dashboard appear in gateway snapshot."""
        _populate_dashboard_state()
        token = _obtain_token(gateway_client)
        resp = gateway_client.get(
            "/api/v1/snapshot",
            headers={"Authorization": f"Bearer {token}"},
        )
        data = resp.json()
        assert data["agents"]["technical"]["signal"] == "bullish"
        assert data["agents"]["sentiment"]["signal"] == "neutral"

    def test_snapshot_includes_prices(
        self, gateway_client: TestClient,
    ) -> None:
        """Market prices from dashboard appear in gateway snapshot."""
        _populate_dashboard_state()
        token = _obtain_token(gateway_client)
        resp = gateway_client.get(
            "/api/v1/snapshot",
            headers={"Authorization": f"Bearer {token}"},
        )
        data = resp.json()
        assert data["prices"]["BTC/USDT"] == 69_500.0
        assert data["prices"]["ETH/USDT"] == 3_450.0

    def test_paper_status_from_dashboard(
        self, gateway_client: TestClient,
    ) -> None:
        """``/api/v1/paper-status`` returns paper trading data from dashboard."""
        _populate_dashboard_state()
        token = _obtain_token(gateway_client)
        resp = gateway_client.get(
            "/api/v1/paper-status",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["equity"] == 105_000.0
        assert data["cash"] == 55_000.0
        assert data["win_rate"] == 0.64

    def test_empty_paper_status_when_not_active(
        self, gateway_client: TestClient,
    ) -> None:
        """When paper state is None, returns a fallback message."""
        # Don't populate paper state — just create the dashboard state
        token = _obtain_token(gateway_client)
        resp = gateway_client.get(
            "/api/v1/paper-status",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("mode") == "live"


# =============================================================================
# Tests: Cross-App Data Consistency
# =============================================================================


class TestCrossAppDataConsistency:
    """Same data is accessible via both the dashboard and gateway APIs."""

    def test_snapshot_matches_across_apps(
        self, gateway_client: TestClient,
        dashboard_client: TestClient,
    ) -> None:
        """Both apps return identical snapshot data."""
        ds = _populate_dashboard_state()

        # Get snapshot from dashboard
        dash_resp = dashboard_client.get("/api/snapshot")
        assert dash_resp.status_code == 200
        dash_data = dash_resp.json()

        # Get snapshot from gateway
        token = _obtain_token(gateway_client)
        gw_resp = gateway_client.get(
            "/api/v1/snapshot",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert gw_resp.status_code == 200
        gw_data = gw_resp.json()

        # Core fields should match
        assert dash_data["portfolio"]["equity"] == gw_data["portfolio"]["equity"]
        assert dash_data["portfolio"]["cash"] == gw_data["portfolio"]["cash"]
        assert dash_data["prices"]["BTC/USDT"] == gw_data["prices"]["BTC/USDT"]
        assert dash_data["agents"]["technical"]["signal"] == \
               gw_data["agents"]["technical"]["signal"]

    def test_config_matches_across_apps(
        self, gateway_client: TestClient,
        dashboard_client: TestClient,
    ) -> None:
        """Config endpoint returns matching data from both apps."""
        token = _obtain_token(gateway_client)

        dash_resp = dashboard_client.get("/api/config")
        gw_resp = gateway_client.get(
            "/api/v1/config",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert dash_resp.status_code == 200
        assert gw_resp.status_code == 200
        assert dash_resp.json() == gw_resp.json()

    def test_health_matches_across_apps(
        self, gateway_client: TestClient,
        dashboard_client: TestClient,
    ) -> None:
        """Both apps report healthy."""
        token = _obtain_token(gateway_client)

        dash_resp = dashboard_client.get("/api/health")
        assert dash_resp.status_code == 200
        assert dash_resp.json()["status"] == "ok"

        gw_resp = gateway_client.get(
            "/api/v1/health",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert gw_resp.status_code == 200
        assert gw_resp.json()["status"] == "ok"


# =============================================================================
# Tests: Dashboard State Update
# =============================================================================


class TestDashboardStateUpdate:
    """Updates to dashboard state are immediately visible via the gateway."""

    def test_updated_portfolio_reflected_in_gateway(
        self, gateway_client: TestClient,
    ) -> None:
        import web.server as ws
        from core.models import PortfolioState
        import asyncio

        # Start with initial state
        _populate_dashboard_state()
        ds = ws.get_dashboard_state()

        # Update portfolio
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(ds.update_portfolio(PortfolioState(
                equity=120_000.0, cash=70_000.0,
                daily_pnl=5_000.0, daily_pnl_pct=0.04,
                peak_equity=120_000.0,
            )))
        finally:
            loop.close()

        token = _obtain_token(gateway_client)
        resp = gateway_client.get(
            "/api/v1/snapshot",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["portfolio"]["equity"] == 120_000.0
        assert data["portfolio"]["cash"] == 70_000.0

    def test_concurrent_order_agent_price_updates(
        self, gateway_client: TestClient,
    ) -> None:
        """Multiple state updates are all reflected in the snapshot."""
        import web.server as ws
        from core.models import Order, OrderStatus, Side, OrderType
        import asyncio
        from datetime import datetime

        ws._state = None  # ensure clean state
        ds = ws.get_dashboard_state()

        loop = asyncio.new_event_loop()
        try:
            # Add an order
            loop.run_until_complete(ds.add_order(Order(
                symbol="BTC/USDT", side=Side.BUY, quantity=0.1,
                order_type=OrderType.MARKET, status=OrderStatus.FILLED,
                session_id="integration-test",
                avg_fill_price=69_500.0,
                created_at=datetime.utcnow(),
            )))
            # Update agent signal
            loop.run_until_complete(ds.update_agent_signal("quant", {
                "signal": "bearish", "confidence": 0.81,
            }))
            # Update price
            loop.run_until_complete(ds.update_price("BTC/USDT", 68_900.0))
        finally:
            loop.close()

        token = _obtain_token(gateway_client)
        resp = gateway_client.get(
            "/api/v1/snapshot",
            headers={"Authorization": f"Bearer {token}"},
        )
        data = resp.json()
        assert len(data["orders"]) == 1
        assert data["orders"][0]["symbol"] == "BTC/USDT"
        assert data["agents"]["quant"]["signal"] == "bearish"
        assert data["prices"]["BTC/USDT"] == 68_900.0


# =============================================================================
# Tests: Gateway Rate Limiting with Dashboard
# =============================================================================


class TestGatewayRateLimitingWithDashboard:
    """Rate limiting applies to proxy endpoints even with active dashboard."""

    def test_rate_limit_blocks_at_limit(
        self, _mock_settings, gateway_client: TestClient,
    ) -> None:
        """When rate limit is hit, the gateway returns 429."""
        _mock_settings.gateway_rate_limit_per_minute = 5
        token = _obtain_token(gateway_client)

        # Use up the budget
        for _ in range(5):
            resp = gateway_client.get(
                "/api/v1/health",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert resp.status_code == 200

        # 6th request → 429
        resp = gateway_client.get(
            "/api/v1/health",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 429
        assert "Rate limit exceeded" in resp.text

    def test_login_exempt_during_rate_limit(
        self, _mock_settings, gateway_client: TestClient,
    ) -> None:
        """``/auth/login`` is never rate-limited."""
        _mock_settings.gateway_rate_limit_per_minute = 5
        token = _obtain_token(gateway_client)

        # Exhaust rate limit
        for _ in range(5):
            gateway_client.get(
                "/api/v1/health",
                headers={"Authorization": f"Bearer {token}"},
            )

        # Login is still allowed
        resp = gateway_client.post(
            "/auth/login",
            headers={"X-API-Key": "int-test-admin-key"},
        )
        assert resp.status_code == 200

    def test_rate_limit_resets_between_windows(
        self, _mock_settings, gateway_client: TestClient,
    ) -> None:
        """After the window passes, new requests are allowed."""
        _mock_settings.gateway_rate_limit_per_minute = 2
        token = _obtain_token(gateway_client)

        # Exhaust budget
        for _ in range(2):
            gateway_client.get(
                "/api/v1/health",
                headers={"Authorization": f"Bearer {token}"},
            )
        assert gateway_client.get(
            "/api/v1/health",
            headers={"Authorization": f"Bearer {token}"},
        ).status_code == 429

        # Reset the rate limiter buckets (simulates window rolling)
        from gateway.app import _rate_limiter
        _rate_limiter._buckets.clear()

        # Now requests are allowed again
        resp = gateway_client.get(
            "/api/v1/health",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200


# =============================================================================
# Tests: Admin Endpoints
# =============================================================================


class TestAdminEndpointsThroughGateway:
    """Admin-only endpoints respect role-based access control."""

    def test_admin_token_works(self, gateway_client: TestClient) -> None:
        token = _obtain_token(gateway_client)  # login returns admin role
        resp = gateway_client.get(
            "/api/v1/admin/rate-limits",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "max_requests_per_minute" in data
        assert "active_buckets" in data

    def test_user_token_blocked(self, gateway_client: TestClient,
                                 user_token: str) -> None:
        resp = gateway_client.get(
            "/api/v1/admin/rate-limits",
            headers={"Authorization": f"Bearer {user_token}"},
        )
        assert resp.status_code == 403


# =============================================================================
# Tests: Dashboard State Lifecycle
# =============================================================================


class TestDashboardStateLifecycle:
    """``DashboardState`` is created on first access and persists across calls."""

    def test_state_created_on_first_access(
        self, gateway_client: TestClient,
    ) -> None:
        """Calling a proxy endpoint that needs state auto-creates it."""
        _populate_dashboard_state()
        token = _obtain_token(gateway_client)
        resp = gateway_client.get(
            "/api/v1/snapshot",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        assert "portfolio" in resp.json()

    def test_state_persists_across_calls(
        self, gateway_client: TestClient,
    ) -> None:
        """Data written to state remains available across multiple gateway calls."""
        ds = _populate_dashboard_state()
        token = _obtain_token(gateway_client)

        # First call
        resp1 = gateway_client.get(
            "/api/v1/snapshot",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp1.json()["portfolio"]["equity"] == 105_000.0

        # Second call — same data
        resp2 = gateway_client.get(
            "/api/v1/snapshot",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp2.json()["portfolio"]["equity"] == 105_000.0


# =============================================================================
# Tests: Error Scenarios
# =============================================================================


class TestErrorScenarios:
    """Error handling across the gateway + dashboard boundary."""

    def test_no_state_returns_503(
        self, gateway_client: TestClient, user_token: str,
    ) -> None:
        """When dashboard state can't be obtained, snapshot returns 503."""
        # Force _get_state to return None
        with patch("gateway.app._get_state", return_value=None):
            resp = gateway_client.get(
                "/api/v1/snapshot",
                headers={"Authorization": f"Bearer {user_token}"},
            )
        assert resp.status_code == 503
        assert "Dashboard state not available" in resp.text

    def test_public_health_always_works(
        self, gateway_client: TestClient,
    ) -> None:
        """``/health`` is public and works regardless of dashboard state."""
        resp = gateway_client.get("/health")
        assert resp.status_code == 200

    def test_serve_status_unconfigured(
        self, gateway_client: TestClient, user_token: str,
    ) -> None:
        """``/api/v1/serve-status`` returns configured=False when no URL set."""
        resp = gateway_client.get(
            "/api/v1/serve-status",
            headers={"Authorization": f"Bearer {user_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["configured"] is False
        assert data["serve_url"] == ""


# =============================================================================
# Tests: Multi-request Sequences
# =============================================================================


class TestMultiRequestSequences:
    """Complex multi-step workflows across gateway and dashboard."""

    def test_login_then_populate_then_snapshot(
        self, gateway_client: TestClient,
    ) -> None:
        """Login → populate dashboard → query via gateway — all work in order."""
        # Step 1: Login
        token = _obtain_token(gateway_client)

        # Step 2: Populate dashboard state
        ds = _populate_dashboard_state()

        # Step 3: Query via gateway
        resp = gateway_client.get(
            "/api/v1/snapshot",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        assert resp.json()["portfolio"]["equity"] == 105_000.0

        # Step 4: Query dashboard directly — same data
        from web.server import app as dashboard_app
        dash_resp = TestClient(dashboard_app).get("/api/snapshot")
        assert dash_resp.json()["portfolio"]["equity"] == 105_000.0

    def test_concurrent_gateway_and_dashboard_queries(
        self, gateway_client: TestClient,
        dashboard_client: TestClient,
    ) -> None:
        """Both apps can be queried independently without interference."""
        ds = _populate_dashboard_state()
        token = _obtain_token(gateway_client)

        # Query both in any order
        gw_resp = gateway_client.get(
            "/api/v1/snapshot",
            headers={"Authorization": f"Bearer {token}"},
        )
        dash_resp = dashboard_client.get("/api/snapshot")

        assert gw_resp.status_code == 200
        assert dash_resp.status_code == 200
        assert gw_resp.json()["portfolio"]["equity"] == \
               dash_resp.json()["portfolio"]["equity"]


# =============================================================================
# Tests: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Boundary conditions and unusual states."""

    def test_empty_dashboard_state(
        self, gateway_client: TestClient,
    ) -> None:
        """An empty (just-created) DashboardState still returns valid JSON."""
        # Don't populate — let the singleton auto-create
        import web.server as ws
        ws.get_dashboard_state()  # creates the singleton

        token = _obtain_token(gateway_client)
        resp = gateway_client.get(
            "/api/v1/snapshot",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        # Default portfolio should be present
        assert data["portfolio"]["equity"] == 100_000.0  # default
        assert data["portfolio"]["cash"] == 100_000.0  # default
        # Empty collections should be present
        assert data["orders"] == []
        assert data["agents"] == {}
        assert data["prices"] == {}
