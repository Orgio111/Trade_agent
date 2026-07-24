"""Control-plane tests prove the HTTP surface is read-only and fail-closed."""

from __future__ import annotations

import asyncio
from fastapi.testclient import TestClient
import pytest
from pydantic import SecretStr
from types import SimpleNamespace

from packages.control_plane import DependencyStatus, RuntimeReadiness, create_app
from packages.control_plane.probes import DefaultReadinessProbe
from workers.config import WorkerSettings


class FakeProbe:
    def __init__(self, result: RuntimeReadiness) -> None:
        self.result = result

    async def check(self) -> RuntimeReadiness:
        return self.result


def readiness(*, ready: bool, execution_enabled: bool = False) -> RuntimeReadiness:
    return RuntimeReadiness(
        ready=ready,
        execution_enabled=execution_enabled,
        mode="paper_live",
        dependencies={
            "postgres": DependencyStatus(healthy=ready, detail="checked"),
            "nats": DependencyStatus(healthy=ready, detail="checked"),
            "ollama": DependencyStatus(healthy=ready, detail="checked"),
        },
    )


def test_liveness_does_not_claim_dependency_readiness() -> None:
    client = TestClient(create_app(FakeProbe(readiness(ready=False))))

    assert client.get("/health").json() == {
        "status": "ok",
        "service": "quantex-control-plane",
    }
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["ready"] is False


def test_runtime_reports_kill_switch_execution_boundary() -> None:
    client = TestClient(
        create_app(FakeProbe(readiness(ready=True, execution_enabled=False)))
    )

    response = client.get("/api/v1/runtime")

    assert response.status_code == 200
    assert response.json()["ready"] is True
    assert response.json()["execution_enabled"] is False


def test_control_plane_exposes_no_http_mutation_routes() -> None:
    app = create_app(FakeProbe(readiness(ready=True)))
    paths = app.openapi()["paths"]

    assert paths
    for operations in paths.values():
        assert not ({"post", "put", "patch", "delete"} & set(operations))


def test_metrics_exposes_readiness_without_secrets_or_high_cardinality_labels() -> None:
    client = TestClient(create_app(FakeProbe(readiness(ready=True, execution_enabled=False))))

    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "quantex_runtime_ready 1" in response.text
    assert "quantex_execution_enabled 0" in response.text
    assert 'quantex_dependency_healthy{dependency="postgres"} 1' in response.text
    assert "postgresql://" not in response.text


class FakeDatabaseConnection:
    def __init__(self, authority: dict[str, object]) -> None:
        self.authority = authority
        self.closed = False

    async def fetchval(self, query: str) -> int:
        assert "schema_migrations" in query
        return 3

    async def fetchrow(self, query: str, account_id: str) -> dict[str, object]:
        assert "risk_policies" in query
        assert "portfolio_snapshots" in query
        assert "instrument_constraints" in query
        assert account_id == "paper-main"
        return self.authority

    async def fetch(self, query: str, *_args: object) -> list[dict[str, object]]:
        assert "worker_leases" in query
        return [
            {"service_name": name, "status": "ready", "consumer_lag": 0}
            for name in (
                "market-producer",
                "market-data-worker",
                "feature-worker",
                "candidate-worker",
                "reconciliation-worker",
                "decision-worker",
                "execution-worker",
            )
        ]

    async def close(self) -> None:
        self.closed = True


def probe_settings() -> WorkerSettings:
    return WorkerSettings(
        service_name="control-plane",
        database_url=SecretStr(
            "postgresql://quantex:secret@127.0.0.1:5432/quantex"
        ),
        durable_name="control-plane-readonly-v1",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("authority", "expected"),
    [
        (
            {
                "active_policy_count": 1,
                "portfolio_fresh": True,
                "constraints_initialized": True,
                "feature_state_fresh": True,
                "kill_switch_active": False,
            },
            True,
        ),
        (
            {
                "active_policy_count": 0,
                "portfolio_fresh": False,
                "constraints_initialized": False,
                "feature_state_fresh": True,
                "kill_switch_active": None,
            },
            False,
        ),
    ],
)
async def test_database_execution_readiness_requires_every_authority_input(
    monkeypatch: pytest.MonkeyPatch,
    authority: dict[str, object],
    expected: bool,
) -> None:
    connection = FakeDatabaseConnection(authority)

    async def connect(_dsn: str) -> FakeDatabaseConnection:
        return connection

    monkeypatch.setattr("packages.control_plane.probes.asyncpg.connect", connect)
    status, execution_enabled = await DefaultReadinessProbe(
        probe_settings()
    )._check_database()

    assert status.healthy is True
    assert execution_enabled is expected
    assert connection.closed is True


@pytest.mark.asyncio
async def test_expired_worker_lease_makes_database_unready(monkeypatch) -> None:
    connection = FakeDatabaseConnection(
        {
            "active_policy_count": 1,
            "portfolio_fresh": True,
            "constraints_initialized": True,
            "feature_state_fresh": True,
            "kill_switch_active": False,
        }
    )

    async def fetch(_query: str, *_args: object) -> list[dict[str, object]]:
        return [
            {
                "service_name": "market-data-worker",
                "status": "expired",
                "consumer_lag": 0,
            }
        ]

    connection.fetch = fetch

    async def connect(_dsn: str) -> FakeDatabaseConnection:
        return connection

    monkeypatch.setattr("packages.control_plane.probes.asyncpg.connect", connect)
    status, execution_enabled = await DefaultReadinessProbe(
        probe_settings()
    )._check_database()

    assert status.healthy is False
    assert execution_enabled is False
    assert "worker_leases_ready=False" in status.detail


@pytest.mark.asyncio
async def test_stale_feature_state_makes_database_unready(monkeypatch) -> None:
    connection = FakeDatabaseConnection(
        {
            "active_policy_count": 1,
            "portfolio_fresh": True,
            "constraints_initialized": True,
            "feature_state_fresh": False,
            "kill_switch_active": False,
        }
    )

    async def connect(_dsn: str) -> FakeDatabaseConnection:
        return connection

    monkeypatch.setattr("packages.control_plane.probes.asyncpg.connect", connect)
    status, execution_enabled = await DefaultReadinessProbe(
        probe_settings()
    )._check_database()

    assert status.healthy is False
    assert execution_enabled is False
    assert "feature_state_fresh=False" in status.detail


@pytest.mark.asyncio
async def test_nats_probe_disables_reconnect_and_caps_connect_timeout(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def connect(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)

        async def flush(*, timeout: int) -> None:
            assert timeout == 5

        async def close() -> None:
            return None

        return SimpleNamespace(flush=flush, close=close)

    monkeypatch.setattr("packages.control_plane.probes.nats.connect", connect)

    status, reachable = await DefaultReadinessProbe(
        probe_settings(), timeout_seconds=5
    )._check_nats()

    assert status.healthy is True
    assert reachable is True
    assert captured["allow_reconnect"] is False
    assert captured["connect_timeout"] == 1.0


@pytest.mark.asyncio
async def test_bounded_probe_accepts_a_stricter_dependency_timeout() -> None:
    async def blocked() -> tuple[DependencyStatus, bool]:
        await asyncio.sleep(1)
        raise AssertionError("dependency timeout was not enforced")

    status, enabled = await DefaultReadinessProbe(probe_settings())._bounded(
        blocked, timeout_seconds=0.01
    )

    assert status == DependencyStatus(healthy=False, detail="probe timed out")
    assert enabled is False
