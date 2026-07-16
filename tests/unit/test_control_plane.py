"""Control-plane tests prove the HTTP surface is read-only and fail-closed."""

from __future__ import annotations

from fastapi.testclient import TestClient
import pytest
from pydantic import SecretStr

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
                "portfolio_initialized": True,
                "constraints_initialized": True,
                "kill_switch_active": False,
            },
            True,
        ),
        (
            {
                "active_policy_count": 0,
                "portfolio_initialized": False,
                "constraints_initialized": False,
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
