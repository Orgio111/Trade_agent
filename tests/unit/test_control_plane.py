"""Control-plane tests prove the HTTP surface is read-only and fail-closed."""

from __future__ import annotations

from fastapi.testclient import TestClient

from packages.control_plane import DependencyStatus, RuntimeReadiness, create_app


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
