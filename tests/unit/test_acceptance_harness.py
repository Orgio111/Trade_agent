"""Acceptance harness keeps destructive operations inside an isolated project."""

from __future__ import annotations

from pathlib import Path

from scripts.acceptance import prepare_compose, validate_project_name


def test_prepare_compose_removes_global_names_ports_and_uses_ephemeral_state(
    tmp_path: Path,
) -> None:
    rendered = {
        "services": {
            "postgres": {
                "container_name": "quantex-postgres",
                "ports": [{"published": "5432", "target": 5432}],
                "environment": {"POSTGRES_PASSWORD": "secret"},
                "volumes": [
                    {
                        "type": "volume",
                        "source": "postgres_data",
                        "target": "/var/lib/postgresql/data",
                    }
                ],
            },
            "redis": {
                "environment": {},
                "volumes": [
                    {"type": "volume", "source": "redis_data", "target": "/data"}
                ],
            },
            "nats": {
                "environment": {},
                "volumes": [
                    {"type": "volume", "source": "nats_data", "target": "/data"}
                ],
            },
            "worker": {
                "container_name": "quantex-worker",
                "environment": {
                    "DATABASE_URL": "postgresql://quantex:***@postgres:5432/quantex"
                },
            },
            "control-plane": {"ports": [{"published": "8001", "target": 8001}]},
        },
        "volumes": {"postgres_data": {"name": "quantex_postgres", "external": True}},
        "networks": {"default": {"name": "quantex_default"}},
    }

    isolated = prepare_compose(
        rendered,
        password="disposable-only",
        secret_directory=tmp_path,
        state_directory=tmp_path,
    )

    assert all("container_name" not in service for service in isolated["services"].values())
    assert "ports" not in isolated["services"]["postgres"]
    assert "ports" not in isolated["services"]["worker"]
    assert isolated["services"]["control-plane"]["ports"][0]["host_ip"] == "127.0.0.1"
    assert isolated["services"]["worker"]["environment"]["DATABASE_URL"] == (
        "postgresql://quantex:disposable-only@postgres:5432/quantex"
    )
    assert isolated["services"]["postgres"]["tmpfs"] == [
        "/var/lib/postgresql/data:rw,noexec,nosuid,size=512m"
    ]
    assert isolated["services"]["redis"]["tmpfs"] == [
        "/data:rw,noexec,nosuid,size=512m,uid=999,gid=1000,mode=0770"
    ]
    assert isolated["services"]["nats"]["volumes"] == [
        {
            "type": "bind",
            "source": str(tmp_path / "nats"),
            "target": "/data",
            "read_only": False,
        }
    ]
    assert "volumes" not in isolated
    assert isolated["networks"]["default"] == {}
    assert isolated["secrets"]["postgres_admin_password"]["file"].endswith(
        "postgres_admin_password"
    )


def test_project_name_must_be_disposable_namespace() -> None:
    assert validate_project_name("tradeagent_acceptance_20260723") == (
        "tradeagent_acceptance_20260723"
    )

    for unsafe in ("tradeagent", "quantex", "../tradeagent_acceptance_bad", ""):
        try:
            validate_project_name(unsafe)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe project name accepted: {unsafe!r}")
