"""Static contract for the profile-free local paper runtime."""

from __future__ import annotations

from pathlib import Path
import re

import yaml


COMPOSE_PATH = Path(__file__).parents[2] / "docker-compose.yml"
OVERRIDE_PATH = Path(__file__).parents[2] / "docker-compose.override.yml"
EXPECTED_DEFAULT_SERVICES = {
    "postgres",
    "redis",
    "migrate",
    "chroma",
    "nats",
    "control-plane",
    "market-producer",
    "market-data-worker",
    "feature-worker",
    "candidate-worker",
    "reconciliation-worker",
    "decision-worker",
    "execution-worker",
}


def _services() -> dict[str, dict[str, object]]:
    payload = yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))
    services = payload.get("services")
    assert isinstance(services, dict)
    return services


def _override_services() -> dict[str, dict[str, object]]:
    payload = yaml.safe_load(OVERRIDE_PATH.read_text(encoding="utf-8"))
    services = payload.get("services")
    assert isinstance(services, dict)
    return services


def test_profile_free_runtime_contains_only_canonical_local_services() -> None:
    services = _services()
    defaults = {name for name, service in services.items() if "profiles" not in service}

    assert defaults == EXPECTED_DEFAULT_SERVICES
    for name, service in services.items():
        if name in defaults:
            continue
        profiles = service.get("profiles")
        assert isinstance(profiles, list)
        assert "legacy" in profiles


def test_canonical_execution_worker_is_paper_only_and_has_no_broker_secret() -> None:
    execution = _services()["execution-worker"]
    environment = execution["environment"]

    assert execution["command"] == ["python", "-m", "workers.execution"]
    assert environment["RUNTIME_MODE"] == "paper_live"
    assert not any(str(key).startswith("BINANCE_") for key in environment)
    assert "OPENAI_API_KEY" not in environment
    assert "GROQ_API_KEY" not in environment


def test_profile_free_host_ports_are_loopback_only() -> None:
    for name, service in _services().items():
        if "profiles" in service:
            continue
        for port in service.get("ports", []):
            rendered = str(port)
            assert rendered.startswith("127.0.0.1:"), (
                f"default service {name} exposes a non-loopback host port: {rendered}"
            )


def test_automatic_development_override_can_only_modify_legacy_services() -> None:
    base = _services()

    for name in _override_services():
        assert "legacy" in base[name].get("profiles", [])


def test_legacy_override_never_mounts_host_secret_files() -> None:
    for service in _override_services().values():
        for volume in service.get("volumes", []):
            assert ".env" not in str(volume)


def test_legacy_vllm_host_port_is_loopback_only() -> None:
    for port in _services()["vllm"]["ports"]:
        assert str(port).startswith("127.0.0.1:")


def test_profile_free_compose_requires_only_the_postgres_secret() -> None:
    compose = COMPOSE_PATH.read_text(encoding="utf-8")
    required_variables = set(re.findall(r"\$\{([A-Z0-9_]+):\?", compose))

    assert required_variables == {"POSTGRES_PASSWORD"}
