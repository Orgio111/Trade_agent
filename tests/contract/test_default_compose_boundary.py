"""Static contract for the profile-free local paper runtime."""

from __future__ import annotations

import os
from pathlib import Path
import re

import yaml


ROOT = Path(__file__).parents[2]
COMPOSE_PATH = ROOT / "docker-compose.yml"
OVERRIDE_PATH = ROOT / "docker-compose.override.yml"
IMAGE_DIGEST = re.compile(r"@sha256:[0-9a-f]{64}$")
EXPECTED_DEFAULT_SERVICES = {
    "postgres",
    "redis",
    "migrate",
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
        assert set(profiles) <= {"legacy", "knowledge", "gpu", "certbot"}


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


def test_profile_free_compose_uses_read_only_database_secret_files() -> None:
    payload = yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))
    assert payload["secrets"] == {
        "postgres_admin_password": {
            "file": "${POSTGRES_PASSWORD_FILE_PATH:-.local/secrets/postgres_admin_password}"
        },
        "db_runtime_password": {
            "file": "${DB_RUNTIME_PASSWORD_FILE_PATH:-.local/secrets/db_runtime_password}"
        },
        "grafana_admin_password": {
            "file": "${GRAFANA_ADMIN_PASSWORD_FILE_PATH:-.local/secrets/grafana_admin_password}"
        },
    }
    assert _services()["postgres"]["environment"]["POSTGRES_PASSWORD_FILE"] == (
        "/run/secrets/postgres_admin_password"
    )
    assert _services()["migrate"]["environment"]["PGPASSWORD_FILE"] == (
        "/run/secrets/postgres_admin_password"
    )


def test_canonical_runtime_never_connects_as_database_owner() -> None:
    services = _services()
    for name in EXPECTED_DEFAULT_SERVICES - {"postgres", "migrate", "redis", "nats"}:
        environment = services[name]["environment"]
        assert environment["DB_USER"] == "quantex_runtime"
        assert environment["DB_RUNTIME_PASSWORD_FILE"] == (
            "/run/secrets/db_runtime_password"
        )
        assert "DATABASE_URL" not in environment

    assert services["migrate"]["environment"]["PGUSER"] == "${POSTGRES_USER:-quantex}"
    assert services["migrate"]["environment"]["DB_RUNTIME_USER"] == "quantex_runtime"


def test_affected_chroma_is_not_on_the_trading_network() -> None:
    payload = yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))
    chroma = payload["services"]["chroma"]

    assert chroma["profiles"] == ["knowledge"]
    assert chroma["restart"] == "no"
    assert chroma["networks"] == ["knowledge"]
    assert payload["networks"]["knowledge"]["internal"] is True


def test_canonical_processes_are_hardened_and_logs_are_bounded() -> None:
    services = _services()
    process_services = EXPECTED_DEFAULT_SERVICES - {"postgres", "redis", "nats"}
    for name in process_services:
        service = services[name]
        assert service["read_only"] is True
        assert service["cap_drop"] == ["ALL"]
        assert "no-new-privileges:true" in service["security_opt"]
        assert service["deploy"]["resources"]["limits"]["pids"] <= 128
        assert service["logging"]["options"] == {
            "max-size": "10m",
            "max-file": "5",
        }


def test_all_declared_host_ports_are_loopback_only() -> None:
    for name, service in _services().items():
        for port in service.get("ports", []):
            assert str(port).startswith("127.0.0.1:"), (
                f"service {name} exposes a non-loopback host port: {port}"
            )


def test_vllm_is_fail_fast_and_requires_explicit_gpu_profile() -> None:
    vllm = _services()["vllm"]

    assert vllm["profiles"] == ["gpu"]
    assert vllm["restart"] == "no"


def test_prometheus_mounts_every_referenced_rule_file_read_only() -> None:
    mounts = _services()["prometheus"]["volumes"]

    assert (
        "./monitoring/recording_rules.yml:/etc/prometheus/recording_rules.yml:ro"
        in mounts
    )


def test_compose_images_and_dockerfile_bases_are_immutable() -> None:
    for path in (COMPOSE_PATH, OVERRIDE_PATH):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        for name, service in payload.get("services", {}).items():
            image = service.get("image")
            if image is not None:
                assert IMAGE_DIGEST.search(image), (
                    f"{path.name}:{name} uses a mutable image reference: {image}"
                )

    dockerfiles: list[Path] = []
    ignored = {
        ".git",
        ".local",
        ".venv",
        "node_modules",
        "__pycache__",
        "target",
    }
    for directory, children, files in os.walk(ROOT):
        children[:] = [name for name in children if name not in ignored]
        if "Dockerfile" in files:
            dockerfiles.append(Path(directory) / "Dockerfile")
    assert dockerfiles
    for path in dockerfiles:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.startswith("FROM "):
                continue
            image = line.split()[1]
            assert IMAGE_DIGEST.search(image), (
                f"{path.relative_to(ROOT)} uses a mutable base image: {image}"
            )
