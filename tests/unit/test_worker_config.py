"""Safety tests for canonical worker process configuration."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from packages.domain import SourceMode
from workers.config import WorkerSettings


def settings(**overrides: object) -> WorkerSettings:
    values: dict[str, object] = {
        "service_name": "decision-worker",
        "durable_name": "decision-v1",
        "database_url": SecretStr(
            "postgresql://quantex:local-only@postgres:5432/quantex"
        ),
        "nats_url": "nats://nats:4222",
        "ollama_base_url": "http://host.docker.internal:11434",
    }
    values.update(overrides)
    return WorkerSettings.model_validate(values)


@pytest.mark.parametrize("mode", [SourceMode.REPLAY, SourceMode.PAPER_LIVE])
def test_worker_settings_admit_only_non_live_execution_modes(mode: SourceMode) -> None:
    assert settings(mode=mode).mode is mode


@pytest.mark.parametrize("mode", [SourceMode.HISTORICAL, SourceMode.LIVE])
def test_worker_settings_reject_non_executable_or_live_modes(mode: SourceMode) -> None:
    with pytest.raises(ValidationError, match="replay or paper_live"):
        settings(mode=mode)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("nats_url", "nats://remote.example:4222"),
        (
            "database_url",
            SecretStr("postgresql://user:secret@remote.example/db"),
        ),
        ("ollama_base_url", "http://remote.example:11434"),
    ],
)
def test_worker_settings_reject_remote_runtime_dependencies(
    field: str, value: object
) -> None:
    with pytest.raises(ValidationError, match="local"):
        settings(**{field: value})


def test_public_summary_never_contains_database_credentials() -> None:
    configured = settings()
    summary = configured.public_summary()

    assert "database_url" not in summary
    assert "local-only" not in repr(summary)
    assert summary["mode"] == "paper_live"
    assert summary["ollama_base_url"] == "http://host.docker.internal:11434"
    assert summary["candidate_provider"] == "ollama"


def test_worker_settings_reject_unknown_candidate_provider() -> None:
    with pytest.raises(ValidationError, match="candidate_provider"):
        settings(candidate_provider="force_trade")


def test_alpha_shadow_requires_an_absolute_digest_bound_artifact() -> None:
    artifact_path = (
        Path("C:/run/alpha-candidate/candidate.joblib")
        if os.name == "nt"
        else Path("/run/alpha-candidate/candidate.joblib")
    )
    configured = settings(
        candidate_provider="alpha_shadow",
        candidate_artifact_path=artifact_path,
        candidate_artifact_sha256="a" * 64,
    )
    assert configured.candidate_provider == "alpha_shadow"
    assert configured.candidate_artifact_sha256 == "a" * 64

    with pytest.raises(ValidationError, match="requires candidate artifact"):
        settings(candidate_provider="alpha_shadow")

    with pytest.raises(ValidationError, match="reserved for alpha_shadow"):
        settings(
            candidate_provider="ollama",
            candidate_artifact_path=artifact_path,
            candidate_artifact_sha256="a" * 64,
        )


def test_from_env_never_loads_or_mutates_dotenv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql://quantex:secret@postgres:5432/quantex"
    )
    monkeypatch.setenv("NATS_URL", "nats://nats:4222")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")

    configured = WorkerSettings.from_env(
        service_name="market-data-worker",
        durable_name="market-data-v1",
    )

    assert configured.service_name == "market-data-worker"
    assert configured.database_url.get_secret_value().endswith("/quantex")


def test_from_env_builds_local_dsn_from_read_only_secret_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    secret = tmp_path / "db-runtime"
    secret.write_text("reserved:/?#[]@! password", encoding="utf-8")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("DB_RUNTIME_PASSWORD_FILE", str(secret.resolve()))
    monkeypatch.setenv("DB_HOST", "postgres")
    monkeypatch.setenv("DB_USER", "quantex_runtime")

    configured = WorkerSettings.from_env(
        service_name="execution-worker",
        durable_name="execution-v1",
    )

    assert configured.database_url.get_secret_value() == (
        "postgresql://quantex_runtime:reserved%3A%2F%3F%23%5B%5D%40%21%20password"
        "@postgres:5432/quantex"
    )


def test_from_env_rejects_ambiguous_direct_and_file_secret_sources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    secret = tmp_path / "db-runtime"
    secret.write_text("runtime-password-at-least-16", encoding="utf-8")
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql://quantex:secret@postgres:5432/quantex"
    )
    monkeypatch.setenv("DB_RUNTIME_PASSWORD_FILE", str(secret.resolve()))

    with pytest.raises(RuntimeError, match="mutually exclusive"):
        WorkerSettings.from_env(
            service_name="execution-worker",
            durable_name="execution-v1",
        )
