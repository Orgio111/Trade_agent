"""Fail-closed settings shared by the canonical worker processes."""

from __future__ import annotations

import ipaddress
import os
from pathlib import Path
from typing import Literal, cast
from urllib.parse import quote
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from packages.domain import SourceMode
from packages.local_ai import LOCAL_MODEL_BY_ROLE, normalize_local_http_url


_LOCAL_SERVICE_NAMES = frozenset(
    {"localhost", "postgres", "nats", "redis", "host.docker.internal"}
)
_MAX_SECRET_FILE_BYTES = 4096


def _is_local_service_host(host: str) -> bool:
    normalized = host.casefold().rstrip(".")
    if normalized in _LOCAL_SERVICE_NAMES:
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _require_local_service_url(
    value: str,
    *,
    service: str,
    schemes: frozenset[str],
) -> str:
    if not value or any(character.isspace() for character in value):
        raise ValueError(f"{service} URL must be non-empty")
    parsed = urlsplit(value)
    if parsed.scheme.casefold() not in schemes:
        raise ValueError(f"{service} URL has an unsupported scheme")
    host = parsed.hostname
    if host is None or not _is_local_service_host(host):
        raise ValueError(f"{service} URL must target the local runtime")
    return value


def _read_secret_file(variable: str) -> str:
    path_value = os.getenv(variable)
    if not path_value:
        raise RuntimeError(f"{variable} is required")
    path = Path(path_value)
    if not path.is_absolute():
        raise RuntimeError(f"{variable} must reference an absolute path")
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{variable} must reference a regular non-symlink file")
    if path.stat().st_size > _MAX_SECRET_FILE_BYTES:
        raise RuntimeError(f"{variable} exceeds the secret size limit")
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise RuntimeError(f"{variable} references an empty secret")
    return value


def _database_url_from_env() -> str:
    direct = os.getenv("DATABASE_URL")
    password_file = os.getenv("DB_RUNTIME_PASSWORD_FILE")
    if direct and password_file:
        raise RuntimeError(
            "DATABASE_URL and DB_RUNTIME_PASSWORD_FILE are mutually exclusive"
        )
    if direct:
        return direct
    password = _read_secret_file("DB_RUNTIME_PASSWORD_FILE")
    host = os.getenv("DB_HOST", "postgres")
    port = os.getenv("DB_PORT", "5432")
    database = os.getenv("DB_NAME", "quantex")
    user = os.getenv("DB_USER", "quantex_runtime")
    return (
        f"postgresql://{quote(user, safe='')}:{quote(password, safe='')}"
        f"@{host}:{port}/{quote(database, safe='')}"
    )


class WorkerSettings(BaseModel):
    """Immutable process settings; only paper/replay modes are admitted."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    service_name: str = Field(min_length=2, max_length=64)
    mode: SourceMode = SourceMode.PAPER_LIVE
    nats_url: str = "nats://127.0.0.1:4222"
    database_url: SecretStr
    ollama_base_url: str = "http://127.0.0.1:11434"
    account_id: str = Field(default="paper-main", min_length=2, max_length=64)
    stream_name: str = Field(default="QUANTEX_CORE", pattern=r"^[A-Z][A-Z0-9_]+$")
    durable_name: str = Field(min_length=2, max_length=128)
    max_deliver: int = Field(default=5, ge=1, le=100)
    ack_wait_seconds: float = Field(default=30.0, gt=0, le=300)
    heartbeat_interval_seconds: float = Field(default=5.0, ge=1, le=60)
    lease_ttl_seconds: float = Field(default=15.0, ge=2, le=180)
    max_consumer_lag: int = Field(default=64, ge=1, le=1_000_000)
    candidate_provider: Literal["ollama", "deterministic_baseline"] = "ollama"

    @field_validator("lease_ttl_seconds")
    @classmethod
    def require_lease_longer_than_heartbeat(cls, value: float, info) -> float:
        heartbeat = info.data.get("heartbeat_interval_seconds", 5.0)
        if value <= heartbeat:
            raise ValueError("lease TTL must exceed heartbeat interval")
        return value

    @field_validator("mode")
    @classmethod
    def reject_non_paper_modes(cls, value: SourceMode) -> SourceMode:
        if value not in {SourceMode.REPLAY, SourceMode.PAPER_LIVE}:
            raise ValueError("canonical workers admit only replay or paper_live mode")
        return value

    @field_validator("nats_url")
    @classmethod
    def validate_nats_url(cls, value: str) -> str:
        return _require_local_service_url(
            value,
            service="NATS",
            schemes=frozenset({"nats", "tls"}),
        )

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, value: SecretStr) -> SecretStr:
        _require_local_service_url(
            value.get_secret_value(),
            service="PostgreSQL",
            schemes=frozenset({"postgres", "postgresql"}),
        )
        return value

    @field_validator("ollama_base_url")
    @classmethod
    def validate_ollama_url(cls, value: str) -> str:
        return normalize_local_http_url(
            value,
            service="Ollama",
            default_port=11434,
        )

    @classmethod
    def from_env(
        cls,
        *,
        service_name: str,
        durable_name: str,
    ) -> WorkerSettings:
        """Load explicit process environment without reading a dotenv file."""

        database_url = _database_url_from_env()
        return cls(
            service_name=service_name,
            mode=SourceMode(os.getenv("RUNTIME_MODE", "paper_live")),
            nats_url=os.getenv("NATS_URL", "nats://127.0.0.1:4222"),
            database_url=SecretStr(database_url),
            ollama_base_url=os.getenv(
                "OLLAMA_BASE_URL", "http://127.0.0.1:11434"
            ),
            account_id=os.getenv("ACCOUNT_ID", "paper-main"),
            stream_name=os.getenv("NATS_STREAM", "QUANTEX_CORE"),
            durable_name=durable_name,
            max_deliver=int(os.getenv("NATS_MAX_DELIVER", "5")),
            ack_wait_seconds=float(os.getenv("NATS_ACK_WAIT_SECONDS", "30")),
            heartbeat_interval_seconds=float(
                os.getenv("WORKER_HEARTBEAT_INTERVAL_SECONDS", "5")
            ),
            lease_ttl_seconds=float(os.getenv("WORKER_LEASE_TTL_SECONDS", "15")),
            max_consumer_lag=int(os.getenv("NATS_MAX_CONSUMER_LAG", "64")),
            candidate_provider=cast(
                Literal["ollama", "deterministic_baseline"],
                os.getenv("CANDIDATE_PROVIDER", "ollama"),
            ),
        )

    def public_summary(self) -> dict[str, object]:
        """Return readiness metadata that cannot reveal the database secret."""

        return {
            "service": self.service_name,
            "mode": self.mode.value,
            "nats_host": urlsplit(self.nats_url).hostname,
            "ollama_base_url": self.ollama_base_url,
            "account_id": self.account_id,
            "stream": self.stream_name,
            "durable": self.durable_name,
            "approved_models": tuple(sorted(set(LOCAL_MODEL_BY_ROLE.values()))),
            "candidate_provider": self.candidate_provider,
        }
