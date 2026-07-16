"""Fail-closed settings shared by the canonical worker processes."""

from __future__ import annotations

import ipaddress
import os
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from packages.domain import SourceMode
from packages.local_ai import LOCAL_MODEL_BY_ROLE, normalize_local_http_url


_LOCAL_SERVICE_NAMES = frozenset(
    {"localhost", "postgres", "nats", "redis", "host.docker.internal"}
)


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

        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise RuntimeError("DATABASE_URL is required for canonical workers")
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
        }
