"""Admission rules for Ollama endpoints that remain on the local machine."""

from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit, urlunsplit


_LOCAL_HOSTNAMES = frozenset({"localhost", "host.docker.internal"})


def _is_local_host(host: str) -> bool:
    normalized = host.casefold().rstrip(".")
    if normalized in _LOCAL_HOSTNAMES:
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def normalize_local_http_url(
    value: str,
    *,
    service: str,
    default_port: int,
) -> str:
    """Normalize a credential-free HTTP URL restricted to the local host.

    ``host.docker.internal`` is admitted because Windows Docker containers use
    it to reach the native Ollama process without exposing inference to a cloud
    provider or another network host.
    """

    if not value or any(character.isspace() for character in value):
        raise ValueError(f"{service} URL must be a non-empty HTTP URL")
    parsed = urlsplit(value)
    if parsed.scheme.casefold() != "http":
        raise ValueError(f"{service} URL must use http")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{service} URL cannot contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError(f"{service} URL cannot contain query or fragment data")
    if parsed.path not in {"", "/"}:
        raise ValueError(f"{service} URL cannot contain a path")
    host = parsed.hostname
    if host is None or not _is_local_host(host):
        raise ValueError(f"{service} URL must resolve to the local machine")
    try:
        port = parsed.port or default_port
    except ValueError as exc:
        raise ValueError(f"{service} URL contains an invalid port") from exc
    if not 1 <= port <= 65_535:
        raise ValueError(f"{service} URL contains an invalid port")
    normalized_host = host.casefold().rstrip(".")
    netloc_host = (
        f"[{normalized_host}]" if ":" in normalized_host else normalized_host
    )
    return urlunsplit(("http", f"{netloc_host}:{port}", "", "", ""))
