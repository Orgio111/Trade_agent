"""Shared network-locality validation for manifest and live adapters."""

from urllib.parse import urlsplit


LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def require_loopback_host(host: str, *, service: str) -> str:
    """Return a normalized local host or reject remote service authority."""

    normalized = host.strip().lower()
    if normalized not in LOOPBACK_HOSTS:
        raise ValueError(f"{service} must use a loopback host")
    return normalized


def normalize_loopback_http_url(
    value: str,
    *,
    service: str,
    default_port: int,
) -> str:
    """Normalize an unauthenticated loopback HTTP origin."""

    parsed = urlsplit(value)
    if parsed.scheme != "http" or parsed.hostname is None:
        raise ValueError(f"{service} URL must use HTTP on a loopback host")
    host = require_loopback_host(parsed.hostname, service=service)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(
            f"{service} URL cannot contain credentials, query, or fragment"
        )
    if parsed.path not in ("", "/"):
        raise ValueError(f"{service} URL must not contain an API path")
    try:
        port = parsed.port or default_port
    except ValueError as exc:
        raise ValueError(f"{service} URL has an invalid port") from exc
    if not 1 <= port <= 65535:
        raise ValueError(f"{service} URL has an invalid port")
    display_host = f"[{host}]" if ":" in host else host
    return f"http://{display_host}:{port}"
