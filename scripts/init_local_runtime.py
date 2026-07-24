"""Create one least-scope local canonical runtime environment file."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import secrets


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / ".local" / "canonical.env"
DEFAULT_SECRET_DIRECTORY = ROOT / ".local" / "secrets"


def _write_private(path: Path, payload: str) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    if os.name != "nt":
        os.chmod(path, 0o600)


def _compose_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def initialize(
    path: Path,
    *,
    secret_directory: Path | None = None,
) -> Path:
    """Create a new file atomically and never overwrite existing credentials."""

    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    secrets_path = (secret_directory or DEFAULT_SECRET_DIRECTORY).resolve()
    secrets_path.mkdir(parents=True, exist_ok=True)
    owner_secret = secrets_path / "postgres_admin_password"
    runtime_secret = secrets_path / "db_runtime_password"
    values = {
        "POSTGRES_PASSWORD_FILE_PATH": _compose_path(owner_secret),
        "DB_RUNTIME_PASSWORD_FILE_PATH": _compose_path(runtime_secret),
        "POSTGRES_DB": "quantex",
        "POSTGRES_USER": "quantex",
        "RUNTIME_MODE": "paper_live",
        "PAPER_TRADING": "true",
        "PAPER_ACCOUNT_ID": "paper-main",
        "OLLAMA_BASE_URL": "http://host.docker.internal:11434",
        "CANDIDATE_PROVIDER": "ollama",
    }
    payload = "".join(f"{name}={value}\n" for name, value in values.items())
    if resolved.exists():
        raise FileExistsError(resolved)
    created: list[Path] = []
    try:
        _write_private(owner_secret, secrets.token_urlsafe(48))
        created.append(owner_secret)
        _write_private(runtime_secret, secrets.token_urlsafe(48))
        created.append(runtime_secret)
        _write_private(resolved, payload)
    except Exception:
        for generated in created:
            generated.unlink(missing_ok=True)
        resolved.unlink(missing_ok=True)
        raise
    return resolved


def migrate_inline_secrets(path: Path, *, secret_directory: Path) -> Path:
    """Move the two old inline local DB values to files without printing them."""

    resolved = path.resolve(strict=True)
    records: dict[str, str] = {}
    for raw in resolved.read_text(encoding="utf-8").splitlines():
        if raw.strip() and not raw.lstrip().startswith("#") and "=" in raw:
            name, value = raw.split("=", 1)
            records[name.strip()] = value.strip()
    owner = records.pop("POSTGRES_PASSWORD")
    runtime = records.pop("DB_RUNTIME_PASSWORD")
    secret_directory = secret_directory.resolve()
    secret_directory.mkdir(parents=True, exist_ok=True)
    owner_secret = secret_directory / "postgres_admin_password"
    runtime_secret = secret_directory / "db_runtime_password"
    created: list[Path] = []
    replacement = resolved.with_suffix(resolved.suffix + ".migrating")
    try:
        _write_private(owner_secret, owner)
        created.append(owner_secret)
        _write_private(runtime_secret, runtime)
        created.append(runtime_secret)
        rewritten = {
            "POSTGRES_PASSWORD_FILE_PATH": _compose_path(owner_secret),
            "DB_RUNTIME_PASSWORD_FILE_PATH": _compose_path(runtime_secret),
            **records,
        }
        _write_private(
            replacement,
            "".join(f"{name}={value}\n" for name, value in rewritten.items()),
        )
        os.replace(replacement, resolved)
    except Exception:
        replacement.unlink(missing_ok=True)
        for generated in created:
            generated.unlink(missing_ok=True)
        raise
    return resolved


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_PATH,
        help="new ignored env file; existing files are never overwritten",
    )
    result.add_argument(
        "--secret-directory",
        type=Path,
        default=DEFAULT_SECRET_DIRECTORY,
    )
    result.add_argument(
        "--migrate-inline-secrets",
        action="store_true",
        help="move the two prior inline local DB values into new secret files",
    )
    return result


def main() -> int:
    args = parser().parse_args()
    if args.migrate_inline_secrets:
        path = migrate_inline_secrets(
            args.path,
            secret_directory=args.secret_directory,
        )
    else:
        path = initialize(
            args.path,
            secret_directory=args.secret_directory,
        )
    print(f"created={path}")
    print("next=restrict env/secret ACLs, run config_preflight.py, then deploy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
