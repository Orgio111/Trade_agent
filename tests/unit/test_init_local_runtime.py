from __future__ import annotations

import os
from pathlib import Path

import pytest

from scripts.init_local_runtime import initialize, migrate_inline_secrets


def test_initialize_creates_only_canonical_paper_inputs(tmp_path: Path) -> None:
    secret_directory = tmp_path / "secrets"
    path = initialize(
        tmp_path / "canonical.env",
        secret_directory=secret_directory,
    )

    records = dict(
        line.split("=", 1)
        for line in path.read_text(encoding="utf-8").splitlines()
    )
    assert set(records) == {
        "POSTGRES_PASSWORD_FILE_PATH",
        "DB_RUNTIME_PASSWORD_FILE_PATH",
        "POSTGRES_DB",
        "POSTGRES_USER",
        "RUNTIME_MODE",
        "PAPER_TRADING",
            "PAPER_ACCOUNT_ID",
            "OLLAMA_BASE_URL",
            "CANDIDATE_PROVIDER",
        }
    assert len((secret_directory / "postgres_admin_password").read_text()) >= 48
    assert len((secret_directory / "db_runtime_password").read_text()) >= 48
    assert records["RUNTIME_MODE"] == "paper_live"
    assert records["CANDIDATE_PROVIDER"] == "ollama"
    assert records["PAPER_TRADING"] == "true"
    assert not any(
        name.startswith(("BINANCE_", "OPENAI_", "GROQ_", "NVIDIA_"))
        for name in records
    )
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600


def test_initialize_refuses_to_replace_existing_credentials(
    tmp_path: Path,
) -> None:
    path = tmp_path / "canonical.env"
    path.write_text("preserve=true\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        initialize(path, secret_directory=tmp_path / "secrets")

    assert path.read_text(encoding="utf-8") == "preserve=true\n"


def test_migrate_inline_secrets_preserves_values_without_leaving_inline_copy(
    tmp_path: Path,
) -> None:
    path = tmp_path / "canonical.env"
    path.write_text(
        "POSTGRES_PASSWORD=owner-password-at-least-16\n"
        "DB_RUNTIME_PASSWORD=runtime-password-at-least-16\n"
        "RUNTIME_MODE=paper_live\n",
        encoding="utf-8",
    )
    secret_directory = tmp_path / "secrets"

    migrate_inline_secrets(path, secret_directory=secret_directory)

    rewritten = path.read_text(encoding="utf-8")
    assert "POSTGRES_PASSWORD=" not in rewritten
    assert "DB_RUNTIME_PASSWORD=" not in rewritten
    assert "POSTGRES_PASSWORD_FILE_PATH=" in rewritten
    assert (secret_directory / "postgres_admin_password").read_text() == (
        "owner-password-at-least-16"
    )
    assert (secret_directory / "db_runtime_password").read_text() == (
        "runtime-password-at-least-16"
    )
