from __future__ import annotations

from pathlib import Path

from scripts.config_preflight import parse_env, validate_env


def write_env(path: Path, text: str) -> list:
    path.write_text(text, encoding="utf-8")
    return parse_env(path)


def test_canonical_preflight_accepts_minimal_paper_configuration(
    tmp_path: Path,
) -> None:
    secrets_dir = tmp_path / "secrets"
    secrets_dir.mkdir()
    (secrets_dir / "owner").write_text(
        "owner-password-at-least-16", encoding="utf-8"
    )
    (secrets_dir / "runtime").write_text(
        "runtime-password-at-least-16", encoding="utf-8"
    )
    records = write_env(
        tmp_path / ".env",
        "\n".join(
            [
                "POSTGRES_PASSWORD_FILE_PATH=secrets/owner",
                "DB_RUNTIME_PASSWORD_FILE_PATH=secrets/runtime",
                "RUNTIME_MODE=paper_live",
                "PAPER_TRADING=true",
                "OLLAMA_BASE_URL=http://127.0.0.1:11434",
            ]
        ),
    )

    assert validate_env(
        records,
        scope="canonical",
        allow_placeholders=False,
        base_directory=tmp_path,
    ) == []


def test_preflight_reports_names_and_lines_without_secret_values(tmp_path: Path) -> None:
    secret = "must-never-appear-in-finding"
    records = write_env(
        tmp_path / ".env",
        "\n".join(
            [
                f"POSTGRES_PASSWORD={secret}",
                f"POSTGRES_PASSWORD={secret}",
                "DB_RUNTIME_PASSWORD=short",
                f"BINANCE_API_KEY={secret}",
            ]
        ),
    )

    findings = validate_env(records, scope="canonical", allow_placeholders=False)
    rendered = repr(findings)

    assert secret not in rendered
    assert {finding.code for finding in findings} >= {
        "duplicate_key",
        "inline_secret_disallowed",
        "out_of_scope",
        "forbidden_secret_scope",
    }


def test_preflight_rejects_remote_dependencies_and_live_mode(tmp_path: Path) -> None:
    secrets_dir = tmp_path / "secrets"
    secrets_dir.mkdir()
    (secrets_dir / "owner").write_text(
        "owner-password-at-least-16", encoding="utf-8"
    )
    (secrets_dir / "runtime").write_text(
        "runtime-password-at-least-16", encoding="utf-8"
    )
    records = write_env(
        tmp_path / ".env",
        "\n".join(
            [
                "POSTGRES_PASSWORD_FILE_PATH=secrets/owner",
                "DB_RUNTIME_PASSWORD_FILE_PATH=secrets/runtime",
                "RUNTIME_MODE=live",
                "PAPER_TRADING=false",
                "OLLAMA_BASE_URL=https://remote.example:11434",
            ]
        ),
    )

    codes = {
        finding.code
        for finding in validate_env(
            records,
            scope="canonical",
            allow_placeholders=False,
            base_directory=tmp_path,
        )
    }

    assert {"runtime_mode_conflict", "paper_mode_conflict", "remote_dependency"} <= codes
