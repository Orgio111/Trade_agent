"""Hermes service launcher safety tests."""

from __future__ import annotations

import pytest

from packages.hermes import cli
from packages.hermes.bootstrap import create_local_app_from_environment
from packages.hermes.errors import JournalConfigurationError


def test_cli_binds_factory_to_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(app: str, **options: object) -> None:
        captured["app"] = app
        captured.update(options)

    monkeypatch.setenv("HERMES_API_PORT", "8123")
    monkeypatch.setattr(cli.uvicorn, "run", fake_run)

    cli.main()

    assert captured == {
        "app": "packages.hermes.bootstrap:create_local_app_from_environment",
        "factory": True,
        "host": "127.0.0.1",
        "port": 8123,
        "access_log": True,
    }


@pytest.mark.parametrize("value", ["not-a-port", "0", "65536"])
def test_cli_rejects_invalid_port(
    value: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HERMES_API_PORT", value)

    with pytest.raises(SystemExit, match="HERMES_API_PORT"):
        cli._port_from_environment()


def test_app_factory_requires_an_absolute_vault(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HERMES_VAULT_PATH", "relative-vault")

    with pytest.raises(JournalConfigurationError, match="must be absolute"):
        create_local_app_from_environment()
