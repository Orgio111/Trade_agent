"""Filesystem and boundary tests for the certification controller."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from packages.certification.controller import (
    ControllerPaths,
    ProcessLock,
    RuntimeConfig,
    SleepInhibitor,
    ensure_signing_key,
    load_signing_key,
    read_status,
)
from packages.certification.core import sign_document


def test_runtime_config_rejects_legacy_project_and_remote_control_plane(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="canonical"):
        RuntimeConfig(project="trade_agent", env_file=tmp_path / "env")
    with pytest.raises(ValueError, match="loopback"):
        RuntimeConfig(
            project="trade_agent_canonical",
            env_file=tmp_path / "env",
            runtime_url="https://example.com/api/v1/runtime",
        )


def test_signing_key_is_stable_and_status_verifies(tmp_path: Path) -> None:
    paths = ControllerPaths(
        state_directory=tmp_path,
        restore_evidence=tmp_path / "restore.json",
        promoted_artifact=tmp_path / "promoted.json",
    )
    key = ensure_signing_key(paths.signing_key)
    assert ensure_signing_key(paths.signing_key) == key
    document = sign_document(
        {"certification_state": {"status": "running"}},
        key,
    )
    paths.signed_status.write_text(json.dumps(document), encoding="utf-8")

    assert read_status(paths)["certification_state"]["status"] == "running"
    tampered = dict(document)
    tampered["certification_state"] = {"status": "promoted"}
    paths.signed_status.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RuntimeError, match="signature"):
        read_status(paths)
    paths.signing_key.unlink()
    with pytest.raises(FileNotFoundError):
        load_signing_key(paths.signing_key)


def test_process_lock_rejects_active_pid(tmp_path: Path) -> None:
    lock_path = tmp_path / "controller.lock"
    with ProcessLock(lock_path):
        with pytest.raises(RuntimeError, match="already runs"):
            with ProcessLock(lock_path):
                raise AssertionError("unreachable")


def test_sleep_inhibitor_enters_and_restores() -> None:
    with SleepInhibitor() as inhibitor:
        if __import__("os").name == "nt":
            assert inhibitor.active is True
    assert inhibitor.active is False
