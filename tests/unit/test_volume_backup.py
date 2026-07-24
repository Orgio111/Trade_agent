from __future__ import annotations

from pathlib import Path

import pytest
from cryptography.exceptions import InvalidTag

from scripts.volume_backup import (
    _decrypt,
    _encrypt,
    _validate_container_path,
    _validate_container_user,
    _validate_image_reference,
    _validate_volume_name,
)


def test_encrypted_backup_round_trip_and_tamper_rejection(tmp_path: Path) -> None:
    source = tmp_path / "source.tar.gz"
    encrypted = tmp_path / "backup.aes256gcm"
    key = tmp_path / "backup.key"
    restored = tmp_path / "restored.tar.gz"
    source.write_bytes((b"canonical-ledger\n" * 1000) + b"\x00\xff")

    evidence = _encrypt(source, encrypted, key)
    _decrypt(encrypted, restored, key)

    assert restored.read_bytes() == source.read_bytes()
    assert len(evidence["ciphertext_sha256"]) == 64
    assert len(evidence["key_id"]) == 16
    tampered = bytearray(encrypted.read_bytes())
    tampered[-17] ^= 1
    encrypted.write_bytes(tampered)
    with pytest.raises(InvalidTag):
        _decrypt(encrypted, tmp_path / "rejected.tar.gz", key)


@pytest.mark.parametrize(
    "name",
    ["../volume", "volume/name", "", "volume name", "a" * 129],
)
def test_volume_name_rejects_shell_and_path_syntax(name: str) -> None:
    with pytest.raises(ValueError, match="invalid Docker volume name"):
        _validate_volume_name(name)


def test_archive_image_requires_digest_pin() -> None:
    assert _validate_image_reference(
        "alpine:3.20@sha256:" + ("a" * 64)
    ).endswith("a" * 64)
    for mutable in ("alpine:latest", "alpine:3.20", "alpine@sha256:short"):
        with pytest.raises(ValueError, match="immutable sha256 digest"):
            _validate_image_reference(mutable)


def test_volume_target_requires_normalized_absolute_path() -> None:
    assert _validate_container_path("/var/lib/postgresql/data") == (
        "/var/lib/postgresql/data"
    )
    for unsafe in ("source", "/source/../data", "/source//data", "/source,$HOME"):
        with pytest.raises(ValueError, match="normalized absolute"):
            _validate_container_path(unsafe)


def test_archive_user_rejects_shell_syntax() -> None:
    assert _validate_container_user("postgres") == "postgres"
    assert _validate_container_user("999:999") == "999:999"
    for unsafe in ("user name", "--privileged", "user;id", ""):
        with pytest.raises(ValueError, match="simple container user"):
            _validate_container_user(unsafe)
