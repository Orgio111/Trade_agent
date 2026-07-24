"""Create and restore authenticated encrypted backups of stopped Docker volumes."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import tempfile

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


ROOT = Path(__file__).resolve().parents[1]
ALPINE_IMAGE = (
    "alpine:3.20@sha256:"
    "d9e853e87e55526f6b2917df91a2115c36dd7c696a35be12163d44e6e2a4b6bc"
)
MAGIC = b"QUANTEX-BACKUP-V1\0"
NONCE_BYTES = 12
TAG_BYTES = 16
CHUNK_BYTES = 1024 * 1024
VOLUME_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
IMAGE_REFERENCE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9./:_-]+@sha256:[0-9a-f]{64}$"
)
CONTAINER_PATH = re.compile(r"^/[A-Za-z0-9._/-]+$")
CONTAINER_USER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")


def _run(command: list[str], *, capture: bool = False) -> str:
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    return completed.stdout or ""


def _validate_volume_name(value: str) -> str:
    if not VOLUME_NAME.fullmatch(value):
        raise ValueError("invalid Docker volume name")
    return value


def _validate_image_reference(value: str) -> str:
    if not IMAGE_REFERENCE.fullmatch(value):
        raise ValueError("archive image must use an immutable sha256 digest")
    return value


def _validate_container_path(value: str) -> str:
    if (
        not CONTAINER_PATH.fullmatch(value)
        or "//" in value
        or ".." in value.split("/")
    ):
        raise ValueError("volume target must be a normalized absolute container path")
    return value


def _validate_container_user(value: str) -> str:
    if not CONTAINER_USER.fullmatch(value):
        raise ValueError("archive user must be a simple container user or uid")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _new_private_file(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    if os.name != "nt":
        os.chmod(path, 0o600)


def _encrypt(source: Path, destination: Path, key_path: Path) -> dict[str, str]:
    key = secrets.token_bytes(32)
    nonce = secrets.token_bytes(NONCE_BYTES)
    header = MAGIC + nonce
    _new_private_file(key_path, key)
    encryptor = Cipher(algorithms.AES(key), modes.GCM(nonce)).encryptor()
    encryptor.authenticate_additional_data(header)
    try:
        with source.open("rb") as plain, destination.open("xb") as encrypted:
            encrypted.write(header)
            while chunk := plain.read(CHUNK_BYTES):
                encrypted.write(encryptor.update(chunk))
            encrypted.write(encryptor.finalize())
            encrypted.write(encryptor.tag)
    except Exception:
        destination.unlink(missing_ok=True)
        key_path.unlink(missing_ok=True)
        raise
    if os.name != "nt":
        os.chmod(destination, 0o600)
    return {
        "ciphertext_sha256": _sha256(destination),
        "key_id": hashlib.sha256(key).hexdigest()[:16],
    }


def _decrypt(source: Path, destination: Path, key_path: Path) -> None:
    key = key_path.read_bytes()
    if len(key) != 32:
        raise ValueError("backup key must be exactly 32 bytes")
    size = source.stat().st_size
    header_bytes = len(MAGIC) + NONCE_BYTES
    if size <= header_bytes + TAG_BYTES:
        raise ValueError("encrypted backup is truncated")
    with source.open("rb") as encrypted:
        header = encrypted.read(header_bytes)
        if not header.startswith(MAGIC):
            raise ValueError("encrypted backup header is invalid")
        nonce = header[len(MAGIC):]
        encrypted.seek(-TAG_BYTES, os.SEEK_END)
        tag = encrypted.read(TAG_BYTES)
        encrypted.seek(header_bytes)
        remaining = size - header_bytes - TAG_BYTES
        decryptor = Cipher(algorithms.AES(key), modes.GCM(nonce, tag)).decryptor()
        decryptor.authenticate_additional_data(header)
        try:
            with destination.open("xb") as plain:
                while remaining:
                    chunk = encrypted.read(min(CHUNK_BYTES, remaining))
                    if not chunk:
                        raise ValueError("encrypted backup is truncated")
                    remaining -= len(chunk)
                    plain.write(decryptor.update(chunk))
                plain.write(decryptor.finalize())
        except Exception:
            destination.unlink(missing_ok=True)
            raise


def _assert_volume_stopped(volume: str) -> None:
    _run(["docker", "volume", "inspect", volume], capture=True)
    running = _run(
        ["docker", "ps", "--quiet", "--filter", f"volume={volume}"],
        capture=True,
    ).strip()
    if running:
        raise RuntimeError("refusing to back up a volume mounted by a running container")


def backup_volume(
    volume: str,
    *,
    output_directory: Path,
    key_directory: Path,
    archive_image: str = ALPINE_IMAGE,
    volume_target: str = "/source",
    archive_user: str | None = None,
) -> Path:
    volume = _validate_volume_name(volume)
    archive_image = _validate_image_reference(archive_image)
    volume_target = _validate_container_path(volume_target)
    if archive_user is not None:
        archive_user = _validate_container_user(archive_user)
    _assert_volume_stopped(volume)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    stem = f"{volume}-{timestamp}"
    output_directory.mkdir(parents=True, exist_ok=True)
    key_directory.mkdir(parents=True, exist_ok=True)
    encrypted_path = (output_directory / f"{stem}.tar.gz.aes256gcm").resolve()
    key_path = (key_directory / f"{stem}.key").resolve()
    manifest_path = encrypted_path.with_suffix(encrypted_path.suffix + ".json")
    with tempfile.TemporaryDirectory(prefix="quantex-volume-backup-") as directory:
        work = Path(directory).resolve()
        archive = work / "volume.tar.gz"
        command = [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges:true",
                "--mount",
                f"type=volume,source={volume},target={volume_target},readonly",
                "--mount",
                f"type=bind,source={work},target=/backup",
        ]
        if archive_user is not None:
            command.extend(["--user", archive_user])
        command.extend(
            [
                archive_image,
                "tar",
                "-C",
                volume_target,
                "-czf",
                "/backup/volume.tar.gz",
                ".",
            ]
        )
        _run(command)
        cryptographic = _encrypt(archive, encrypted_path, key_path)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "source_volume": volume,
        "archive": encrypted_path.name,
        "archive_image": archive_image,
        "archive_user": archive_user,
        "volume_target": volume_target,
        "encryption": "AES-256-GCM",
        **cryptographic,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest_path


def restore_volume(
    encrypted_path: Path,
    *,
    key_path: Path,
    target_volume: str,
    archive_image: str = ALPINE_IMAGE,
    volume_target: str = "/restore",
    archive_user: str | None = None,
) -> None:
    target_volume = _validate_volume_name(target_volume)
    archive_image = _validate_image_reference(archive_image)
    volume_target = _validate_container_path(volume_target)
    if archive_user is not None:
        archive_user = _validate_container_user(archive_user)
    with tempfile.TemporaryDirectory(prefix="quantex-volume-restore-") as directory:
        work = Path(directory).resolve()
        archive = work / "volume.tar.gz"
        _decrypt(encrypted_path.resolve(), archive, key_path.resolve())
        try:
            _run(["docker", "volume", "inspect", target_volume], capture=True)
        except subprocess.CalledProcessError:
            _run(
                [
                    "docker",
                    "volume",
                    "create",
                    "--label",
                    "quantex.restore-drill=true",
                    target_volume,
                ],
                capture=True,
            )
        else:
            raise RuntimeError("restore target volume already exists")
        command = [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges:true",
                "--mount",
                f"type=volume,source={target_volume},target={volume_target}",
                "--mount",
                f"type=bind,source={work},target=/backup,readonly",
        ]
        if archive_user is not None:
            command.extend(["--user", archive_user])
        command.extend(
            [
                archive_image,
                "tar",
                "-C",
                volume_target,
                "-xzf",
                "/backup/volume.tar.gz",
            ]
        )
        _run(command)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    backup = commands.add_parser("backup")
    backup.add_argument("volume", type=_validate_volume_name)
    backup.add_argument(
        "--output-directory",
        type=Path,
        default=ROOT / ".local" / "backups",
    )
    backup.add_argument(
        "--key-directory",
        type=Path,
        default=ROOT / ".local" / "backup-keys",
    )
    backup.add_argument(
        "--archive-image",
        type=_validate_image_reference,
        default=ALPINE_IMAGE,
    )
    backup.add_argument(
        "--volume-target",
        type=_validate_container_path,
        default="/source",
    )
    backup.add_argument("--archive-user", type=_validate_container_user)
    restore = commands.add_parser("restore")
    restore.add_argument("encrypted_path", type=Path)
    restore.add_argument("--key-path", type=Path, required=True)
    restore.add_argument("--target-volume", type=_validate_volume_name, required=True)
    restore.add_argument(
        "--archive-image",
        type=_validate_image_reference,
        default=ALPINE_IMAGE,
    )
    restore.add_argument(
        "--volume-target",
        type=_validate_container_path,
        default="/restore",
    )
    restore.add_argument("--archive-user", type=_validate_container_user)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.command == "backup":
        manifest = backup_volume(
            args.volume,
            output_directory=args.output_directory,
            key_directory=args.key_directory,
            archive_image=args.archive_image,
            volume_target=args.volume_target,
            archive_user=args.archive_user,
        )
        print(f"manifest={manifest}")
        print("off_host_copy=required")
        return 0
    restore_volume(
        args.encrypted_path,
        key_path=args.key_path,
        target_volume=args.target_volume,
        archive_image=args.archive_image,
        volume_target=args.volume_target,
        archive_user=args.archive_user,
    )
    print(f"restored_volume={args.target_volume}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
