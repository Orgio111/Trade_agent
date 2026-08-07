"""Redacting configuration preflight for the canonical paper runtime."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

import yaml


ROOT = Path(__file__).resolve().parents[1]
_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
_PLACEHOLDER = re.compile(
    r"(?i)(change[-_ ]?me|placeholder|your[_-]|example|dummy|todo|x{3,}|<[^>]+>)"
)
_SECRET_NAME = re.compile(
    r"(?i)(PASSWORD|SECRET|TOKEN|API_KEY|CREDENTIAL|DATABASE_URL|DSN)"
)
_CANONICAL_ALLOWED = frozenset(
    {
        "POSTGRES_PASSWORD_FILE_PATH",
        "DB_RUNTIME_PASSWORD_FILE_PATH",
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PORT",
        "REDIS_PORT",
        "NATS_PORT",
        "NATS_MONITOR_PORT",
        "CONTROL_PLANE_PORT",
        "RUNTIME_MODE",
        "PAPER_ACCOUNT_ID",
        "OLLAMA_BASE_URL",
        "PAPER_TRADING",
        "CANDIDATE_PROVIDER",
        "ALPHA_CANDIDATE_DIRECTORY",
        "ALPHA_CANDIDATE_ARTIFACT",
        "ALPHA_CANDIDATE_SHA256",
    }
)
_FORBIDDEN_CANONICAL_PREFIXES = (
    "BINANCE_",
    "GROQ_",
    "NVIDIA_",
    "OPENAI_",
    "OPENROUTER_",
    "TELEGRAM_",
)
_REQUIRED_SECRET_FILES = frozenset(
    {"POSTGRES_PASSWORD_FILE_PATH", "DB_RUNTIME_PASSWORD_FILE_PATH"}
)
_INLINE_SECRETS = frozenset({"POSTGRES_PASSWORD", "DB_RUNTIME_PASSWORD"})
_LOCAL_HOSTS = frozenset(
    {"127.0.0.1", "::1", "localhost", "postgres", "nats", "host.docker.internal"}
)


@dataclass(frozen=True)
class EnvRecord:
    name: str
    value: str
    line: int


@dataclass(frozen=True)
class Finding:
    code: str
    name: str
    detail: str


def parse_env(path: Path) -> list[EnvRecord]:
    records: list[EnvRecord] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[7:].strip()
        if "=" not in stripped:
            records.append(EnvRecord(name="", value="", line=line_number))
            continue
        name, value = stripped.split("=", 1)
        records.append(
            EnvRecord(
                name=name.strip(),
                value=value.strip().strip("\"'"),
                line=line_number,
            )
        )
    return records


def _url_finding(name: str, value: str) -> Finding | None:
    parsed = urlsplit(value)
    schemes = {
        "DATABASE_URL": {"postgres", "postgresql"},
        "NATS_URL": {"nats", "tls"},
        "OLLAMA_BASE_URL": {"http", "https"},
    }[name]
    if parsed.scheme.casefold() not in schemes or parsed.hostname is None:
        return Finding("malformed_url", name, "URL scheme or host is invalid")
    if parsed.hostname.casefold().rstrip(".") not in _LOCAL_HOSTS:
        return Finding("remote_dependency", name, "canonical dependency must be local")
    return None


def validate_env(
    records: list[EnvRecord],
    *,
    scope: str,
    allow_placeholders: bool,
    base_directory: Path = ROOT,
) -> list[Finding]:
    findings: list[Finding] = []
    grouped: dict[str, list[EnvRecord]] = {}
    for record in records:
        if not record.name or not _NAME.fullmatch(record.name):
            findings.append(
                Finding(
                    "malformed_entry", f"line:{record.line}", "invalid variable name"
                )
            )
            continue
        grouped.setdefault(record.name, []).append(record)

    for name, entries in sorted(grouped.items()):
        if len(entries) > 1:
            lines = ",".join(str(item.line) for item in entries)
            findings.append(
                Finding(
                    "duplicate_key", name, f"defined more than once at lines {lines}"
                )
            )

    if scope == "canonical":
        for name in sorted(set(grouped) & _INLINE_SECRETS):
            findings.append(
                Finding(
                    "inline_secret_disallowed",
                    name,
                    "canonical secrets must be referenced through read-only files",
                )
            )
        for name in sorted(set(grouped) - _CANONICAL_ALLOWED):
            findings.append(
                Finding("out_of_scope", name, "not admitted by canonical paper runtime")
            )
        for name in sorted(grouped):
            if name.startswith(_FORBIDDEN_CANONICAL_PREFIXES):
                findings.append(
                    Finding(
                        "forbidden_secret_scope", name, "not allowed in paper runtime"
                    )
                )

    for name in sorted(_REQUIRED_SECRET_FILES):
        entries = grouped.get(name, [])
        if not entries:
            findings.append(
                Finding("missing_required", name, "required variable is absent")
            )
            continue
        value = entries[0].value
        if not value:
            findings.append(
                Finding("empty_secret_path", name, "required secret path is empty")
            )
        elif allow_placeholders:
            continue
        elif _PLACEHOLDER.search(value):
            findings.append(
                Finding(
                    "placeholder_secret_path",
                    name,
                    "required secret path is a placeholder",
                )
            )
        else:
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = base_directory / candidate
            try:
                resolved = candidate.resolve(strict=True)
            except OSError:
                findings.append(
                    Finding(
                        "missing_secret_file",
                        name,
                        "required secret file does not exist",
                    )
                )
                continue
            if candidate.is_symlink() or not resolved.is_file():
                findings.append(
                    Finding(
                        "invalid_secret_file",
                        name,
                        "required secret path must be a regular non-symlink file",
                    )
                )
                continue
            payload = resolved.read_bytes()
            if len(payload) > 4096:
                findings.append(
                    Finding(
                        "oversized_secret_file",
                        name,
                        "required secret file exceeds 4096 bytes",
                    )
                )
            elif len(payload.strip()) < 16:
                findings.append(
                    Finding(
                        "weak_secret_length",
                        name,
                        "required secret file is shorter than 16 bytes",
                    )
                )
            elif _PLACEHOLDER.search(payload.decode("utf-8", errors="replace").strip()):
                findings.append(
                    Finding(
                        "placeholder_secret",
                        name,
                        "required secret file contains a placeholder",
                    )
                )

    for name in sorted(set(grouped) & _INLINE_SECRETS):
        if grouped[name][0].value:
            findings.append(
                Finding(
                    "inline_secret_present",
                    name,
                    "inline secret material is not admitted",
                )
            )

    for name, entries in grouped.items():
        value = entries[0].value
        file_reference = name.endswith("_FILE_PATH")
        if _SECRET_NAME.search(name) and not value and not file_reference:
            findings.append(
                Finding("empty_secret", name, "secret-bearing variable is empty")
            )
        if (
            _SECRET_NAME.search(name)
            and not file_reference
            and value
            and not allow_placeholders
            and _PLACEHOLDER.search(value)
        ):
            findings.append(
                Finding(
                    "placeholder_secret",
                    name,
                    "secret-bearing variable is a placeholder",
                )
            )
        if name in {"DATABASE_URL", "NATS_URL", "OLLAMA_BASE_URL"} and value:
            finding = _url_finding(name, value)
            if finding:
                findings.append(finding)

    paper = grouped.get("PAPER_TRADING")
    if paper and paper[0].value.casefold() not in {"1", "true", "yes", "on"}:
        findings.append(
            Finding("paper_mode_conflict", "PAPER_TRADING", "must be explicitly true")
        )
    runtime_mode = grouped.get("RUNTIME_MODE")
    if runtime_mode and runtime_mode[0].value not in {"paper_live", "replay"}:
        findings.append(
            Finding(
                "runtime_mode_conflict",
                "RUNTIME_MODE",
                "canonical runtime admits only paper_live or replay",
            )
        )
    provider = grouped.get("CANDIDATE_PROVIDER")
    if provider:
        provider_value = provider[0].value
        if provider_value not in {
            "ollama",
            "deterministic_baseline",
            "alpha_shadow",
        }:
            findings.append(
                Finding(
                    "candidate_provider_invalid",
                    "CANDIDATE_PROVIDER",
                    "must be ollama, deterministic_baseline, or alpha_shadow",
                )
            )
        artifact_directory = grouped.get("ALPHA_CANDIDATE_DIRECTORY")
        artifact_path = grouped.get("ALPHA_CANDIDATE_ARTIFACT")
        artifact_sha256 = grouped.get("ALPHA_CANDIDATE_SHA256")
        if provider_value == "alpha_shadow":
            if (
                not artifact_directory
                or not artifact_directory[0].value
                or not artifact_path
                or artifact_path[0].value != "/run/alpha-candidate/candidate.joblib"
                or not artifact_sha256
                or re.fullmatch(r"[0-9a-f]{64}", artifact_sha256[0].value) is None
            ):
                findings.append(
                    Finding(
                        "alpha_candidate_incomplete",
                        "CANDIDATE_PROVIDER",
                        "alpha_shadow requires directory, exact container path, and SHA-256",
                    )
                )
            elif not allow_placeholders:
                directory = Path(artifact_directory[0].value)
                if not directory.is_absolute():
                    directory = base_directory / directory
                if directory.is_symlink() or not directory.resolve().is_dir():
                    findings.append(
                        Finding(
                            "alpha_candidate_directory_invalid",
                            "ALPHA_CANDIDATE_DIRECTORY",
                            "must reference a regular non-symlink directory",
                        )
                    )
        elif (
            artifact_path
            and artifact_path[0].value
            or artifact_sha256
            and artifact_sha256[0].value
        ):
            findings.append(
                Finding(
                    "alpha_candidate_scope_conflict",
                    "CANDIDATE_PROVIDER",
                    "artifact path and digest require alpha_shadow",
                )
            )
    if "NVIDIA_API_KEY" in grouped and "NVIDIA_NIM_API_KEY" in grouped:
        findings.append(
            Finding(
                "provider_alias_conflict",
                "NVIDIA_NIM_API_KEY",
                "define only NVIDIA_NIM_API_KEY; NVIDIA_API_KEY is a legacy alias",
            )
        )
    return findings


def validate_compose(path: Path) -> list[Finding]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    services = payload.get("services", {})
    findings: list[Finding] = []
    for service_name, service in services.items():
        if service.get("profiles"):
            continue
        environment = service.get("environment", {})
        if isinstance(environment, list):
            names = {str(item).split("=", 1)[0] for item in environment}
        else:
            names = {str(name) for name in environment}
        for name in sorted(names):
            if name.startswith(_FORBIDDEN_CANONICAL_PREFIXES):
                findings.append(
                    Finding(
                        "compose_forbidden_secret",
                        f"{service_name}:{name}",
                        "profile-free paper service receives a forbidden secret",
                    )
                )
        for volume in service.get("volumes", []):
            if ".env" in str(volume):
                findings.append(
                    Finding(
                        "writable_env_mount",
                        service_name,
                        "profile-free service must not mount an environment file",
                    )
                )
    return findings


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--env-file", type=Path, default=ROOT / ".env")
    result.add_argument(
        "--compose-file", type=Path, default=ROOT / "docker-compose.yml"
    )
    result.add_argument("--scope", choices=("canonical", "legacy"), default="canonical")
    result.add_argument(
        "--allow-placeholders",
        action="store_true",
        help="validate a tracked template without accepting it for runtime use",
    )
    return result


def main() -> int:
    args = parser().parse_args()
    if not args.env_file.is_file():
        payload = {
            "status": "error",
            "checked_names": [],
            "findings": [
                asdict(
                    Finding(
                        "missing_env_file",
                        str(args.env_file),
                        "configuration file does not exist",
                    )
                )
            ],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2
    records = parse_env(args.env_file)
    findings = validate_env(
        records,
        scope=args.scope,
        allow_placeholders=args.allow_placeholders,
        base_directory=ROOT,
    )
    findings.extend(validate_compose(args.compose_file))
    payload = {
        "status": "error" if findings else "ok",
        "scope": args.scope,
        "checked_names": sorted({record.name for record in records if record.name}),
        "findings": [asdict(finding) for finding in findings],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 2 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
