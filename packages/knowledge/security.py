"""Fail-closed source, path, secret, and import validation."""

from __future__ import annotations

import ast
import math
import re
from collections import Counter
from pathlib import Path, PurePosixPath

from .errors import ManifestError, SecretDetectedError


FORBIDDEN_PARTS = frozenset(
    {
        ".git",
        ".local",
        ".obsidian",
        ".pytest_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "target",
    }
)
FORBIDDEN_ROOTS = frozenset(
    {"backtest_results", "data", "logs", "memory", "models", "raw"}
)
FORBIDDEN_SUFFIXES = frozenset(
    {
        ".crt",
        ".db",
        ".dll",
        ".dylib",
        ".exe",
        ".key",
        ".log",
        ".pem",
        ".pyc",
        ".pyo",
        ".sqlite",
        ".sqlite3",
        ".so",
        ".tmp",
    }
)
ALLOWED_TEXT_SUFFIXES = frozenset(
    {
        ".css",
        ".conf",
        ".go",
        ".html",
        ".json",
        ".md",
        ".ps1",
        ".py",
        ".rs",
        ".sh",
        ".sql",
        ".tf",
        ".toml",
        ".ts",
        ".tsx",
        ".txt",
        ".yaml",
        ".yml",
    }
)

_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "private key material",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    ),
    (
        "credential-bearing URI",
        re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^\s/:@]{3,}:[^\s/@]{8,}@"),
    ),
    (
        "literal credential assignment",
        re.compile(
            r"(?i)\b(?:api[_-]?key|api[_-]?secret|access[_-]?token|auth[_-]?token|password)"
            r"\s*[:=]\s*['\"]([^'\"\r\n]{16,})['\"]"
        ),
    ),
    (
        "known token prefix",
        re.compile(
            r"\b(?:sk-[A-Za-z0-9_-]{20,}|gsk_[A-Za-z0-9_-]{20,}|nvapi-[A-Za-z0-9_-]{20,})\b"
        ),
    ),
)

_STRUCTURED_CONFIG_SUFFIXES = frozenset({".toml", ".yaml", ".yml"})
_CONFIG_ASSIGNMENT = re.compile(
    r"^\s*(?:-\s*)?(?P<key>[A-Za-z][A-Za-z0-9_.-]*)\s*[:=]\s*(?P<value>.+?)\s*$"
)
_SAFE_CONFIG_REFERENCES = (
    re.compile(r"^\$[A-Za-z_][A-Za-z0-9_]*$"),
    re.compile(r"^\$\{[A-Za-z_][A-Za-z0-9_]*(?:(?::)?\?[^{}\r\n]+)?\}$"),
    re.compile(r"^(?:var|local|module|data)\.[A-Za-z0-9_.-]+$"),
    re.compile(r"^\{\{.+\}\}$"),
    re.compile(r"^\$\{\{.+\}\}$"),
    re.compile(r"^\*[A-Za-z_][A-Za-z0-9_-]*$"),
)
_NON_CREDENTIAL_LITERALS = frozenset({"", "false", "none", "null", "true", "~"})


def validate_relative_pattern(pattern: str) -> None:
    """Reject absolute, Windows-specific, and repository-escaping globs."""

    if not pattern or "\\" in pattern:
        raise ManifestError(f"invalid POSIX repository pattern: {pattern!r}")
    candidate = PurePosixPath(pattern)
    if candidate.is_absolute() or ".." in candidate.parts or "\x00" in pattern:
        raise ManifestError(f"repository pattern escapes the root: {pattern!r}")


def validate_source_path(root: Path, path: Path) -> str:
    """Resolve a source and return its safe repository-relative POSIX path."""

    if path.is_symlink():
        raise ManifestError(f"symlink sources are forbidden: {path}")
    resolved_root = root.resolve()
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ManifestError(f"source escapes repository root: {path}") from exc
    posix = relative.as_posix()
    lowered_parts = {part.lower() for part in relative.parts}
    wiki_raw = len(relative.parts) > 1 and tuple(
        part.lower() for part in relative.parts[:2]
    ) == ("wiki", "raw")
    if (
        lowered_parts & FORBIDDEN_PARTS
        or relative.parts[0].lower() in FORBIDDEN_ROOTS
        or wiki_raw
    ):
        raise ManifestError(f"temporary/private source is forbidden: {posix}")
    if (
        path.name.lower().startswith(".env")
        or path.suffix.lower() in FORBIDDEN_SUFFIXES
    ):
        raise ManifestError(f"secret/runtime source is forbidden: {posix}")
    if path.suffix.lower() not in ALLOWED_TEXT_SUFFIXES:
        raise ManifestError(f"unsupported embedding source type: {posix}")
    return posix


def _shannon_entropy(value: str) -> float:
    counts = Counter(value)
    length = len(value)
    return -sum(
        (count / length) * math.log2(count / length) for count in counts.values()
    )


def _is_structured_config(path: str) -> bool:
    candidate = PurePosixPath(path)
    name = candidate.name.lower()
    return (
        candidate.suffix.lower() in _STRUCTURED_CONFIG_SUFFIXES
        or name == ".env"
        or name.startswith(".env.")
    )


def _is_sensitive_config_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
    if normalized.endswith("_file"):
        return False
    parts = set(normalized.split("_"))
    return bool(parts & {"password", "passphrase", "secret", "token"}) or any(
        marker in normalized
        for marker in ("api_key", "private_key", "access_key", "client_secret")
    )


def _literal_config_credential(line: str) -> bool:
    match = _CONFIG_ASSIGNMENT.fullmatch(line)
    if match is None or not _is_sensitive_config_key(match.group("key")):
        return False
    value = match.group("value").strip()
    if value[:1] in {"'", '"'} and len(value) >= 2 and value[-1] == value[0]:
        value = value[1:-1].strip()
    else:
        value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
    lowered = value.lower()
    if lowered in _NON_CREDENTIAL_LITERALS:
        return False
    return not any(pattern.fullmatch(value) for pattern in _SAFE_CONFIG_REFERENCES)


def scan_secrets(path: str, text: str) -> None:
    """Raise without ever placing the matched value in the error message."""

    for line_number, line in enumerate(text.splitlines(), start=1):
        if _is_structured_config(path) and _literal_config_credential(line):
            raise SecretDetectedError(
                f"{path}:{line_number}: literal credential in structured config"
            )
        for reason, pattern in _SECRET_PATTERNS:
            if pattern.search(line):
                raise SecretDetectedError(f"{path}:{line_number}: {reason}")

        # Long, mixed-alphabet quoted literals are suspicious even without a known prefix.
        for candidate in re.findall(r"['\"]([A-Za-z0-9_+/=-]{40,})['\"]", line):
            alphabets = sum(
                bool(re.search(expression, candidate))
                for expression in (r"[a-z]", r"[A-Z]", r"[0-9]")
            )
            if alphabets >= 3 and _shannon_entropy(candidate) >= 4.2:
                raise SecretDetectedError(f"{path}:{line_number}: high-entropy literal")


def scan_forbidden_imports(path: str, text: str, forbidden: tuple[str, ...]) -> None:
    """Enforce local-only dependency boundaries with Python's parser."""

    if not forbidden or not path.endswith(".py"):
        return
    try:
        tree = ast.parse(text, filename=path)
    except SyntaxError as exc:
        raise ManifestError(
            f"cannot parse governed Python source {path}: {exc.msg}"
        ) from exc
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".", 1)[0])
    blocked = sorted(imported & set(forbidden))
    if blocked:
        raise ManifestError(f"{path}: forbidden import(s): {', '.join(blocked)}")
