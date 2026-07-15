"""Fail-closed architecture-manifest validation tests."""

from pathlib import Path
from types import SimpleNamespace

import pytest

import packages.knowledge.manifest as manifest_module
from packages.knowledge.errors import ManifestError, SecretDetectedError
from packages.knowledge.lockfile import build_lock
from packages.knowledge.manifest import load_inventory
from packages.knowledge.security import scan_secrets
from tests.fakes.knowledge import write_test_project


def _replace(manifest: Path, old: str, new: str) -> None:
    text = manifest.read_text(encoding="utf-8")
    assert old in text
    manifest.write_text(text.replace(old, new), encoding="utf-8", newline="\n")


def test_duplicate_ownership_is_rejected(tmp_path: Path) -> None:
    manifest = write_test_project(tmp_path)
    with manifest.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            """
[[components]]
id = "duplicate-component"
owner = "other-team"
primary_responsibility = "Create an intentional duplicate ownership conflict"
owned_paths = ["src/app.py"]
tests = []
documentation = ["wiki/content/test-component.md"]
architecture = ["wiki/content/test-component.md"]
embedding_inputs = []
"""
        )

    with pytest.raises(ManifestError, match="duplicate ownership for src/app.py"):
        load_inventory(manifest)


def test_unowned_governed_file_is_rejected(tmp_path: Path) -> None:
    manifest = write_test_project(tmp_path)
    (tmp_path / "src" / "unowned.py").write_text("VALUE = 1\n", encoding="utf-8")
    _replace(manifest, 'owned_paths = ["src/**/*.py",', 'owned_paths = ["src/app.py",')

    with pytest.raises(ManifestError, match="unowned governed paths") as failure:
        load_inventory(manifest)

    assert "src/unowned.py" in str(failure.value)


def test_new_tracked_production_root_is_governed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = write_test_project(tmp_path)
    service = tmp_path / "new_service" / "app.py"
    service.parent.mkdir()
    service.write_text("VALUE = 1\n", encoding="utf-8")
    _replace(
        manifest,
        'include = ["src/**/*.py", "wiki/index.md", "wiki/content/**/*.md"]',
        'include = ["src/**/*.py", "wiki/index.md", "wiki/content/**/*.md"]\n'
        'tracked_include = ["**/*.py"]',
    )
    monkeypatch.setattr(
        manifest_module,
        "_git_tracked_paths",
        lambda _root: {"src/app.py", "new_service/app.py"},
    )

    with pytest.raises(ManifestError, match="unowned governed paths") as failure:
        load_inventory(manifest)

    assert "new_service/app.py" in str(failure.value)


def test_explicit_generated_exclusion_is_not_governed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = write_test_project(tmp_path)
    generated = tmp_path / "generated" / "cache.py"
    generated.parent.mkdir()
    generated.write_text("VALUE = 1\n", encoding="utf-8")
    _replace(
        manifest,
        'include = ["src/**/*.py", "wiki/index.md", "wiki/content/**/*.md"]',
        'include = ["src/**/*.py", "wiki/index.md", "wiki/content/**/*.md"]\n'
        'tracked_include = ["**/*.py"]\n'
        'tracked_exclude = ["generated/**/*"]',
    )
    monkeypatch.setattr(
        manifest_module,
        "_git_tracked_paths",
        lambda _root: {"src/app.py", "generated/cache.py"},
    )

    inventory = load_inventory(manifest)

    assert "generated/cache.py" not in inventory.owned_by


def test_tracked_pytest_temp_artifact_is_forbidden(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = write_test_project(tmp_path)
    _replace(
        manifest,
        'include = ["src/**/*.py", "wiki/index.md", "wiki/content/**/*.md"]',
        'include = ["src/**/*.py", "wiki/index.md", "wiki/content/**/*.md"]\n'
        'forbidden_repository_paths = [".pytest_tmp*/**/*"]',
    )
    monkeypatch.setattr(
        manifest_module,
        "_git_tracked_paths",
        lambda _root: {".pytest_tmp_full_root/result.txt"},
    )

    with pytest.raises(
        ManifestError,
        match="forbidden generated/runtime paths are tracked",
    ):
        load_inventory(manifest)


def test_git_inventory_retains_broken_symlinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(
        manifest_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=b"broken-link.py\x00deleted.py\x00",
        ),
    )
    monkeypatch.setattr(
        manifest_module.os.path,
        "lexists",
        lambda path: Path(path).name == "broken-link.py",
    )

    assert manifest_module._git_tracked_paths(tmp_path) == {"broken-link.py"}


def test_missing_documentation_is_rejected(tmp_path: Path) -> None:
    manifest = write_test_project(tmp_path)
    _replace(
        manifest,
        'documentation = ["wiki/content/test-component.md"]',
        'documentation = ["wiki/content/missing.md"]',
    )

    with pytest.raises(ManifestError, match="documentation pattern matched no files"):
        load_inventory(manifest)


def test_repository_path_traversal_is_rejected(tmp_path: Path) -> None:
    manifest = write_test_project(tmp_path)
    _replace(
        manifest,
        'include = ["src/**/*.py", "wiki/index.md", "wiki/content/**/*.md"]',
        'include = ["../*.py"]',
    )

    with pytest.raises(ManifestError, match="escapes the root"):
        load_inventory(manifest)


def test_case_colliding_coverage_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulate a collision so this regression runs on case-insensitive Windows too."""

    manifest = write_test_project(tmp_path)
    original_expand = manifest_module._expand_patterns

    def expand_with_collision(
        root: Path,
        patterns: tuple[str, ...],
        *,
        label: str,
        require_each: bool = True,
    ) -> set[str]:
        if label == "coverage.include":
            return {"src/Component.py", "src/component.py"}
        return original_expand(
            root,
            patterns,
            label=label,
            require_each=require_each,
        )

    monkeypatch.setattr(manifest_module, "_expand_patterns", expand_with_collision)

    with pytest.raises(ManifestError, match="case-colliding repository paths"):
        load_inventory(manifest)


def test_secret_detection_redacts_the_secret_value(tmp_path: Path) -> None:
    secret = "sk-THIS_VALUE_MUST_NEVER_APPEAR_1234567890"
    manifest = write_test_project(
        tmp_path,
        source_text=f'API_KEY = "{secret}"\n',
    )

    with pytest.raises(SecretDetectedError) as failure:
        load_inventory(manifest)

    assert "src/app.py:1:" in str(failure.value)
    assert secret not in str(failure.value)


@pytest.mark.parametrize(
    ("path", "line", "credential"),
    [
        ("docker-compose.yml", "POSTGRES_PASSWORD: secret", "secret"),
        ("settings.toml", "OPENAI_API_KEY = sk-placeholder-key", "sk-placeholder-key"),
        (".env.local", "GF_SECURITY_ADMIN_PASSWORD=quantex123", "quantex123"),
        ("compose.yaml", "- POSTGRES_PASSWORD=secret", "secret"),
        (
            "docker-compose.yaml",
            "POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-secret}",
            "${POSTGRES_PASSWORD:-secret}",
        ),
    ],
)
def test_structured_config_credentials_are_redacted(
    path: str,
    line: str,
    credential: str,
) -> None:
    with pytest.raises(SecretDetectedError) as failure:
        scan_secrets(path, line + "\n")

    assert f"{path}:1:" in str(failure.value)
    assert credential not in str(failure.value)


@pytest.mark.parametrize(
    "line",
    [
        "POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}",
        "POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:?POSTGRES_PASSWORD is required}",
        "POSTGRES_PASSWORD: ${POSTGRES_PASSWORD?POSTGRES_PASSWORD is required}",
        "OPENAI_API_KEY: $OPENAI_API_KEY",
        "CLIENT_SECRET: var.client_secret",
        "SERVICE_TOKEN: {{ .Values.serviceToken }}",
        "BINANCE_API_KEY: ${{ secrets.BINANCE_API_KEY }}",
    ],
)
def test_structured_config_secret_references_are_allowed(line: str) -> None:
    scan_secrets("config.yml", line + "\n")


def test_quarantine_is_frozen_and_excluded_from_lock_sources(tmp_path: Path) -> None:
    manifest = write_test_project(tmp_path)
    with manifest.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            """
[[quarantines]]
id = "legacy-vector"
reason = "Legacy trading vector module is isolated"
scope = ["src/app.py"]
baseline_paths = ["src/app.py"]
"""
        )

    inventory = load_inventory(manifest)
    lock = build_lock(inventory)

    assert inventory.quarantined_paths == frozenset({"src/app.py"})
    assert "src/app.py" not in {source["path"] for source in lock["sources"]}


@pytest.mark.parametrize("collection", ["abc-", "abc_"])
def test_collection_name_must_end_alphanumeric(
    tmp_path: Path,
    collection: str,
) -> None:
    manifest = write_test_project(tmp_path)
    _replace(
        manifest,
        'collection = "test_knowledge_v1"',
        f'collection = "{collection}"',
    )

    with pytest.raises(ManifestError, match="collection"):
        load_inventory(manifest)


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("ollama_base_url", "https://models.example.com"),
        ("chroma_host", "chroma.example.com"),
    ],
)
def test_remote_knowledge_endpoints_are_rejected_by_manifest(
    tmp_path: Path,
    setting: str,
    value: str,
) -> None:
    options = {setting: value}
    manifest = write_test_project(tmp_path, **options)

    with pytest.raises(ManifestError, match="loopback"):
        load_inventory(manifest)
