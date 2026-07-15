"""Fail-closed architecture-manifest validation tests."""

from pathlib import Path

import pytest

import packages.knowledge.manifest as manifest_module
from packages.knowledge.errors import ManifestError, SecretDetectedError
from packages.knowledge.manifest import load_inventory
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
