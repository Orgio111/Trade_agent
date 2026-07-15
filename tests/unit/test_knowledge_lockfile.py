"""Deterministic ownership-evidence tests for the project knowledge lock."""

from __future__ import annotations

from pathlib import Path

import pytest

from packages.knowledge.errors import DriftError
from packages.knowledge.lockfile import (
    build_lock,
    canonical_json_bytes,
    check_lock,
    write_lock,
)
from packages.knowledge.manifest import load_inventory
from tests.fakes.knowledge import write_test_project


def _replace(manifest: Path, old: str, new: str) -> None:
    text = manifest.read_text(encoding="utf-8")
    assert old in text
    manifest.write_text(text.replace(old, new), encoding="utf-8", newline="\n")


def test_non_embedding_owned_file_change_causes_drift(tmp_path: Path) -> None:
    manifest = write_test_project(tmp_path)
    inventory = load_inventory(manifest)
    write_lock(inventory)

    index = tmp_path / "wiki" / "index.md"
    index.write_text(
        index.read_text(encoding="utf-8") + "\n<!-- governed change -->\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(
        DriftError,
        match=r"changed governed files: wiki/index\.md",
    ):
        check_lock(load_inventory(manifest))


def test_quarantined_file_remains_governed_but_is_not_embedded(
    tmp_path: Path,
) -> None:
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

    lock = build_lock(load_inventory(manifest))
    governed = {item["path"]: item for item in lock["governed_files"]}
    sources = {item["path"] for item in lock["sources"]}

    assert governed["src/app.py"]["component"] == "test-component"
    assert governed["src/app.py"]["owner"] == "test-team"
    assert "src/app.py" not in sources


def test_lock_is_deterministic_and_does_not_hash_itself(tmp_path: Path) -> None:
    manifest = write_test_project(tmp_path)
    generated_lock = tmp_path / "project.manifest.lock.json"
    generated_lock.write_text("{}\n", encoding="utf-8", newline="\n")
    _replace(
        manifest,
        'include = ["src/**/*.py", "wiki/index.md", "wiki/content/**/*.md"]',
        'include = ["project.manifest.lock.json", "src/**/*.py", '
        '"wiki/index.md", "wiki/content/**/*.md"]',
    )
    _replace(
        manifest,
        'owned_paths = ["src/**/*.py", "wiki/index.md", "wiki/content/**/*.md"]',
        'owned_paths = ["project.manifest.lock.json", "src/**/*.py", '
        '"wiki/index.md", "wiki/content/**/*.md"]',
    )

    first_inventory = load_inventory(manifest)
    first = build_lock(first_inventory)
    assert "project.manifest.lock.json" in first_inventory.owned_by
    assert "project.manifest.lock.json" not in {
        item["path"] for item in first["governed_files"]
    }

    write_lock(first_inventory, first)
    second = build_lock(load_inventory(manifest))

    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    governed_paths = [item["path"] for item in second["governed_files"]]
    assert governed_paths == sorted(governed_paths)
    assert second["summary"]["governed_file_count"] == len(governed_paths)
    assert second["summary"]["governed_bytes"] == sum(
        item["bytes"] for item in second["governed_files"]
    )
