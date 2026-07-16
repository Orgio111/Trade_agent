"""Runtime authority and deployment reachability manifest tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from packages.knowledge.authority import authority_summary
from packages.knowledge.cli import main
from packages.knowledge.errors import ManifestError
from packages.knowledge.manifest import load_inventory
from packages.knowledge.security import endpoint_host
from tests.fakes.knowledge import write_test_project


def _replace(manifest: Path, old: str, new: str) -> None:
    text = manifest.read_text(encoding="utf-8")
    assert old in text
    manifest.write_text(text.replace(old, new), encoding="utf-8", newline="\n")


def _add_runtime_policy(manifest: Path) -> None:
    _replace(
        manifest,
        "[coverage]\n",
        """[runtime_policy]
local_inference_hosts = ["127.0.0.1", "localhost"]
allowed_models = ["qwen3:8b", "nomic-embed-text"]
allowed_modes = ["offline", "paper", "replay", "research"]

[coverage]
""",
    )


def _make_active(manifest: Path, *, extra: str = "") -> None:
    _replace(
        manifest,
        'primary_responsibility = "Provide deterministic offline test behavior"\n',
        'primary_responsibility = "Provide deterministic offline test behavior"\n'
        'runtime_status = "active"\n'
        'entrypoints = [{ path = "src/app.py", kind = "cli", default = true }]\n'
        f"{extra}",
    )


def _add_component_source(root: Path, name: str) -> None:
    (root / "src" / f"{name}.py").write_text(
        f"COMPONENT = {name!r}\n",
        encoding="utf-8",
        newline="\n",
    )


def test_legacy_manifest_defaults_components_to_experimental(tmp_path: Path) -> None:
    manifest = write_test_project(tmp_path)

    inventory = load_inventory(manifest)

    component = inventory.manifest.components[0]
    assert component.runtime_status == "experimental"
    assert component.entrypoints == ()
    assert component.authorities == ()


def test_active_component_cannot_reach_quarantined_dependency(
    tmp_path: Path,
) -> None:
    manifest = write_test_project(tmp_path)
    _add_component_source(tmp_path, "middle")
    _add_component_source(tmp_path, "legacy")
    _replace(manifest, 'owned_paths = ["src/**/*.py",', 'owned_paths = ["src/app.py",')
    _make_active(manifest, extra='depends_on = ["middle-component"]\n')
    with manifest.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            """
[[components]]
id = "middle-component"
owner = "test-middle"
primary_responsibility = "Provide intermediate runtime behavior"
runtime_status = "experimental"
depends_on = ["legacy-component"]
owned_paths = ["src/middle.py"]
tests = []
documentation = ["wiki/content/test-component.md"]
architecture = ["wiki/content/test-component.md"]

[[components]]
id = "legacy-component"
owner = "test-legacy"
primary_responsibility = "Preserve isolated legacy runtime behavior"
runtime_status = "quarantined"
owned_paths = ["src/legacy.py"]
tests = []
documentation = ["wiki/content/test-component.md"]
architecture = ["wiki/content/test-component.md"]
"""
        )

    with pytest.raises(ManifestError, match="reaches quarantined component"):
        load_inventory(manifest)


def test_sensitive_authority_must_have_one_owner(tmp_path: Path) -> None:
    manifest = write_test_project(tmp_path)
    _add_component_source(tmp_path, "second")
    _replace(manifest, 'owned_paths = ["src/**/*.py",', 'owned_paths = ["src/app.py",')
    _make_active(
        manifest,
        extra=('authorities = [{ id = "risk.authorization", sensitive = true }]\n'),
    )
    with manifest.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            """
[[components]]
id = "second-component"
owner = "test-second"
primary_responsibility = "Provide a conflicting sensitive authority"
runtime_status = "experimental"
authorities = [{ id = "risk.authorization", sensitive = true }]
owned_paths = ["src/second.py"]
tests = []
documentation = ["wiki/content/test-component.md"]
architecture = ["wiki/content/test-component.md"]
"""
        )

    with pytest.raises(ManifestError, match="sensitive authority.*multiple owners"):
        load_inventory(manifest)


def test_quarantined_component_cannot_hold_sensitive_authority(
    tmp_path: Path,
) -> None:
    manifest = write_test_project(tmp_path)
    _replace(
        manifest,
        'primary_responsibility = "Provide deterministic offline test behavior"\n',
        'primary_responsibility = "Provide deterministic offline test behavior"\n'
        'runtime_status = "quarantined"\n'
        'authorities = [{ id = "execution.order", sensitive = true }]\n',
    )

    with pytest.raises(ManifestError, match="quarantined component cannot hold"):
        load_inventory(manifest)


def test_active_inference_rejects_remote_egress(tmp_path: Path) -> None:
    manifest = write_test_project(tmp_path)
    _add_runtime_policy(manifest)
    _make_active(
        manifest,
        extra=(
            'authorities = [{ id = "inference.reasoning", sensitive = true }]\n'
            'allowed_models = ["qwen3:8b"]\n'
            'allowed_egress = ["https://models.example.com"]\n'
        ),
    )

    with pytest.raises(ManifestError, match="inference egress is not locally allowed"):
        load_inventory(manifest)


def test_active_inference_rejects_model_outside_allowlist(tmp_path: Path) -> None:
    manifest = write_test_project(tmp_path)
    secret = "sk-THIS_MUST_NOT_APPEAR_12345678901234567890"
    _add_runtime_policy(manifest)
    _make_active(
        manifest,
        extra=(
            'authorities = [{ id = "inference.reasoning", sensitive = true }]\n'
            f'allowed_models = ["{secret}"]\n'
            'allowed_egress = ["http://127.0.0.1:11434"]\n'
        ),
    )

    with pytest.raises(
        ManifestError, match="model.*outside runtime allowlist"
    ) as failure:
        load_inventory(manifest)
    assert secret not in str(failure.value)


def test_egress_validation_redacts_embedded_credentials() -> None:
    secret = "DO_NOT_EXPOSE_THIS_PASSWORD"

    with pytest.raises(ManifestError) as failure:
        endpoint_host(f"http://agent:{secret}@127.0.0.1:11434")

    assert secret not in str(failure.value)


def test_active_default_entrypoint_must_exist(tmp_path: Path) -> None:
    manifest = write_test_project(tmp_path)
    _replace(
        manifest,
        'primary_responsibility = "Provide deterministic offline test behavior"\n',
        'primary_responsibility = "Provide deterministic offline test behavior"\n'
        'runtime_status = "active"\n'
        'entrypoints = [{ path = "src/missing.py", kind = "cli", default = true }]\n',
    )

    with pytest.raises(ManifestError, match="active entrypoint does not exist"):
        load_inventory(manifest)


def test_active_component_cannot_own_path_level_quarantine(tmp_path: Path) -> None:
    manifest = write_test_project(tmp_path)
    _make_active(manifest)
    with manifest.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            """
[[quarantines]]
id = "legacy-path"
reason = "Keep this legacy runtime path isolated"
scope = ["src/app.py"]
baseline_paths = ["src/app.py"]
"""
        )

    with pytest.raises(ManifestError, match="active component owns quarantined paths"):
        load_inventory(manifest)


def test_valid_local_inference_policy_has_stable_summary(tmp_path: Path) -> None:
    manifest = write_test_project(tmp_path)
    _add_runtime_policy(manifest)
    _make_active(
        manifest,
        extra=(
            'authorities = [{ id = "inference.reasoning", sensitive = true }]\n'
            'allowed_modes = ["offline"]\n'
            'allowed_models = ["qwen3:8b"]\n'
            'allowed_egress = ["http://localhost:11434"]\n'
        ),
    )

    loaded = load_inventory(manifest).manifest

    assert authority_summary(loaded) == {
        "active_components": ["test-component"],
        "quarantined_components": [],
        "sensitive_authorities": ["inference.reasoning"],
        "default_entrypoints": ["test-component:src/app.py"],
    }


def test_validate_command_runs_without_lock_or_network(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = write_test_project(tmp_path)
    _make_active(manifest)

    assert main(["--manifest", str(manifest), "validate"]) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "valid"
    assert result["active_components"] == ["test-component"]
