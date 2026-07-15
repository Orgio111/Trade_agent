"""Load and enforce the executable architecture manifest."""

from __future__ import annotations

import os
import re
import subprocess
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pydantic import ValidationError

from .chunking import canonical_text, sha256_text
from .errors import ManifestError
from .models import Component, ProjectManifest
from .security import (
    scan_forbidden_imports,
    scan_secrets,
    validate_relative_pattern,
    validate_source_path,
)


@dataclass(frozen=True, slots=True)
class SourceFile:
    """A validated embedding input and its architecture owner."""

    path: str
    absolute_path: Path
    component_id: str
    text: str
    sha256: str
    byte_count: int
    categories: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GovernedFile:
    """Canonical evidence for one owned repository text file."""

    path: str
    absolute_path: Path
    component_id: str
    owner: str
    text: str
    sha256: str
    byte_count: int


@dataclass(frozen=True, slots=True)
class ManifestInventory:
    """Resolved repository inventory used by both CI and live sync."""

    root: Path
    manifest_path: Path
    manifest: ProjectManifest
    manifest_sha256: str
    owned_by: dict[str, str]
    quarantined_paths: frozenset[str]
    governed_files: tuple[GovernedFile, ...]
    sources: tuple[SourceFile, ...]


def _expand_patterns(
    root: Path,
    patterns: tuple[str, ...],
    *,
    label: str,
    require_each: bool = True,
) -> set[str]:
    result: set[str] = set()
    for pattern in patterns:
        validate_relative_pattern(pattern)
        matches: set[str] = set()
        for candidate in root.glob(pattern):
            if not candidate.is_file():
                continue
            resolved = candidate.resolve(strict=True)
            try:
                relative = resolved.relative_to(root.resolve())
            except ValueError as exc:
                raise ManifestError(
                    f"{label} path escapes repository: {candidate}"
                ) from exc
            if candidate.is_symlink():
                raise ManifestError(
                    f"{label} contains a symlink: {relative.as_posix()}"
                )
            matches.add(relative.as_posix())
        if require_each and not matches:
            raise ManifestError(f"{label} pattern matched no files: {pattern}")
        result.update(matches)
    return result


def _read_stable_utf8(path: Path, relative_path: str) -> str:
    try:
        before = path.stat()
        text = path.read_text(encoding="utf-8")
        after = path.stat()
    except UnicodeDecodeError as exc:
        raise ManifestError(
            f"governed file is not valid UTF-8: {relative_path}"
        ) from exc
    except OSError as exc:
        raise ManifestError(f"cannot read governed file: {relative_path}") from exc
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ManifestError(
            f"governed file changed while being scanned: {relative_path}"
        )
    return canonical_text(text)


def _configured_lock_path(root: Path, configured: str) -> str:
    """Return the safe repository-relative lock path without requiring it to exist."""

    validate_relative_pattern(configured)
    if any(character in configured for character in "*?["):
        raise ManifestError("knowledge.lock_file must be one exact repository path")
    candidate = (root / configured).resolve()
    try:
        return candidate.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ManifestError("knowledge.lock_file escapes the repository") from exc


def _validate_wiki_document(root: Path, relative_path: str, index_text: str) -> None:
    path = root / relative_path
    text = _read_stable_utf8(path, relative_path)
    if not text.startswith("---\n"):
        raise ManifestError(f"wiki document lacks YAML frontmatter: {relative_path}")
    end = text.find("\n---\n", 4)
    if end == -1:
        raise ManifestError(
            f"wiki document has unterminated frontmatter: {relative_path}"
        )
    frontmatter = text[4:end]
    required = ("title", "type", "tags", "created", "updated", "status")
    for field in required:
        if not any(line.startswith(f"{field}:") for line in frontmatter.splitlines()):
            raise ManifestError(f"{relative_path}: frontmatter missing {field}")
    if relative_path.startswith("wiki/content/"):
        slug = Path(relative_path).stem
        if f"[[{slug}]]" not in index_text:
            raise ManifestError(f"wiki/index.md does not catalog [[{slug}]]")


def _git_tracked_paths(root: Path) -> set[str] | None:
    if not (root / ".git").exists():
        return None
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise ManifestError("cannot read the Git tracked-file inventory")
    tracked = {
        item.decode("utf-8").replace("\\", "/")
        for item in completed.stdout.split(b"\x00")
        if item
    }
    # A file removed in the current worktree remains in Git's index until the
    # removal is committed. Drift checks describe the tree being validated, so
    # do not report already-removed generated artifacts as live repository data.
    return {path for path in tracked if os.path.lexists(root / Path(path))}


@lru_cache(maxsize=512)
def _compile_repository_pattern(pattern: str) -> re.Pattern[str]:
    """Compile the manifest's small POSIX-glob dialect with true ``**`` support."""

    expression: list[str] = ["^"]
    index = 0
    while index < len(pattern):
        character = pattern[index]
        if character == "*":
            if index + 1 < len(pattern) and pattern[index + 1] == "*":
                index += 2
                if index < len(pattern) and pattern[index] == "/":
                    expression.append("(?:.*/)?")
                    index += 1
                else:
                    expression.append(".*")
                continue
            expression.append("[^/]*")
        elif character == "?":
            expression.append("[^/]")
        else:
            expression.append(re.escape(character))
        index += 1
    expression.append("$")
    return re.compile("".join(expression))


def _matches_pattern(path: str, pattern: str) -> bool:
    return _compile_repository_pattern(pattern).fullmatch(path) is not None


def _tracked_coverage(
    tracked: set[str] | None,
    *,
    include: tuple[str, ...],
    exclude: tuple[str, ...],
) -> set[str]:
    """Select every relevant tracked file, including files under unknown new roots."""

    for pattern in (*include, *exclude):
        validate_relative_pattern(pattern)
    if tracked is None or not include:
        return set()
    return {
        item
        for item in tracked
        if any(_matches_pattern(item, pattern) for pattern in include)
        and not any(_matches_pattern(item, pattern) for pattern in exclude)
    }


def _categories_for(
    path: str,
    *,
    documentation: set[str],
    architecture: set[str],
) -> tuple[str, ...]:
    categories: list[str] = []
    if path in documentation:
        categories.append("documentation")
    if path in architecture:
        categories.append("architecture")
    if path.endswith((".py", ".go", ".rs", ".ts", ".tsx", ".sql", ".sh", ".ps1")):
        categories.append("code")
    if not categories:
        categories.append("configuration")
    return tuple(categories)


def _validate_component_links(
    root: Path,
    component: Component,
    *,
    coverage: set[str],
    index_text: str,
) -> tuple[set[str], set[str], set[str]]:
    if "\n" in component.primary_responsibility:
        raise ManifestError(f"{component.id}: primary responsibility must be one line")
    docs = _expand_patterns(
        root,
        component.documentation,
        label=f"{component.id}.documentation",
    )
    architecture = _expand_patterns(
        root,
        component.architecture,
        label=f"{component.id}.architecture",
    )
    tests = _expand_patterns(
        root,
        component.tests,
        label=f"{component.id}.tests",
        require_each=bool(component.tests),
    )
    linked = docs | architecture | tests
    outside = linked - coverage
    if outside:
        raise ManifestError(
            f"{component.id}: linked files are outside coverage: {', '.join(sorted(outside))}"
        )
    for path in sorted(docs | architecture):
        if path.startswith("wiki/content/"):
            _validate_wiki_document(root, path, index_text)
    return docs, architecture, tests


def load_inventory(
    manifest_path: str | Path = "project.manifest.toml",
) -> ManifestInventory:
    """Load, resolve, and fully validate the project architecture inventory."""

    path = Path(manifest_path).resolve(strict=True)
    root = path.parent
    try:
        raw_bytes = path.read_bytes()
        raw = tomllib.loads(raw_bytes.decode("utf-8"))
        manifest = ProjectManifest.model_validate(raw)
    except (UnicodeDecodeError, tomllib.TOMLDecodeError, ValidationError) as exc:
        raise ManifestError(f"invalid project manifest: {exc}") from exc

    tracked = _git_tracked_paths(root)
    coverage = _expand_patterns(
        root,
        manifest.coverage.include,
        label="coverage.include",
    )
    coverage.update(
        _tracked_coverage(
            tracked,
            include=manifest.coverage.tracked_include,
            exclude=manifest.coverage.tracked_exclude,
        )
    )
    lowered: dict[str, str] = {}
    for item in sorted(coverage):
        collision = lowered.setdefault(item.casefold(), item)
        if collision != item:
            raise ManifestError(f"case-colliding repository paths: {collision}, {item}")

    owned_by: dict[str, str] = {}
    component_paths: dict[str, set[str]] = {}
    for component in manifest.components:
        owned = _expand_patterns(
            root,
            component.owned_paths,
            label=f"{component.id}.owned_paths",
        )
        outside = owned - coverage
        if outside:
            raise ManifestError(
                f"{component.id}: owned paths are outside coverage: {', '.join(sorted(outside))}"
            )
        component_paths[component.id] = owned
        for item in sorted(owned):
            previous = owned_by.get(item)
            if previous is not None:
                raise ManifestError(
                    f"duplicate ownership for {item}: {previous}, {component.id}"
                )
            owned_by[item] = component.id

    unowned = coverage - set(owned_by)
    if unowned:
        preview = ", ".join(sorted(unowned)[:20])
        raise ManifestError(f"unowned governed paths ({len(unowned)}): {preview}")

    index_path = root / "wiki" / "index.md"
    index_text = _read_stable_utf8(index_path, "wiki/index.md")
    component_docs: dict[str, set[str]] = {}
    component_architecture: dict[str, set[str]] = {}
    for component in manifest.components:
        docs, architecture, _ = _validate_component_links(
            root,
            component,
            coverage=coverage,
            index_text=index_text,
        )
        component_docs[component.id] = docs
        component_architecture[component.id] = architecture

    components_by_id = {component.id: component for component in manifest.components}
    configured_lock = _configured_lock_path(root, manifest.knowledge.lock_file)
    governed_by_path: dict[str, GovernedFile] = {}
    for governed_path, component_id in sorted(owned_by.items()):
        # The generated lock cannot hash itself. Its canonical byte-for-byte
        # comparison in check_lock is the evidence for this one owned path.
        if governed_path == configured_lock:
            continue
        absolute = root / governed_path
        text = _read_stable_utf8(absolute, governed_path)
        component = components_by_id[component_id]
        governed_by_path[governed_path] = GovernedFile(
            path=governed_path,
            absolute_path=absolute,
            component_id=component_id,
            owner=component.owner,
            text=text,
            sha256=sha256_text(text),
            byte_count=len(text.encode("utf-8")),
        )

    for component in manifest.components:
        for governed_path in sorted(component_paths[component.id]):
            if not governed_path.endswith(".py"):
                continue
            governed = governed_by_path.get(governed_path)
            if governed is None:
                # Only the configured generated lock is omitted from evidence.
                continue
            scan_forbidden_imports(
                governed_path,
                governed.text,
                component.forbidden_imports,
            )

    if tracked is not None:
        forbidden_patterns = manifest.coverage.forbidden_repository_paths
        for pattern in forbidden_patterns:
            validate_relative_pattern(pattern)
        forbidden = {
            item
            for item in tracked
            if any(_matches_pattern(item, pattern) for pattern in forbidden_patterns)
        }
        exceptions = set(manifest.coverage.legacy_repository_exceptions)
        unexpected = (forbidden & tracked) - exceptions
        if unexpected:
            raise ManifestError(
                "forbidden generated/runtime paths are tracked: "
                + ", ".join(sorted(unexpected))
            )

    quarantined_paths: set[str] = set()
    for quarantine in manifest.quarantines:
        actual = _expand_patterns(
            root,
            quarantine.scope,
            label=f"{quarantine.id}.scope",
            require_each=False,
        )
        expected = set(quarantine.baseline_paths)
        if actual != expected:
            added = sorted(actual - expected)
            removed = sorted(expected - actual)
            detail = []
            if added:
                detail.append("added=" + ",".join(added))
            if removed:
                detail.append("removed=" + ",".join(removed))
            raise ManifestError(f"{quarantine.id} scope drift: {'; '.join(detail)}")
        outside_coverage = actual - coverage
        if outside_coverage:
            raise ManifestError(
                f"{quarantine.id}: quarantined paths are outside ownership coverage: "
                + ", ".join(sorted(outside_coverage))
            )
        quarantined_paths.update(actual)

    sources_by_path: dict[str, SourceFile] = {}
    for component in manifest.components:
        embedding_paths = _expand_patterns(
            root,
            component.embedding_inputs,
            label=f"{component.id}.embedding_inputs",
            require_each=bool(component.embedding_inputs),
        )
        embedding_paths.difference_update(quarantined_paths)
        for source_path in sorted(embedding_paths):
            if source_path == configured_lock:
                raise ManifestError(
                    "the generated knowledge lock cannot be an embedding input"
                )
            owner = owned_by.get(source_path)
            if owner != component.id:
                raise ManifestError(
                    f"{component.id}: embedding input {source_path} is owned by {owner!r}"
                )
            absolute = root / source_path
            safe_path = validate_source_path(root, absolute)
            governed = governed_by_path[safe_path]
            scan_secrets(safe_path, governed.text)
            sources_by_path[safe_path] = SourceFile(
                path=safe_path,
                absolute_path=absolute,
                component_id=component.id,
                text=governed.text,
                sha256=governed.sha256,
                byte_count=governed.byte_count,
                categories=_categories_for(
                    safe_path,
                    documentation=component_docs[component.id],
                    architecture=component_architecture[component.id],
                ),
            )

    return ManifestInventory(
        root=root,
        manifest_path=path,
        manifest=manifest,
        manifest_sha256=sha256_text(raw_bytes.decode("utf-8")),
        owned_by=owned_by,
        quarantined_paths=frozenset(quarantined_paths),
        governed_files=tuple(
            governed_by_path[path] for path in sorted(governed_by_path)
        ),
        sources=tuple(sources_by_path[path] for path in sorted(sources_by_path)),
    )
