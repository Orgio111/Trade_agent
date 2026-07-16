"""Offline validation for runtime reachability and sensitive authority ownership."""

from __future__ import annotations

import ipaddress
from pathlib import Path

from .errors import ManifestError
from .models import ProjectManifest
from .security import endpoint_host, validate_relative_pattern


def _is_local_inference_host(host: str) -> bool:
    normalized = host.casefold().rstrip(".")
    if normalized in {"localhost", "host.docker.internal"}:
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _validate_dependency_graph(manifest: ProjectManifest) -> None:
    components = {component.id: component for component in manifest.components}
    for component in manifest.components:
        for dependency in component.depends_on:
            if dependency == component.id:
                raise ManifestError(
                    f"{component.id}: component cannot depend on itself"
                )
            if dependency not in components:
                raise ManifestError(
                    f"{component.id}: unknown component dependency: {dependency}"
                )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(component_id: str) -> None:
        if component_id in visiting:
            raise ManifestError(f"component dependency cycle includes: {component_id}")
        if component_id in visited:
            return
        visiting.add(component_id)
        for dependency in components[component_id].depends_on:
            visit(dependency)
        visiting.remove(component_id)
        visited.add(component_id)

    for component_id in sorted(components):
        visit(component_id)

    for component in manifest.components:
        if component.runtime_status != "active":
            continue
        reachable: set[str] = set()
        pending = list(component.depends_on)
        while pending:
            dependency = pending.pop()
            if dependency in reachable:
                continue
            reachable.add(dependency)
            pending.extend(components[dependency].depends_on)
        quarantined = sorted(
            dependency
            for dependency in reachable
            if components[dependency].runtime_status == "quarantined"
        )
        if quarantined:
            raise ManifestError(
                f"{component.id}: active component reaches quarantined component(s): "
                + ", ".join(quarantined)
            )


def _validate_authorities(manifest: ProjectManifest) -> None:
    sensitive_owner: dict[str, str] = {}
    for component in manifest.components:
        for authority in component.authorities:
            if not authority.sensitive:
                continue
            if component.runtime_status == "quarantined":
                raise ManifestError(
                    f"{component.id}: quarantined component cannot hold sensitive "
                    f"authority {authority.id}"
                )
            previous = sensitive_owner.setdefault(authority.id, component.id)
            if previous != component.id:
                raise ManifestError(
                    f"sensitive authority {authority.id} has multiple owners: "
                    f"{previous}, {component.id}"
                )


def _validate_runtime_allowlists(manifest: ProjectManifest) -> None:
    policy = manifest.runtime_policy
    configured_hosts = tuple(
        host.casefold().rstrip(".") for host in policy.local_inference_hosts
    )
    if len(configured_hosts) != len(set(configured_hosts)):
        raise ManifestError(
            "runtime policy local inference hosts must be unique after normalization"
        )
    invalid_hosts = sorted(
        host
        for host in configured_hosts
        if not host or not _is_local_inference_host(host)
    )
    if invalid_hosts:
        raise ManifestError(
            "runtime policy contains non-local inference host(s): "
            + ", ".join(invalid_hosts)
        )

    allowed_models = set(policy.allowed_models)
    allowed_modes = set(policy.allowed_modes)
    local_hosts = set(configured_hosts)
    for component in manifest.components:
        egress_hosts = {
            endpoint_host(endpoint) for endpoint in component.allowed_egress
        }
        if component.runtime_status == "quarantined":
            continue
        unknown_models = sorted(set(component.allowed_models) - allowed_models)
        if unknown_models:
            raise ManifestError(
                f"{component.id}: {len(unknown_models)} model(s) outside runtime allowlist"
            )
        unknown_modes = sorted(set(component.allowed_modes) - allowed_modes)
        if unknown_modes:
            raise ManifestError(
                f"{component.id}: {len(unknown_modes)} mode(s) outside runtime allowlist"
            )

        inference_authority = any(
            authority.id.startswith("inference.") for authority in component.authorities
        )
        if component.runtime_status != "active" or not inference_authority:
            continue
        if not component.allowed_models:
            raise ManifestError(
                f"{component.id}: active inference authority needs allowed_models"
            )
        if not component.allowed_egress:
            raise ManifestError(
                f"{component.id}: active inference authority needs local allowed_egress"
            )
        remote_hosts = sorted(host for host in egress_hosts if host not in local_hosts)
        if remote_hosts:
            raise ManifestError(
                f"{component.id}: active inference egress is not locally allowed: "
                + ", ".join(remote_hosts)
            )


def validate_runtime_authority(
    manifest: ProjectManifest,
    *,
    root: Path,
    owned_by: dict[str, str],
    quarantined_paths: set[str],
) -> None:
    """Validate the default runtime without network or service dependencies."""

    _validate_dependency_graph(manifest)
    _validate_authorities(manifest)
    _validate_runtime_allowlists(manifest)

    for component in manifest.components:
        active_quarantine = sorted(
            path
            for path, owner in owned_by.items()
            if owner == component.id
            and path in quarantined_paths
            and component.runtime_status == "active"
        )
        if active_quarantine:
            raise ManifestError(
                f"{component.id}: active component owns quarantined paths: "
                + ", ".join(active_quarantine)
            )
        for entrypoint in component.entrypoints:
            validate_relative_pattern(entrypoint.path)
            if any(character in entrypoint.path for character in "*?["):
                raise ManifestError(
                    f"{component.id}: entrypoint must be an exact path: "
                    f"{entrypoint.path}"
                )
            if component.runtime_status != "active":
                continue
            candidate = root / entrypoint.path
            if not candidate.is_file() or candidate.is_symlink():
                raise ManifestError(
                    f"{component.id}: active entrypoint does not exist: "
                    f"{entrypoint.path}"
                )
            if owned_by.get(entrypoint.path) != component.id:
                raise ManifestError(
                    f"{component.id}: active entrypoint is not component-owned: "
                    f"{entrypoint.path}"
                )
            if entrypoint.path in quarantined_paths:
                raise ManifestError(
                    f"{component.id}: active entrypoint is quarantined: "
                    f"{entrypoint.path}"
                )


def authority_summary(manifest: ProjectManifest) -> dict[str, object]:
    """Return stable, non-secret runtime policy evidence for the CLI."""

    active = sorted(
        component.id
        for component in manifest.components
        if component.runtime_status == "active"
    )
    quarantined = sorted(
        component.id
        for component in manifest.components
        if component.runtime_status == "quarantined"
    )
    sensitive = sorted(
        authority.id
        for component in manifest.components
        for authority in component.authorities
        if authority.sensitive
    )
    defaults = sorted(
        f"{component.id}:{entrypoint.path}"
        for component in manifest.components
        if component.runtime_status == "active"
        for entrypoint in component.entrypoints
        if entrypoint.default
    )
    return {
        "active_components": active,
        "quarantined_components": quarantined,
        "sensitive_authorities": sensitive,
        "default_entrypoints": defaults,
    }
