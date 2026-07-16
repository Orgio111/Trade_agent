"""Public, non-secret status contracts for the local control plane."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DependencyStatus(_StrictModel):
    healthy: bool
    detail: str = Field(min_length=1, max_length=512)


class RuntimeReadiness(_StrictModel):
    ready: bool
    execution_enabled: bool
    mode: str
    service: str = "quantex-control-plane"
    dependencies: Mapping[str, DependencyStatus]


class Liveness(_StrictModel):
    status: str = "ok"
    service: str = "quantex-control-plane"
