"""Resumable, artifact-bound paper-shadow certification state."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import math
from typing import Any, Mapping


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(UTC)


def _iso(value: datetime) -> str:
    return _utc(value).isoformat()


def _parse(value: str) -> datetime:
    return _utc(datetime.fromisoformat(value))


def _sha256(value: str, *, name: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class ShadowThresholds:
    duration_seconds: int = 7 * 24 * 60 * 60
    sample_seconds: int = 60
    sample_coverage_ratio: float = 0.90
    availability_ratio: float = 0.995
    max_drawdown: float = 0.15
    min_candidate_events: int = 1
    min_fills: int = 1

    def __post_init__(self) -> None:
        if self.duration_seconds < 1 or self.sample_seconds < 1:
            raise ValueError("shadow duration and sample interval must be positive")
        for name in ("sample_coverage_ratio", "availability_ratio"):
            value = getattr(self, name)
            if not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if not 0 < self.max_drawdown < 1:
            raise ValueError("max_drawdown must be in (0, 1)")
        if self.min_candidate_events < 1 or self.min_fills < 1:
            raise ValueError("shadow activity thresholds must be positive")


def create_shadow_state(
    *,
    artifact_sha256: str,
    alpha_model_card_sha256: str,
    reliability_document_sha256: str,
    git_commit: str,
    code_sha256: str,
    now: datetime,
    thresholds: ShadowThresholds,
    baseline_image_ids: Mapping[str, str],
) -> dict[str, Any]:
    """Create bounded state tied to one candidate and one reliability report."""

    _sha256(artifact_sha256, name="artifact_sha256")
    _sha256(alpha_model_card_sha256, name="alpha_model_card_sha256")
    _sha256(reliability_document_sha256, name="reliability_document_sha256")
    _sha256(code_sha256, name="code_sha256")
    if not git_commit.strip():
        raise ValueError("git_commit cannot be blank")
    started_at = _iso(now)
    return {
        "schema_version": 1,
        "status": "running",
        "paper_only": True,
        "artifact_sha256": artifact_sha256,
        "alpha_model_card_sha256": alpha_model_card_sha256,
        "reliability_document_sha256": reliability_document_sha256,
        "git_commit": git_commit,
        "code_sha256": code_sha256,
        "started_at": started_at,
        "updated_at": started_at,
        "elapsed_seconds": 0.0,
        "thresholds": asdict(thresholds),
        "baseline_image_ids": dict(sorted(baseline_image_ids.items())),
        "summary": {
            "sample_count": 0,
            "ready_samples": 0,
            "controller_error_samples": 0,
            "paper_policy_violations": 0,
            "artifact_mismatch_samples": 0,
            "provider_mismatch_samples": 0,
            "image_drift": False,
            "duplicate_signals": 0,
            "duplicate_fills": 0,
            "reconciliation_mismatches": 0,
            "candidate_digest_mismatches": 0,
            "candidate_events": 0,
            "fills": 0,
            "max_drawdown": 0.0,
            "last_captured_at": None,
        },
        "gates": {},
    }


def record_shadow_sample(
    state: dict[str, Any],
    sample: Mapping[str, Any],
) -> None:
    """Fold a cumulative runtime observation into a bounded shadow state."""

    if state.get("status") != "running":
        raise ValueError("shadow state is no longer running")
    captured_value = sample.get("captured_at")
    if not isinstance(captured_value, str):
        raise ValueError("shadow sample requires captured_at")
    captured_at = _parse(captured_value)
    summary = state["summary"]
    previous = summary.get("last_captured_at")
    if isinstance(previous, str) and captured_at <= _parse(previous):
        raise ValueError("shadow samples must be strictly chronological")

    summary["sample_count"] += 1
    if sample.get("controller_error"):
        summary["controller_error_samples"] += 1
    elif sample.get("ready") is True:
        summary["ready_samples"] += 1
    if (
        sample.get("paper_only") is not True
        or sample.get("mode") != "paper_live"
        or sample.get("execution_enabled") is not True
    ):
        summary["paper_policy_violations"] += 1
    if sample.get("artifact_sha256") != state["artifact_sha256"]:
        summary["artifact_mismatch_samples"] += 1
    if sample.get("candidate_provider") != "alpha_shadow":
        summary["provider_mismatch_samples"] += 1

    image_ids = sample.get("image_ids")
    baseline_images = state.get("baseline_image_ids")
    if isinstance(image_ids, Mapping) and isinstance(baseline_images, Mapping):
        for service, expected in baseline_images.items():
            if image_ids.get(service) != expected:
                summary["image_drift"] = True

    database = sample.get("database")
    if isinstance(database, Mapping):
        for name in (
            "duplicate_signals",
            "duplicate_fills",
            "reconciliation_mismatches",
            "candidate_digest_mismatches",
            "candidate_events",
            "fills",
        ):
            summary[name] = max(summary[name], int(database.get(name, 0)))
        summary["max_drawdown"] = max(
            float(summary["max_drawdown"]),
            float(database.get("max_drawdown", 0.0)),
        )

    summary["last_captured_at"] = _iso(captured_at)
    state["updated_at"] = _iso(captured_at)


def _gate(passed: bool, observed: Any, required: Any) -> dict[str, Any]:
    return {"passed": bool(passed), "observed": observed, "required": required}


def evaluate_shadow_gates(
    state: Mapping[str, Any],
    *,
    now: datetime,
) -> dict[str, dict[str, Any]]:
    """Evaluate elapsed, coverage, provenance, integrity, and activity gates."""

    thresholds = ShadowThresholds(**state["thresholds"])
    elapsed = max(0.0, (_utc(now) - _parse(str(state["started_at"]))).total_seconds())
    summary = state["summary"]
    sample_count = int(summary["sample_count"])
    expected_samples = max(
        1,
        math.floor(thresholds.duration_seconds / thresholds.sample_seconds),
    )
    required_samples = max(
        1,
        math.floor(expected_samples * thresholds.sample_coverage_ratio),
    )
    healthy_denominator = max(
        1,
        sample_count - int(summary["controller_error_samples"]),
    )
    availability = int(summary["ready_samples"]) / healthy_denominator
    return {
        "elapsed": _gate(
            elapsed >= thresholds.duration_seconds,
            round(elapsed, 3),
            thresholds.duration_seconds,
        ),
        "sample_coverage": _gate(
            sample_count >= required_samples,
            sample_count,
            required_samples,
        ),
        "availability": _gate(
            availability >= thresholds.availability_ratio,
            round(availability, 6),
            thresholds.availability_ratio,
        ),
        "controller_errors": _gate(
            int(summary["controller_error_samples"]) == 0,
            int(summary["controller_error_samples"]),
            0,
        ),
        "paper_only": _gate(
            int(summary["paper_policy_violations"]) == 0,
            int(summary["paper_policy_violations"]),
            0,
        ),
        "artifact_digest": _gate(
            int(summary["artifact_mismatch_samples"]) == 0,
            int(summary["artifact_mismatch_samples"]),
            0,
        ),
        "candidate_provider": _gate(
            int(summary["provider_mismatch_samples"]) == 0,
            int(summary["provider_mismatch_samples"]),
            0,
        ),
        "immutable_images": _gate(
            summary.get("image_drift") is False,
            bool(summary.get("image_drift")),
            False,
        ),
        "duplicate_signals": _gate(
            int(summary["duplicate_signals"]) == 0,
            int(summary["duplicate_signals"]),
            0,
        ),
        "duplicate_fills": _gate(
            int(summary["duplicate_fills"]) == 0,
            int(summary["duplicate_fills"]),
            0,
        ),
        "reconciliation_mismatches": _gate(
            int(summary["reconciliation_mismatches"]) == 0,
            int(summary["reconciliation_mismatches"]),
            0,
        ),
        "candidate_digest_mismatches": _gate(
            int(summary["candidate_digest_mismatches"]) == 0,
            int(summary["candidate_digest_mismatches"]),
            0,
        ),
        "max_drawdown": _gate(
            float(summary["max_drawdown"]) <= thresholds.max_drawdown,
            round(float(summary["max_drawdown"]), 8),
            thresholds.max_drawdown,
        ),
        "candidate_activity": _gate(
            int(summary["candidate_events"]) >= thresholds.min_candidate_events,
            int(summary["candidate_events"]),
            thresholds.min_candidate_events,
        ),
        "fill_activity": _gate(
            int(summary["fills"]) >= thresholds.min_fills,
            int(summary["fills"]),
            thresholds.min_fills,
        ),
    }


def finalize_shadow_state(
    state: dict[str, Any],
    *,
    now: datetime,
) -> bool:
    """Freeze a passing/failed decision after the full elapsed period."""

    gates = evaluate_shadow_gates(state, now=now)
    elapsed = max(0.0, (_utc(now) - _parse(str(state["started_at"]))).total_seconds())
    state["elapsed_seconds"] = round(elapsed, 3)
    state["gates"] = gates
    state["updated_at"] = _iso(now)
    elapsed_complete = gates["elapsed"]["passed"] is True
    if not elapsed_complete:
        return False
    passed = all(gate.get("passed") is True for gate in gates.values())
    state["status"] = "passed" if passed else "failed"
    state["completed_at"] = _iso(now)
    summary = state["summary"]
    state["duplicate_signals"] = int(summary["duplicate_signals"])
    state["duplicate_fills"] = int(summary["duplicate_fills"])
    state["reconciliation_mismatches"] = int(summary["reconciliation_mismatches"])
    return passed
