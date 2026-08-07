"""Pure state, gate, and signing logic for 24/7 paper certification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import hmac
import json
import math
from typing import Any, Mapping


SCHEMA_VERSION = 1
CHAOS_PHASE = "chaos_24h"
PAPER_PHASE = "paper_7d"
DEFAULT_FAULTS = (
    ("nats_outage", 0.10),
    ("candidate_worker_crash", 0.40),
    ("postgres_outage", 0.70),
)


def _iso(timestamp: datetime) -> str:
    if timestamp.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return timestamp.astimezone(UTC).isoformat()


def _parse(timestamp: str) -> datetime:
    parsed = datetime.fromisoformat(timestamp)
    if parsed.tzinfo is None:
        raise ValueError("stored timestamps must be timezone-aware")
    return parsed.astimezone(UTC)


@dataclass(frozen=True)
class Thresholds:
    """Release thresholds recorded into every certification state."""

    chaos_seconds: int = 24 * 60 * 60
    paper_seconds: int = 7 * 24 * 60 * 60
    sample_seconds: int = 60
    availability_ratio: float = 0.995
    sample_coverage_ratio: float = 0.90
    max_restart_delta: int = 3
    max_recovery_seconds: int = 300
    max_rpo_seconds: int = 60 * 60
    max_rto_seconds: int = 30 * 60
    restore_evidence_max_age_seconds: int = 24 * 60 * 60

    def __post_init__(self) -> None:
        integer_fields = (
            "chaos_seconds",
            "paper_seconds",
            "sample_seconds",
            "max_restart_delta",
            "max_recovery_seconds",
            "max_rpo_seconds",
            "max_rto_seconds",
            "restore_evidence_max_age_seconds",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if value <= 0 and name != "max_restart_delta":
                raise ValueError(f"{name} must be positive")
            if name == "max_restart_delta" and value < 0:
                raise ValueError("max_restart_delta must be non-negative")
        for name in ("availability_ratio", "sample_coverage_ratio"):
            value = getattr(self, name)
            if not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")


def canonical_json(document: Mapping[str, Any]) -> bytes:
    """Return deterministic UTF-8 JSON bytes used by evidence signatures."""

    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sign_document(document: Mapping[str, Any], key: bytes) -> dict[str, Any]:
    """Attach an HMAC-SHA256 signature without exposing the local key."""

    if len(key) < 32:
        raise ValueError("signing key must contain at least 32 bytes")
    unsigned = dict(document)
    unsigned.pop("signature", None)
    digest = hmac.new(key, canonical_json(unsigned), hashlib.sha256).hexdigest()
    return {
        **unsigned,
        "signature": {
            "algorithm": "HMAC-SHA256",
            "key_id": hashlib.sha256(key).hexdigest()[:16],
            "value": digest,
        },
    }


def verify_document(document: Mapping[str, Any], key: bytes) -> bool:
    """Verify a signed evidence document with constant-time comparison."""

    signature = document.get("signature")
    if not isinstance(signature, Mapping):
        return False
    if signature.get("algorithm") != "HMAC-SHA256":
        return False
    expected_key_id = hashlib.sha256(key).hexdigest()[:16]
    if not hmac.compare_digest(str(signature.get("key_id", "")), expected_key_id):
        return False
    unsigned = dict(document)
    unsigned.pop("signature", None)
    expected = hmac.new(key, canonical_json(unsigned), hashlib.sha256).hexdigest()
    return hmac.compare_digest(str(signature.get("value", "")), expected)


def _empty_summary(baseline: Mapping[str, Any]) -> dict[str, Any]:
    baseline_ids = baseline.get("container_ids", {})
    return {
        "sample_count": 0,
        "ready_samples": 0,
        "execution_enabled_samples": 0,
        "execution_disabled_samples": 0,
        "unexpected_unready_samples": 0,
        "controller_error_samples": 0,
        "max_duplicate_inbox": 0,
        "max_duplicate_fills": 0,
        "max_reconciliation_discrepancies": 0,
        "max_failed_reconciliations_delta": 0,
        "max_dead_letters_delta": 0,
        "max_restart_delta": 0,
        "image_drift": False,
        "seen_container_ids": {
            str(service): [str(container_id)]
            for service, container_id in baseline_ids.items()
        }
        if isinstance(baseline_ids, Mapping)
        else {},
        "last_pending_outbox": int(baseline.get("pending_outbox", 0)),
        "last_sample": None,
        "baseline": dict(baseline),
    }


def create_state(
    *,
    run_id: str,
    now: datetime,
    thresholds: Thresholds,
    baseline: Mapping[str, Any],
    git_commit: str,
    source_lock_sha256: str,
    project: str,
) -> dict[str, Any]:
    """Create a resumable state with a deterministic 24-hour fault schedule."""

    started = _iso(now)
    faults = [
        {
            "name": name,
            "scheduled_offset_seconds": int(thresholds.chaos_seconds * fraction),
            "status": "pending",
        }
        for name, fraction in DEFAULT_FAULTS
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "project": project,
        "git_commit": git_commit,
        "source_lock_sha256": source_lock_sha256,
        "paper_only": True,
        "status": "running",
        "phase": CHAOS_PHASE,
        "started_at": started,
        "phase_started_at": started,
        "updated_at": started,
        "thresholds": asdict(thresholds),
        "faults": faults,
        "phase_results": [],
        "summary": _empty_summary(baseline),
        "external_evidence": {
            "alert_delivery": None,
            "restore": None,
        },
    }


def record_sample(state: dict[str, Any], sample: Mapping[str, Any]) -> None:
    """Fold one runtime/DB/container sample into bounded aggregate state."""

    summary = state["summary"]
    summary["sample_count"] += 1
    if sample.get("controller_error"):
        summary["controller_error_samples"] += 1
    elif sample.get("ready") is True:
        summary["ready_samples"] += 1
    elif not sample.get("expected_fault", False):
        summary["unexpected_unready_samples"] += 1
    if not sample.get("expected_fault", False) and not sample.get("controller_error"):
        if sample.get("execution_enabled") is True:
            summary["execution_enabled_samples"] += 1
        else:
            summary["execution_disabled_samples"] += 1

    database = sample.get("database")
    if isinstance(database, Mapping):
        summary["max_duplicate_inbox"] = max(
            summary["max_duplicate_inbox"],
            int(database.get("duplicate_inbox", 0)),
        )
        summary["max_duplicate_fills"] = max(
            summary["max_duplicate_fills"],
            int(database.get("duplicate_fills", 0)),
        )
        summary["max_reconciliation_discrepancies"] = max(
            summary["max_reconciliation_discrepancies"],
            int(database.get("reconciliation_discrepancies", 0)),
        )
        baseline_failures = int(summary["baseline"].get("failed_reconciliations", 0))
        summary["max_failed_reconciliations_delta"] = max(
            int(summary.get("max_failed_reconciliations_delta", 0)),
            max(
                0,
                int(database.get("failed_reconciliations", 0)) - baseline_failures,
            ),
        )
        baseline_dead_letters = int(summary["baseline"].get("dead_letters", 0))
        summary["max_dead_letters_delta"] = max(
            summary["max_dead_letters_delta"],
            max(0, int(database.get("dead_letters", 0)) - baseline_dead_letters),
        )
        summary["last_pending_outbox"] = int(database.get("pending_outbox", 0))

    restarts = sample.get("restart_counts")
    container_ids = sample.get("container_ids")
    image_ids = sample.get("image_ids")
    baseline_restarts = summary["baseline"].get("restart_counts", {})
    baseline_images = summary["baseline"].get("image_ids", {})
    if isinstance(image_ids, Mapping) and isinstance(baseline_images, Mapping):
        for service, image_id in image_ids.items():
            if service in baseline_images and image_id != baseline_images[service]:
                summary["image_drift"] = True
    if isinstance(restarts, Mapping) and isinstance(baseline_restarts, Mapping):
        for service, value in restarts.items():
            delta = max(0, int(value) - int(baseline_restarts.get(service, 0)))
            if isinstance(container_ids, Mapping) and service in container_ids:
                seen = summary["seen_container_ids"].setdefault(str(service), [])
                container_id = str(container_ids[service])
                if container_id not in seen:
                    seen.append(container_id)
                delta += max(0, len(seen) - 1)
            summary["max_restart_delta"] = max(summary["max_restart_delta"], delta)

    summary["last_sample"] = dict(sample)
    state["updated_at"] = str(sample.get("captured_at", state["updated_at"]))


def record_fault(
    state: dict[str, Any],
    *,
    name: str,
    started_at: datetime,
    recovered_at: datetime | None,
    passed: bool,
    detail: str,
) -> None:
    """Record exactly one scheduled fault result."""

    matching = [fault for fault in state["faults"] if fault["name"] == name]
    if len(matching) != 1:
        raise ValueError(f"unknown or duplicate fault name: {name}")
    fault = matching[0]
    if fault["status"] != "pending":
        raise ValueError(f"fault was already attempted: {name}")
    recovery_seconds = None
    if recovered_at is not None:
        recovery_seconds = max(0.0, (recovered_at - started_at).total_seconds())
    fault.update(
        {
            "status": "passed" if passed else "failed",
            "started_at": _iso(started_at),
            "recovered_at": _iso(recovered_at) if recovered_at else None,
            "recovery_seconds": recovery_seconds,
            "detail": detail,
        }
    )
    state["updated_at"] = _iso(recovered_at or started_at)


def _gate(passed: bool, observed: Any, required: Any) -> dict[str, Any]:
    return {"passed": bool(passed), "observed": observed, "required": required}


def evaluate_core_gates(
    state: Mapping[str, Any],
    *,
    now: datetime,
) -> dict[str, dict[str, Any]]:
    """Evaluate elapsed, availability, invariants, restart, and chaos gates."""

    thresholds = Thresholds(**state["thresholds"])
    phase = str(state["phase"])
    required_seconds = (
        thresholds.chaos_seconds if phase == CHAOS_PHASE else thresholds.paper_seconds
    )
    elapsed = max(
        0.0, (now.astimezone(UTC) - _parse(state["phase_started_at"])).total_seconds()
    )
    summary = state["summary"]
    sample_count = int(summary["sample_count"])
    expected_samples = max(1, math.floor(required_seconds / thresholds.sample_seconds))
    required_samples = max(
        1,
        math.floor(expected_samples * thresholds.sample_coverage_ratio),
    )
    healthy_denominator = max(
        1,
        sample_count - int(summary["controller_error_samples"]),
    )
    availability = int(summary["ready_samples"]) / healthy_denominator
    execution_availability = (
        int(summary["execution_enabled_samples"]) / healthy_denominator
    )
    gates = {
        "elapsed": _gate(
            elapsed >= required_seconds, round(elapsed, 3), required_seconds
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
        "paper_execution_enabled": _gate(
            execution_availability >= thresholds.availability_ratio,
            {
                "ratio": round(execution_availability, 6),
                "disabled_samples": int(summary["execution_disabled_samples"]),
            },
            thresholds.availability_ratio,
        ),
        "duplicate_inbox": _gate(
            int(summary["max_duplicate_inbox"]) == 0,
            int(summary["max_duplicate_inbox"]),
            0,
        ),
        "duplicate_fills": _gate(
            int(summary["max_duplicate_fills"]) == 0,
            int(summary["max_duplicate_fills"]),
            0,
        ),
        "reconciliation_mismatches": _gate(
            int(summary["max_reconciliation_discrepancies"]) == 0,
            int(summary["max_reconciliation_discrepancies"]),
            0,
        ),
        "failed_reconciliations": _gate(
            int(summary.get("max_failed_reconciliations_delta", 0)) == 0,
            int(summary.get("max_failed_reconciliations_delta", 0)),
            0,
        ),
        "dead_letters": _gate(
            int(summary["max_dead_letters_delta"]) == 0,
            int(summary["max_dead_letters_delta"]),
            0,
        ),
        "pending_outbox": _gate(
            int(summary["last_pending_outbox"]) == 0,
            int(summary["last_pending_outbox"]),
            0,
        ),
        "bounded_restarts": _gate(
            int(summary["max_restart_delta"]) <= thresholds.max_restart_delta,
            int(summary["max_restart_delta"]),
            thresholds.max_restart_delta,
        ),
        "immutable_images": _gate(
            summary.get("image_drift") is False,
            bool(summary.get("image_drift")),
            False,
        ),
    }
    if phase == CHAOS_PHASE:
        completed = [fault for fault in state["faults"] if fault["status"] == "passed"]
        failed = [fault for fault in state["faults"] if fault["status"] == "failed"]
        recoveries = [
            float(fault["recovery_seconds"])
            for fault in completed
            if fault.get("recovery_seconds") is not None
        ]
        maximum_recovery = max(recoveries, default=0.0)
        gates["fault_matrix"] = _gate(
            len(completed) == len(DEFAULT_FAULTS) and not failed,
            {
                "passed": [fault["name"] for fault in completed],
                "failed": [fault["name"] for fault in failed],
            },
            [name for name, _ in DEFAULT_FAULTS],
        )
        gates["fault_recovery"] = _gate(
            bool(recoveries)
            and maximum_recovery <= thresholds.max_recovery_seconds
            and len(recoveries) == len(DEFAULT_FAULTS),
            round(maximum_recovery, 3),
            thresholds.max_recovery_seconds,
        )
    return gates


def evaluate_external_gates(
    *,
    alert_delivery: Mapping[str, Any] | None,
    restore: Mapping[str, Any] | None,
    thresholds: Thresholds,
    run_started_at: str,
    now: datetime,
) -> dict[str, dict[str, Any]]:
    """Evaluate acknowledged alert delivery and fresh off-host restore evidence."""

    alert_passed = bool(
        alert_delivery
        and alert_delivery.get("acknowledged") is True
        and alert_delivery.get("status_code") in range(200, 300)
    )
    restore_completed: datetime | None = None
    if restore and isinstance(restore.get("completed_at"), str):
        try:
            restore_completed = _parse(str(restore["completed_at"]))
        except ValueError:
            restore_completed = None
    restore_age = (
        max(0.0, (now.astimezone(UTC) - restore_completed).total_seconds())
        if restore_completed
        else None
    )
    not_from_future = bool(
        restore_completed and restore_completed <= now.astimezone(UTC)
    )
    after_run_start = bool(
        restore_completed and restore_completed >= _parse(run_started_at)
    )
    rpo = restore.get("rpo_seconds") if restore else None
    rto = restore.get("rto_seconds") if restore else None
    restore_passed = bool(
        restore
        and restore.get("passed") is True
        and restore.get("off_host_copy_verified") is True
        and isinstance(rpo, (int, float))
        and isinstance(rto, (int, float))
        and rpo <= thresholds.max_rpo_seconds
        and rto <= thresholds.max_rto_seconds
        and restore_age is not None
        and restore_age <= thresholds.restore_evidence_max_age_seconds
        and after_run_start
        and not_from_future
    )
    return {
        "alert_delivery": _gate(alert_passed, alert_delivery, "acknowledged HTTP 2xx"),
        "restore_rpo_rto": _gate(
            restore_passed,
            {
                "passed": restore.get("passed") if restore else None,
                "off_host_copy_verified": (
                    restore.get("off_host_copy_verified") if restore else None
                ),
                "rpo_seconds": rpo,
                "rto_seconds": rto,
                "evidence_age_seconds": restore_age,
                "after_run_start": after_run_start,
                "not_from_future": not_from_future,
            },
            {
                "rpo_seconds_max": thresholds.max_rpo_seconds,
                "rto_seconds_max": thresholds.max_rto_seconds,
                "evidence_age_seconds_max": thresholds.restore_evidence_max_age_seconds,
                "off_host_copy_verified": True,
            },
        ),
    }


def _all_pass(gates: Mapping[str, Mapping[str, Any]]) -> bool:
    return all(gate.get("passed") is True for gate in gates.values())


def advance_state(
    state: dict[str, Any],
    *,
    now: datetime,
    baseline: Mapping[str, Any],
    external_gates: Mapping[str, Mapping[str, Any]] | None = None,
) -> bool:
    """Advance chaos→paper→promoted only after the relevant gates pass."""

    core = evaluate_core_gates(state, now=now)
    if not _all_pass(core):
        return False
    completed_phase = next(
        (
            result
            for result in state["phase_results"]
            if result.get("phase") == state["phase"]
        ),
        None,
    )
    if completed_phase is None:
        state["phase_results"].append(
            {
                "phase": state["phase"],
                "completed_at": _iso(now),
                "gates": core,
            }
        )
    if state["phase"] == CHAOS_PHASE:
        state["phase"] = PAPER_PHASE
        state["phase_started_at"] = _iso(now)
        state["summary"] = _empty_summary(baseline)
        state["updated_at"] = _iso(now)
        return True
    if external_gates is None or not _all_pass(external_gates):
        state["status"] = "awaiting_external_evidence"
        state["external_gates"] = dict(external_gates or {})
        state["updated_at"] = _iso(now)
        return False
    state["external_gates"] = dict(external_gates)
    state["status"] = "promoted"
    state["completed_at"] = _iso(now)
    state["updated_at"] = _iso(now)
    return True
