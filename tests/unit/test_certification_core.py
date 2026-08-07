"""Pure release-gate tests for long-duration paper certification."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime, timedelta

from packages.certification.core import (
    CHAOS_PHASE,
    PAPER_PHASE,
    Thresholds,
    advance_state,
    create_state,
    evaluate_core_gates,
    evaluate_external_gates,
    record_fault,
    record_sample,
    sign_document,
    verify_document,
)


NOW = datetime(2026, 7, 28, 0, 0, tzinfo=UTC)


def _thresholds() -> Thresholds:
    return Thresholds(
        chaos_seconds=30,
        paper_seconds=60,
        sample_seconds=10,
        availability_ratio=0.90,
        sample_coverage_ratio=0.60,
        max_restart_delta=3,
        max_recovery_seconds=10,
        max_rpo_seconds=60,
        max_rto_seconds=120,
        restore_evidence_max_age_seconds=300,
    )


def _baseline() -> dict[str, object]:
    return {
        "dead_letters": 2,
        "failed_reconciliations": 5,
        "pending_outbox": 0,
        "restart_counts": {"nats": 4, "candidate-worker": 1},
    }


def _healthy_sample(at: datetime) -> dict[str, object]:
    return {
        "captured_at": at.isoformat(),
        "ready": True,
        "execution_enabled": True,
        "database": {
            "duplicate_inbox": 0,
            "duplicate_fills": 0,
            "reconciliation_discrepancies": 0,
            "failed_reconciliations": 5,
            "dead_letters": 2,
            "pending_outbox": 0,
        },
        "restart_counts": {"nats": 4, "candidate-worker": 2},
    }


def _state() -> dict[str, object]:
    return create_state(
        run_id="cert-20260728",
        now=NOW,
        thresholds=_thresholds(),
        baseline=_baseline(),
        git_commit="abc123",
        source_lock_sha256="f" * 64,
        project="trade_agent_canonical",
    )


def test_signed_status_detects_tampering() -> None:
    key = b"k" * 32
    signed = sign_document({"status": "running", "samples": 4}, key)

    assert verify_document(signed, key)
    tampered = deepcopy(signed)
    tampered["samples"] = 5
    assert not verify_document(tampered, key)
    assert not verify_document(signed, b"x" * 32)


def test_zero_duplicate_mismatch_and_restart_bounds_are_hard_gates() -> None:
    state = _state()
    record_sample(state, _healthy_sample(NOW + timedelta(seconds=10)))
    bad = _healthy_sample(NOW + timedelta(seconds=20))
    bad["database"]["duplicate_fills"] = 1
    bad["database"]["reconciliation_discrepancies"] = 1
    bad["database"]["failed_reconciliations"] = 6
    bad["restart_counts"]["candidate-worker"] = 9
    record_sample(state, bad)

    gates = evaluate_core_gates(state, now=NOW + timedelta(seconds=31))

    assert gates["duplicate_fills"]["passed"] is False
    assert gates["reconciliation_mismatches"]["passed"] is False
    assert gates["failed_reconciliations"]["passed"] is False
    assert gates["bounded_restarts"]["passed"] is False


def test_chaos_phase_advances_only_after_all_faults_and_elapsed_coverage() -> None:
    state = _state()
    for offset in (10, 20):
        record_sample(state, _healthy_sample(NOW + timedelta(seconds=offset)))
    for index, name in enumerate(
        ("nats_outage", "candidate_worker_crash", "postgres_outage")
    ):
        started = NOW + timedelta(seconds=index + 1)
        record_fault(
            state,
            name=name,
            started_at=started,
            recovered_at=started + timedelta(seconds=2),
            passed=True,
            detail="recovered",
        )

    changed = advance_state(
        state,
        now=NOW + timedelta(seconds=31),
        baseline=_baseline(),
    )

    assert changed is True
    assert state["phase"] == PAPER_PHASE
    assert state["status"] == "running"
    assert state["phase_results"][0]["phase"] == CHAOS_PHASE


def test_external_evidence_requires_alert_ack_fresh_off_host_rpo_and_rto() -> None:
    thresholds = _thresholds()
    alert = {"acknowledged": True, "status_code": 204}
    restore = {
        "passed": True,
        "off_host_copy_verified": True,
        "rpo_seconds": 30,
        "rto_seconds": 90,
        "completed_at": (NOW + timedelta(seconds=90)).isoformat(),
    }

    gates = evaluate_external_gates(
        alert_delivery=alert,
        restore=restore,
        thresholds=thresholds,
        run_started_at=NOW.isoformat(),
        now=NOW + timedelta(seconds=100),
    )

    assert all(item["passed"] for item in gates.values())
    restore["off_host_copy_verified"] = False
    blocked = evaluate_external_gates(
        alert_delivery=alert,
        restore=restore,
        thresholds=thresholds,
        run_started_at=NOW.isoformat(),
        now=NOW + timedelta(seconds=100),
    )
    assert blocked["restore_rpo_rto"]["passed"] is False

    restore["off_host_copy_verified"] = True
    restore["completed_at"] = (NOW + timedelta(seconds=101)).isoformat()
    future = evaluate_external_gates(
        alert_delivery=alert,
        restore=restore,
        thresholds=thresholds,
        run_started_at=NOW.isoformat(),
        now=NOW + timedelta(seconds=100),
    )
    assert future["restore_rpo_rto"]["passed"] is False


def test_paper_phase_waits_for_external_evidence_before_promotion() -> None:
    state = _state()
    state["phase"] = PAPER_PHASE
    state["phase_started_at"] = NOW.isoformat()
    state["summary"] = {
        **state["summary"],
        "sample_count": 4,
        "ready_samples": 4,
        "execution_enabled_samples": 4,
        "last_pending_outbox": 0,
    }
    changed = advance_state(
        state,
        now=NOW + timedelta(seconds=61),
        baseline=_baseline(),
        external_gates={
            "alert_delivery": {"passed": True},
            "restore_rpo_rto": {"passed": False},
        },
    )

    assert changed is False
    assert state["status"] == "awaiting_external_evidence"
