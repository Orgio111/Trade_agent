"""Artifact-bound seven-day shadow state and fail-closed gates."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from packages.alpha_certification.shadow import (
    ShadowThresholds,
    create_shadow_state,
    evaluate_shadow_gates,
    finalize_shadow_state,
    record_shadow_sample,
)


NOW = datetime(2026, 7, 30, tzinfo=UTC)


def _state() -> dict:
    return create_shadow_state(
        artifact_sha256="a" * 64,
        alpha_model_card_sha256="b" * 64,
        reliability_document_sha256="c" * 64,
        git_commit="abc123",
        code_sha256="d" * 64,
        now=NOW,
        thresholds=ShadowThresholds(
            duration_seconds=2,
            sample_seconds=1,
            sample_coverage_ratio=1,
            availability_ratio=1,
            min_candidate_events=1,
            min_fills=1,
        ),
        baseline_image_ids={"candidate-worker": "image-one"},
    )


def _sample(offset: int, **overrides: object) -> dict:
    value = {
        "captured_at": (NOW + timedelta(seconds=offset)).isoformat(),
        "ready": True,
        "execution_enabled": True,
        "paper_only": True,
        "mode": "paper_live",
        "artifact_sha256": "a" * 64,
        "candidate_provider": "alpha_shadow",
        "image_ids": {"candidate-worker": "image-one"},
        "database": {
            "candidate_events": offset,
            "fills": offset,
            "duplicate_signals": 0,
            "duplicate_fills": 0,
            "reconciliation_mismatches": 0,
            "candidate_digest_mismatches": 0,
            "max_drawdown": 0.02,
        },
    }
    value.update(overrides)
    return value


def test_shadow_passes_only_after_elapsed_covered_artifact_bound_samples() -> None:
    state = _state()
    record_shadow_sample(state, _sample(1))
    assert not finalize_shadow_state(state, now=NOW + timedelta(seconds=1))
    assert state["status"] == "running"

    record_shadow_sample(state, _sample(2))
    assert finalize_shadow_state(state, now=NOW + timedelta(seconds=2))
    assert state["status"] == "passed"
    assert state["duplicate_signals"] == 0
    assert all(gate["passed"] for gate in state["gates"].values())


def test_shadow_fails_on_digest_drift_and_integrity_mismatch() -> None:
    state = _state()
    record_shadow_sample(state, _sample(1))
    bad = _sample(
        2,
        artifact_sha256="f" * 64,
        database={
            "candidate_events": 2,
            "fills": 2,
            "duplicate_signals": 1,
            "duplicate_fills": 0,
            "reconciliation_mismatches": 1,
            "candidate_digest_mismatches": 1,
            "max_drawdown": 0.20,
        },
    )
    record_shadow_sample(state, bad)

    gates = evaluate_shadow_gates(state, now=NOW + timedelta(seconds=2))
    assert not gates["artifact_digest"]["passed"]
    assert not gates["duplicate_signals"]["passed"]
    assert not gates["reconciliation_mismatches"]["passed"]
    assert not gates["candidate_digest_mismatches"]["passed"]
    assert not gates["max_drawdown"]["passed"]
    assert not finalize_shadow_state(state, now=NOW + timedelta(seconds=2))
    assert state["status"] == "failed"
