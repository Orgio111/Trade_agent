"""Temporary training, strict gates, and dual-signed promotion."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
import hashlib
import json

import pytest

from packages.alpha_certification.data import (
    DatasetSnapshot,
    HistoricalCandle,
    HistoricalRequest,
)
from packages.alpha_certification.promotion import (
    AlphaGateThresholds,
    PromotionDecision,
    PromotionRegistry,
    evaluate_alpha_gates,
)
from packages.alpha_certification.replay import ReplayMetrics
from packages.alpha_certification.shadow import (
    ShadowThresholds,
    create_shadow_state,
    finalize_shadow_state,
    record_shadow_sample,
)
from packages.alpha_certification.training import (
    CandidateArtifact,
    CandidateTrainingConfig,
    SklearnCandidateTrainer,
    TemporaryCandidateDirectory,
    TrainingDataset,
    load_candidate_artifact,
)
from packages.alpha_certification.walk_forward import WalkForwardConfig
from packages.certification.core import canonical_json, sign_document


def _metrics(
    *,
    net_return: float = 0.12,
    buy_hold_return: float = 0.02,
    max_drawdown: float = 0.08,
    profit_factor: float | None = 1.4,
) -> ReplayMetrics:
    return ReplayMetrics(
        bars=1_000,
        candidates=100,
        approved_decisions=50,
        rejected_decisions=50,
        stale_pending_orders=0,
        trades=40,
        net_return=net_return,
        buy_hold_return=buy_hold_return,
        max_drawdown=max_drawdown,
        profit_factor=profit_factor,
        annualized_sharpe=1.2,
        win_rate=0.55,
        expectancy=1.0,
        duplicate_signals=0,
        duplicate_fills=0,
        reconciliation_mismatches=0,
        next_bar_fill_violations=0,
    )


def _training_candles(count: int = 240) -> tuple[HistoricalCandle, ...]:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    price = Decimal("100")
    candles = []
    for index in range(count):
        move = Decimal("0.6") if index % 8 < 4 else Decimal("-0.6")
        close = price + move
        candles.append(
            HistoricalCandle(
                open_time=start + timedelta(hours=index),
                close_time=start + timedelta(hours=index + 1),
                open=price,
                high=max(price, close) + Decimal("0.2"),
                low=min(price, close) - Decimal("0.2"),
                close=close,
                volume=Decimal(100 + index),
                quote_volume=close * Decimal(100 + index),
            )
        )
        price = close
    return tuple(candles)


def _snapshot(tmp_path) -> DatasetSnapshot:
    candle = _training_candles(1)[0]
    data_path = tmp_path / "data.jsonl"
    data_path.write_text("{}\n")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}\n")
    return DatasetSnapshot(
        manifest_path=manifest_path,
        data_path=data_path,
        request=HistoricalRequest(
            symbol="BTCUSDT",
            interval="1h",
            start=candle.open_time,
            end=candle.close_time,
        ),
        candles=(candle,),
        gaps=(),
        data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )


def test_training_artifact_exists_only_inside_temporary_candidate_directory(
    tmp_path,
) -> None:
    parent = tmp_path / "candidates"
    with TemporaryCandidateDirectory(parent) as directory:
        artifact = SklearnCandidateTrainer(
            CandidateTrainingConfig(
                horizon_bars=3,
                buy_return_threshold=0.001,
                sell_return_threshold=-0.001,
                min_training_samples=100,
                n_estimators=20,
            )
        ).train(
            [
                TrainingDataset(
                    symbol="BTCUSDT",
                    interval="1h",
                    candles=_training_candles(),
                    dataset_sha256="a" * 64,
                )
            ],
            directory,
        )
        artifact_path = artifact.path
        assert artifact_path.is_file()
        assert "models" not in artifact_path.parts
        assert artifact.sha256 == hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        loaded = load_candidate_artifact(
            artifact_path,
            expected_sha256=artifact.sha256,
        )
        assert loaded.sha256 == artifact.sha256
        assert loaded.config == artifact.config
    assert not artifact_path.exists()
    assert list(parent.iterdir()) == []


def test_alpha_gates_fail_closed_on_one_unstable_fold() -> None:
    thresholds = AlphaGateThresholds(min_folds=3, min_trades_per_fold=30)
    passed = evaluate_alpha_gates([_metrics(), _metrics(), _metrics()], thresholds)
    assert passed.passed

    failed = evaluate_alpha_gates(
        [
            _metrics(),
            _metrics(),
            _metrics(net_return=-0.3, max_drawdown=0.4, profit_factor=0.7),
        ],
        thresholds,
    )
    assert not failed.passed
    assert not failed.gates["profit_factor"]["passed"]
    assert not failed.gates["max_drawdown"]["passed"]
    assert not failed.gates["fold_stability"]["passed"]


def test_registry_requires_valid_promoted_reliability_evidence(tmp_path) -> None:
    candidate = tmp_path / "candidate.joblib"
    candidate.write_bytes(b"candidate-artifact")
    digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    training_config = CandidateTrainingConfig(min_training_samples=100)
    artifact = CandidateArtifact(
        path=candidate,
        sha256=digest,
        trainer_id="test-trainer",
        feature_schema="test-schema",
        training_samples=100,
        symbols=("BTCUSDT",),
        dataset_sha256s=("a" * 64,),
        config=training_config,
    )
    alpha_key = b"a" * 32
    reliability_key = b"r" * 32
    registry = PromotionRegistry(tmp_path / "registry", alpha_key)
    decision = PromotionDecision(
        passed=True,
        gates={"test": {"passed": True, "observed": True, "required": True}},
    )
    common = {
        "artifact": artifact,
        "datasets": [_snapshot(tmp_path)],
        "walk_forward": WalkForwardConfig(
            train_window=timedelta(days=30),
            test_window=timedelta(days=7),
            purge=timedelta(hours=12),
            embargo=timedelta(hours=12),
            min_train_samples=100,
            min_test_samples=20,
        ),
        "fold_metrics": [_metrics(), _metrics(), _metrics()],
        "fold_evidence": [{"fold_id": "one"}],
        "alpha_decision": decision,
        "thresholds": AlphaGateThresholds(min_shadow_seconds=1),
        "git_commit": "abc123",
        "code_sha256": "c" * 64,
        "source_lock_sha256": "d" * 64,
        "now": datetime(2026, 7, 30, tzinfo=UTC),
    }
    pending, pending_card, pending_stage, pending_model = registry.record(**common)
    assert pending["status"] == "alpha_certified_awaiting_reliability"
    assert pending_stage is not None and pending_stage.is_file()
    assert pending_model is None
    assert registry.verify_card(json.loads(pending_card.read_text()), alpha_key)

    reliability = sign_document(
        {
            "certification_state": {
                "status": "promoted",
                "paper_only": True,
                "git_commit": "abc123",
                "source_lock_sha256": "d" * 64,
            }
        },
        reliability_key,
    )
    reliability_sha256 = hashlib.sha256(canonical_json(reliability)).hexdigest()
    pending_sha256 = hashlib.sha256(canonical_json(pending)).hexdigest()
    shadow_state = create_shadow_state(
        artifact_sha256=digest,
        alpha_model_card_sha256=pending_sha256,
        reliability_document_sha256=reliability_sha256,
        git_commit="abc123",
        code_sha256="c" * 64,
        now=datetime(2026, 7, 30, tzinfo=UTC),
        thresholds=ShadowThresholds(
            duration_seconds=1,
            sample_seconds=1,
            sample_coverage_ratio=1,
            availability_ratio=1,
            min_candidate_events=1,
            min_fills=1,
        ),
        baseline_image_ids={"candidate-worker": "image-one"},
    )
    record_shadow_sample(
        shadow_state,
        {
            "captured_at": datetime(2026, 7, 30, 0, 0, 1, tzinfo=UTC).isoformat(),
            "ready": True,
            "execution_enabled": True,
            "paper_only": True,
            "mode": "paper_live",
            "artifact_sha256": digest,
            "candidate_provider": "alpha_shadow",
            "image_ids": {"candidate-worker": "image-one"},
            "database": {
                "candidate_events": 1,
                "fills": 1,
                "duplicate_signals": 0,
                "duplicate_fills": 0,
                "reconciliation_mismatches": 0,
                "candidate_digest_mismatches": 0,
                "max_drawdown": 0.01,
            },
        },
    )
    assert finalize_shadow_state(
        shadow_state,
        now=datetime(2026, 7, 30, 0, 0, 1, tzinfo=UTC),
    )
    shadow = sign_document(
        {
            "schema_version": 1,
            "document_type": "alpha-shadow-paper",
            "shadow_state": shadow_state,
        },
        reliability_key,
    )
    promoted, promoted_card, promoted_model = registry.promote_staged(
        alpha_model_card=pending,
        reliability_document=reliability,
        reliability_key=reliability_key,
        shadow_document=shadow,
        shadow_key=reliability_key,
        now=datetime(2026, 7, 30, 0, 0, 2, tzinfo=UTC),
    )
    assert promoted["status"] == "promoted"
    assert (
        promoted_model is not None
        and promoted_model.read_bytes() == candidate.read_bytes()
    )
    assert registry.verify_card(json.loads(promoted_card.read_text()), alpha_key)

    mismatched_source_reliability = sign_document(
        {
            "certification_state": {
                "status": "promoted",
                "paper_only": True,
                "git_commit": "abc123",
                "source_lock_sha256": "e" * 64,
            }
        },
        reliability_key,
    )
    with pytest.raises(ValueError, match="reliability"):
        registry.promote_staged(
            alpha_model_card=pending,
            reliability_document=mismatched_source_reliability,
            reliability_key=reliability_key,
            shadow_document=shadow,
            shadow_key=reliability_key,
        )

    invalid_reliability = dict(reliability)
    invalid_reliability["certification_state"] = {
        "status": "failed",
        "paper_only": True,
    }
    with pytest.raises(ValueError, match="reliability"):
        registry.promote_staged(
            alpha_model_card=pending,
            reliability_document=invalid_reliability,
            reliability_key=reliability_key,
            shadow_document=shadow,
            shadow_key=reliability_key,
        )
