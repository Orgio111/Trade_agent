"""End-to-end alpha certification stays offline and fails closed."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
import hashlib
import json

import pytest

from packages.alpha_certification.controller import (
    AlphaCertificationConfig,
    AlphaCertificationController,
)
from packages.alpha_certification.data import (
    DatasetSnapshot,
    HistoricalCandle,
    HistoricalRequest,
)
from packages.alpha_certification.promotion import (
    AlphaGateThresholds,
    PromotionRegistry,
)
from packages.alpha_certification.training import CandidateTrainingConfig
from packages.alpha_certification.walk_forward import WalkForwardConfig
from packages.risk import InstrumentConstraints, RiskPolicy


def _snapshot(tmp_path, count: int = 900) -> DatasetSnapshot:
    start = datetime(2022, 1, 1, tzinfo=UTC)
    price = Decimal("100")
    candles: list[HistoricalCandle] = []
    for index in range(count):
        phase = index % 48
        move = (
            Decimal("0.4")
            if phase < 16
            else Decimal("-0.45")
            if phase < 32
            else Decimal("0.05")
        )
        close = price + move
        candles.append(
            HistoricalCandle(
                open_time=start + timedelta(hours=index),
                close_time=start + timedelta(hours=index + 1),
                open=price,
                high=max(price, close) + Decimal("0.3"),
                low=min(price, close) - Decimal("0.3"),
                close=close,
                volume=Decimal(100 + phase),
                quote_volume=close * Decimal(100 + phase),
                trade_count=10,
            )
        )
        price = close
    data_path = tmp_path / "synthetic.jsonl"
    data_path.write_text("synthetic fixture\n")
    manifest_path = tmp_path / "synthetic.manifest.json"
    manifest_path.write_text("{}\n")
    return DatasetSnapshot(
        manifest_path=manifest_path,
        data_path=data_path,
        request=HistoricalRequest(
            symbol="BTCUSDT",
            interval="1h",
            start=start,
            end=start + timedelta(hours=count),
        ),
        candles=tuple(candles),
        gaps=(),
        data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )


@pytest.mark.asyncio
async def test_pipeline_trains_only_in_temp_and_writes_signed_rejection(
    tmp_path,
) -> None:
    candidate_workspace = tmp_path / "candidates"
    registry = PromotionRegistry(tmp_path / "registry", b"a" * 32)
    controller = AlphaCertificationController(
        config=AlphaCertificationConfig(
            walk_forward=WalkForwardConfig(
                train_window=timedelta(hours=300),
                test_window=timedelta(hours=120),
                purge=timedelta(hours=3),
                embargo=timedelta(hours=12),
                min_train_samples=250,
                min_test_samples=100,
            ),
            training=CandidateTrainingConfig(
                horizon_bars=3,
                buy_return_threshold=0.002,
                sell_return_threshold=-0.002,
                min_training_samples=100,
                n_estimators=10,
            ),
            gates=AlphaGateThresholds(
                min_folds=99,
                min_trades_per_fold=1,
                min_profit_factor=100,
                max_drawdown=0.01,
            ),
            candidate_workspace=candidate_workspace,
        ),
        constraints_by_symbol={
            "BTCUSDT": InstrumentConstraints(
                venue="binance",
                market_type="spot",
                instrument="BTCUSDT",
                tick_size=Decimal("0.01"),
                step_size=Decimal("0.00001"),
                min_quantity=Decimal("0.00001"),
                min_notional=Decimal("10"),
            )
        },
        risk_policy=RiskPolicy(
            version="alpha-integration-test",
            min_confidence=Decimal("0.55"),
        ),
        registry=registry,
    )

    result = await controller.certify(
        [_snapshot(tmp_path)],
        git_commit="abc123",
        code_sha256="c" * 64,
        source_lock_sha256="d" * 64,
    )

    assert result.status == "rejected"
    assert result.staged_model_path is None
    assert result.promoted_model_path is None
    assert len(result.fold_metrics) >= 3
    document = json.loads(result.report_path.read_text())
    assert registry.verify_card(document, b"a" * 32)
    assert document["alpha_gates"]["passed"] is False
    assert "cost_stress" in document["alpha_gates"]["gates"]
    assert len(document["fold_evidence"]) == len(result.fold_metrics)
    assert candidate_workspace.is_dir()
    assert list(candidate_workspace.iterdir()) == []
    assert not (tmp_path / "registry" / "models").exists()
