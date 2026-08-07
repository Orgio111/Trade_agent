"""End-to-end offline alpha certification controller."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

from packages.risk import InstrumentConstraints, RiskPolicy

from .data import DatasetSnapshot, GapPolicy
from .promotion import (
    AlphaGateThresholds,
    PromotionDecision,
    PromotionRegistry,
    evaluate_alpha_gates,
)
from .replay import (
    CanonicalAlphaReplay,
    CanonicalCandidateAdapter,
    ReplayConfig,
    ReplayCostModel,
    ReplayMetrics,
)
from .training import (
    CandidateTrainingConfig,
    SklearnCandidateTrainer,
    SklearnReplayCandidateProducer,
    TemporaryCandidateDirectory,
    TrainingDataset,
    canonical_feature_snapshots,
    classify_feature_regimes,
)
from .walk_forward import (
    PurgedWalkForwardSplitter,
    WalkForwardConfig,
)


@dataclass(frozen=True, slots=True)
class AlphaCertificationConfig:
    walk_forward: WalkForwardConfig
    training: CandidateTrainingConfig
    gates: AlphaGateThresholds = AlphaGateThresholds()
    costs: ReplayCostModel = ReplayCostModel()
    cost_stress_multiplier: Decimal = Decimal("1.5")
    initial_equity: Decimal = Decimal("10000")
    candidate_workspace: Path | None = None

    def __post_init__(self) -> None:
        if (
            not self.cost_stress_multiplier.is_finite()
            or self.cost_stress_multiplier < Decimal("1")
        ):
            raise ValueError("cost_stress_multiplier must be finite and at least 1")


@dataclass(frozen=True, slots=True)
class AlphaCertificationResult:
    status: str
    signed_model_card: Mapping[str, Any]
    report_path: Path
    staged_model_path: Path | None
    promoted_model_path: Path | None
    fold_metrics: tuple[ReplayMetrics, ...]


class AlphaCertificationController:
    """Train in temporary storage, replay OOS, and fail closed on promotion."""

    def __init__(
        self,
        *,
        config: AlphaCertificationConfig,
        constraints_by_symbol: Mapping[str, InstrumentConstraints],
        risk_policy: RiskPolicy,
        registry: PromotionRegistry,
    ) -> None:
        if not constraints_by_symbol:
            raise ValueError("instrument constraints are required")
        self.config = config
        self.constraints_by_symbol = {
            symbol.strip().upper(): constraints
            for symbol, constraints in constraints_by_symbol.items()
        }
        self.risk_policy = risk_policy
        self.registry = registry

    async def certify(
        self,
        snapshots: Sequence[DatasetSnapshot],
        *,
        git_commit: str,
        code_sha256: str,
        source_lock_sha256: str,
        reliability_document: Mapping[str, Any] | None = None,
        reliability_key: bytes | None = None,
        shadow_document: Mapping[str, Any] | None = None,
        shadow_key: bytes | None = None,
    ) -> AlphaCertificationResult:
        if not snapshots:
            raise ValueError("alpha certification requires at least one dataset")
        trainer = SklearnCandidateTrainer(self.config.training)
        fold_metrics: list[ReplayMetrics] = []
        stress_fold_metrics: list[ReplayMetrics] = []
        fold_evidence: list[dict[str, Any]] = []

        for snapshot in snapshots:
            symbol = snapshot.request.symbol
            constraints = self.constraints_by_symbol.get(symbol)
            if constraints is None:
                raise ValueError(f"missing constraints for {symbol}")
            minimum_purge = timedelta(
                seconds=(
                    snapshot.request.interval_seconds
                    * self.config.training.horizon_bars
                )
            )
            if self.config.walk_forward.purge < minimum_purge:
                raise ValueError(
                    f"walk-forward purge for {symbol} must be at least "
                    f"{int(minimum_purge.total_seconds())} seconds"
                )
            folds = PurgedWalkForwardSplitter(self.config.walk_forward).split(
                [candle.open_time for candle in snapshot.candles],
                dataset_end=snapshot.request.end,
                fold_prefix=symbol.lower(),
            )
            regime_labels = classify_feature_regimes(
                canonical_feature_snapshots(
                    TrainingDataset(
                        symbol=symbol,
                        interval=snapshot.request.interval,
                        candles=snapshot.candles,
                        dataset_sha256=snapshot.data_sha256,
                    )
                )
            )
            for fold in folds:
                train_candles = tuple(
                    snapshot.candles[index] for index in fold.train_indices
                )
                test_candles = tuple(
                    snapshot.candles[index] for index in fold.test_indices
                )
                test_start_index = fold.test_indices[0]
                warmup_start_index = max(0, test_start_index - 200)
                warmup_candles = tuple(
                    snapshot.candles[warmup_start_index:test_start_index]
                )
                with TemporaryCandidateDirectory(
                    self.config.candidate_workspace
                ) as candidate_directory:
                    artifact = trainer.train(
                        [
                            TrainingDataset(
                                symbol=symbol,
                                interval=snapshot.request.interval,
                                candles=train_candles,
                                dataset_sha256=snapshot.data_sha256,
                            )
                        ],
                        candidate_directory,
                    )
                    producer = SklearnReplayCandidateProducer(
                        artifact,
                        constraints,
                    )
                    replay = CanonicalAlphaReplay(
                        ReplayConfig(
                            symbol=symbol,
                            interval=snapshot.request.interval,
                            interval_seconds=snapshot.request.interval_seconds,
                            constraints=constraints,
                            risk_policy=self.risk_policy,
                            initial_equity=self.config.initial_equity,
                            costs=self.config.costs,
                            max_holding_bars=self.config.training.horizon_bars,
                            account_id=f"alpha-replay-{symbol.lower()}",
                        )
                    )
                    result = await replay.run(
                        test_candles,
                        CanonicalCandidateAdapter(producer),
                        warmup_candles=warmup_candles,
                    )
                    stress_costs = ReplayCostModel(
                        taker_fee_bps=(
                            self.config.costs.taker_fee_bps
                            * self.config.cost_stress_multiplier
                        ),
                        spread_bps=(
                            self.config.costs.spread_bps
                            * self.config.cost_stress_multiplier
                        ),
                        slippage_bps=(
                            self.config.costs.slippage_bps
                            * self.config.cost_stress_multiplier
                        ),
                    )
                    stress_result = await CanonicalAlphaReplay(
                        ReplayConfig(
                            symbol=symbol,
                            interval=snapshot.request.interval,
                            interval_seconds=snapshot.request.interval_seconds,
                            constraints=constraints,
                            risk_policy=self.risk_policy,
                            initial_equity=self.config.initial_equity,
                            costs=stress_costs,
                            max_holding_bars=self.config.training.horizon_bars,
                            account_id=f"alpha-cost-stress-{symbol.lower()}",
                        )
                    ).run(
                        test_candles,
                        CanonicalCandidateAdapter(
                            SklearnReplayCandidateProducer(
                                artifact,
                                constraints,
                            )
                        ),
                        warmup_candles=warmup_candles,
                    )
                fold_metrics.append(result.metrics)
                stress_fold_metrics.append(stress_result.metrics)
                fold_evidence.append(
                    {
                        "dataset_sha256": snapshot.data_sha256,
                        "symbol": symbol,
                        **fold.to_payload(),
                        "artifact_sha256": artifact.sha256,
                        "feature_warmup": {
                            "samples": len(warmup_candles),
                            "start": (
                                warmup_candles[0].open_time.isoformat()
                                if warmup_candles
                                else None
                            ),
                            "end": (
                                warmup_candles[-1].close_time.isoformat()
                                if warmup_candles
                                else None
                            ),
                        },
                        "base_costs": result.cost_model.to_payload(),
                        "regime_counts": dict(
                            sorted(
                                Counter(
                                    regime_labels[index] for index in fold.test_indices
                                ).items()
                            )
                        ),
                        "trace_digest": result.trace_digest,
                        "metrics": result.metrics.to_payload(),
                        "cost_stress": {
                            "multiplier": str(self.config.cost_stress_multiplier),
                            "costs": stress_result.cost_model.to_payload(),
                            "trace_digest": stress_result.trace_digest,
                            "metrics": stress_result.metrics.to_payload(),
                        },
                    }
                )

        alpha_decision = evaluate_alpha_gates(
            fold_metrics,
            self.config.gates,
        )
        alpha_decision = self._with_dataset_gate(
            alpha_decision,
            snapshots,
            fold_evidence,
            self.config.gates,
            stress_fold_metrics,
            self.config.cost_stress_multiplier,
        )
        final_training_sets = [
            TrainingDataset(
                symbol=snapshot.request.symbol,
                interval=snapshot.request.interval,
                candles=snapshot.candles,
                dataset_sha256=snapshot.data_sha256,
            )
            for snapshot in snapshots
        ]
        with TemporaryCandidateDirectory(
            self.config.candidate_workspace
        ) as candidate_directory:
            final_artifact = trainer.train(
                final_training_sets,
                candidate_directory,
            )
            signed, report_path, staged_path, promoted_path = self.registry.record(
                artifact=final_artifact,
                datasets=snapshots,
                walk_forward=self.config.walk_forward,
                fold_metrics=fold_metrics,
                fold_evidence=fold_evidence,
                alpha_decision=alpha_decision,
                thresholds=self.config.gates,
                git_commit=git_commit,
                code_sha256=code_sha256,
                source_lock_sha256=source_lock_sha256,
                reliability_document=reliability_document,
                reliability_key=reliability_key,
                shadow_document=shadow_document,
                shadow_key=shadow_key,
            )
        return AlphaCertificationResult(
            status=str(signed["status"]),
            signed_model_card=signed,
            report_path=report_path,
            staged_model_path=staged_path,
            promoted_model_path=promoted_path,
            fold_metrics=tuple(fold_metrics),
        )

    @staticmethod
    def _with_dataset_gate(
        decision: PromotionDecision,
        snapshots: Sequence[DatasetSnapshot],
        fold_evidence: Sequence[Mapping[str, Any]],
        thresholds: AlphaGateThresholds,
        stress_fold_metrics: Sequence[ReplayMetrics],
        cost_stress_multiplier: Decimal,
    ) -> PromotionDecision:
        violations = [
            {
                "data_sha256": snapshot.data_sha256,
                "gap_policy": snapshot.request.gap_policy.value,
                "gaps": [value.isoformat() for value in snapshot.gaps],
            }
            for snapshot in snapshots
            if snapshot.gaps and snapshot.request.gap_policy is not GapPolicy.ALLOWLIST
        ]
        regime_sets = [
            {
                str(name)
                for name, count in evidence.get("regime_counts", {}).items()
                if int(count) > 0
            }
            for evidence in fold_evidence
        ]
        observed_regimes = set().union(*regime_sets) if regime_sets else set()
        regime_gate_passed = (
            len(observed_regimes) >= thresholds.min_oos_regimes
            and bool(regime_sets)
            and all(
                len(values) >= thresholds.min_regimes_per_fold for values in regime_sets
            )
        )
        stress_decision = evaluate_alpha_gates(
            stress_fold_metrics,
            thresholds,
        )
        gates = {
            **decision.gates,
            "dataset_gaps": {
                "passed": not violations,
                "observed": violations,
                "required": "zero gaps or an explicit allowlist",
            },
            "multi_regime_oos": {
                "passed": regime_gate_passed,
                "observed": {
                    "regimes": sorted(observed_regimes),
                    "regimes_per_fold": [len(values) for values in regime_sets],
                },
                "required": {
                    "overall_regimes": thresholds.min_oos_regimes,
                    "regimes_per_fold": thresholds.min_regimes_per_fold,
                },
            },
            "cost_stress": {
                "passed": stress_decision.passed,
                "observed": {
                    "multiplier": str(cost_stress_multiplier),
                    "gates": stress_decision.gates,
                },
                "required": (
                    "all alpha quality, stability, baseline, and integrity "
                    "gates pass under stressed costs"
                ),
            },
        }
        return PromotionDecision(
            passed=all(gate.get("passed") is True for gate in gates.values()),
            gates=gates,
        )
