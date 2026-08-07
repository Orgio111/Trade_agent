"""Temporary-only reference training and frozen replay candidate artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN
import hashlib
import math
from pathlib import Path
from statistics import median
import tempfile
from typing import Any, Literal, Mapping, Sequence

from packages.domain import CandlePayload, MarketEvent, SourceMode
from packages.execution import MarketSnapshot
from packages.risk import CandidateSignal, InstrumentConstraints
from workers.contracts import (
    ALPHA_SHADOW_MODEL,
    CandidateForRiskEvent,
)
from workers.features import FeatureSnapshot, IncrementalFeatureEngine

from .data import HistoricalCandle


FEATURE_SCHEMA = "canonical-feature-snapshot-alpha-v1"
FEATURE_NAMES = (
    "rsi_14",
    "trend_strength",
    "volatility",
    "atr_fraction",
    "ema_9_fraction",
    "ema_21_fraction",
    "ema_50_fraction",
    "vwap_fraction",
    "support_fraction",
    "resistance_fraction",
    "liquidity_log",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class TrainingDataset:
    symbol: str
    interval: str
    candles: tuple[HistoricalCandle, ...]
    dataset_sha256: str

    def __post_init__(self) -> None:
        if not self.candles:
            raise ValueError("training dataset cannot be empty")
        if len(self.dataset_sha256) != 64:
            raise ValueError("training dataset hash must be SHA-256")


@dataclass(frozen=True, slots=True)
class CandidateTrainingConfig:
    horizon_bars: int = 12
    buy_return_threshold: float = 0.005
    sell_return_threshold: float = -0.005
    minimum_confidence: float = 0.55
    stop_distance_bps: Decimal = Decimal("100")
    take_profit_distance_bps: Decimal = Decimal("200")
    requested_risk_fraction: Decimal = Decimal("0.0025")
    min_training_samples: int = 500
    random_state: int = 42
    n_estimators: int = 300

    def __post_init__(self) -> None:
        if self.horizon_bars < 1:
            raise ValueError("horizon_bars must be positive")
        if not 0 < self.buy_return_threshold < 1:
            raise ValueError("buy_return_threshold must be in (0, 1)")
        if not -1 < self.sell_return_threshold < 0:
            raise ValueError("sell_return_threshold must be in (-1, 0)")
        if not 0 < self.minimum_confidence <= 1:
            raise ValueError("minimum_confidence must be in (0, 1]")
        if self.stop_distance_bps <= 0 or self.take_profit_distance_bps <= 0:
            raise ValueError("stop and target distances must be positive")
        if not Decimal("0") < self.requested_risk_fraction <= Decimal("1"):
            raise ValueError("requested_risk_fraction must be in (0, 1]")
        if self.min_training_samples < 100:
            raise ValueError("min_training_samples must be at least 100")
        if not 10 <= self.n_estimators <= 2_000:
            raise ValueError("n_estimators must be between 10 and 2000")

    def to_payload(self) -> dict[str, object]:
        return {
            "horizon_bars": self.horizon_bars,
            "buy_return_threshold": self.buy_return_threshold,
            "sell_return_threshold": self.sell_return_threshold,
            "minimum_confidence": self.minimum_confidence,
            "stop_distance_bps": str(self.stop_distance_bps),
            "take_profit_distance_bps": str(self.take_profit_distance_bps),
            "requested_risk_fraction": str(self.requested_risk_fraction),
            "min_training_samples": self.min_training_samples,
            "random_state": self.random_state,
            "n_estimators": self.n_estimators,
        }


@dataclass(frozen=True, slots=True)
class CandidateArtifact:
    path: Path
    sha256: str
    trainer_id: str
    feature_schema: str
    training_samples: int
    symbols: tuple[str, ...]
    dataset_sha256s: tuple[str, ...]
    config: CandidateTrainingConfig

    def to_payload(self) -> dict[str, object]:
        return {
            "artifact_file": self.path.name,
            "artifact_sha256": self.sha256,
            "trainer_id": self.trainer_id,
            "feature_schema": self.feature_schema,
            "training_samples": self.training_samples,
            "symbols": list(self.symbols),
            "dataset_sha256s": list(self.dataset_sha256s),
            "training_config": self.config.to_payload(),
        }


class TemporaryCandidateDirectory:
    """A candidate workspace that cannot target the running model directory."""

    def __init__(self, parent: Path | None = None) -> None:
        self._parent = None if parent is None else parent.resolve()
        self._temporary: tempfile.TemporaryDirectory[str] | None = None

    def __enter__(self) -> Path:
        if self._parent is not None:
            self._parent.mkdir(parents=True, exist_ok=True)
        self._temporary = tempfile.TemporaryDirectory(
            prefix="alpha-candidate-",
            dir=None if self._parent is None else str(self._parent),
        )
        path = Path(self._temporary.name).resolve()
        repository_models = (Path(__file__).resolve().parents[2] / "models").resolve()
        if path == repository_models or repository_models in path.parents:
            self._temporary.cleanup()
            self._temporary = None
            raise ValueError(
                "candidate training cannot use the repository model directory"
            )
        return path

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None


class SklearnCandidateTrainer:
    """Train a deterministic three-class RF on canonical feature snapshots."""

    trainer_id = "sklearn-random-forest-canonical-features-v1"

    def __init__(self, config: CandidateTrainingConfig | None = None) -> None:
        self.config = config or CandidateTrainingConfig()

    def train(
        self,
        datasets: Sequence[TrainingDataset],
        output_directory: Path,
    ) -> CandidateArtifact:
        if not datasets:
            raise ValueError("at least one training dataset is required")
        output_directory = output_directory.resolve()
        repository_models = (Path(__file__).resolve().parents[2] / "models").resolve()
        if (
            output_directory == repository_models
            or repository_models in output_directory.parents
        ):
            raise ValueError("training output cannot target repository models")
        output_directory.mkdir(parents=True, exist_ok=True)
        features: list[tuple[float, ...]] = []
        labels: list[int] = []
        for dataset in datasets:
            dataset_features, dataset_labels = self._dataset_samples(dataset)
            features.extend(dataset_features)
            labels.extend(dataset_labels)
        if len(features) < self.config.min_training_samples:
            raise ValueError(
                f"training produced {len(features)} samples; "
                f"minimum is {self.config.min_training_samples}"
            )
        if len(set(labels)) < 2:
            raise ValueError("training labels must contain at least two classes")
        try:
            import joblib
            import sklearn
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
        except ImportError as exc:
            raise RuntimeError(
                "alpha candidate training requires the dev ML dependencies"
            ) from exc

        pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=self.config.n_estimators,
                        max_depth=12,
                        min_samples_leaf=10,
                        class_weight="balanced",
                        random_state=self.config.random_state,
                        n_jobs=1,
                    ),
                ),
            ]
        )
        pipeline.fit(features, labels)
        payload = {
            "schema_version": 1,
            "trainer_id": self.trainer_id,
            "feature_schema": FEATURE_SCHEMA,
            "feature_names": FEATURE_NAMES,
            "training_config": self.config.to_payload(),
            "training_samples": len(features),
            "symbols": tuple(sorted({dataset.symbol for dataset in datasets})),
            "dataset_sha256s": tuple(
                sorted({dataset.dataset_sha256 for dataset in datasets})
            ),
            "sklearn_version": sklearn.__version__,
            "pipeline": pipeline,
        }
        artifact_path = output_directory / "candidate.joblib"
        if artifact_path.exists():
            raise FileExistsError("candidate artifact path already exists")
        joblib.dump(payload, artifact_path, compress=3)
        digest = _sha256_file(artifact_path)
        return CandidateArtifact(
            path=artifact_path,
            sha256=digest,
            trainer_id=self.trainer_id,
            feature_schema=FEATURE_SCHEMA,
            training_samples=len(features),
            symbols=tuple(payload["symbols"]),
            dataset_sha256s=tuple(payload["dataset_sha256s"]),
            config=self.config,
        )

    def _dataset_samples(
        self,
        dataset: TrainingDataset,
    ) -> tuple[list[tuple[float, ...]], list[int]]:
        snapshots = canonical_feature_snapshots(dataset)
        features: list[tuple[float, ...]] = []
        labels: list[int] = []
        horizon = self.config.horizon_bars
        for index in range(49, len(dataset.candles) - horizon):
            current = dataset.candles[index].close
            future = dataset.candles[index + horizon].close
            future_return = float(future / current - Decimal("1"))
            label = (
                1
                if future_return > self.config.buy_return_threshold
                else -1
                if future_return < self.config.sell_return_threshold
                else 0
            )
            features.append(feature_vector(snapshots[index], current))
            labels.append(label)
        return features, labels


def canonical_feature_snapshots(
    dataset: TrainingDataset,
) -> tuple[FeatureSnapshot, ...]:
    """Compute the same bounded feature state once in chronological order."""

    engine = IncrementalFeatureEngine(max_age_seconds=90)
    snapshots: list[FeatureSnapshot] = []
    for sequence, candle in enumerate(dataset.candles, 1):
        event = _market_event(
            candle,
            symbol=dataset.symbol,
            interval=dataset.interval,
            sequence=sequence,
            ingest_run_id=f"alpha-features-{dataset.dataset_sha256[:16]}",
        )
        snapshots.append(
            engine.update(
                event,
                evaluated_at=candle.close_time + timedelta(milliseconds=50),
            )
        )
    return tuple(snapshots)


def classify_feature_regimes(
    snapshots: Sequence[FeatureSnapshot],
) -> tuple[str, ...]:
    """Assign causal trend/volatility regimes without future information."""

    labels: list[str] = []
    observed_volatility: list[Decimal] = []
    for snapshot in snapshots:
        history = observed_volatility[-50:]
        reference = median(history) if history else snapshot.volatility
        volatility = "high_vol" if snapshot.volatility > reference else "low_vol"
        labels.append(f"{snapshot.trend_direction}_{volatility}")
        observed_volatility.append(snapshot.volatility)
    return tuple(labels)


class SklearnReplayCandidateProducer:
    """Load a digest-bound artifact and emit canonical candidate envelopes."""

    def __init__(
        self,
        artifact: CandidateArtifact,
        constraints: InstrumentConstraints | Mapping[str, InstrumentConstraints],
    ) -> None:
        if _sha256_file(artifact.path) != artifact.sha256:
            raise ValueError("candidate artifact hash mismatch")
        try:
            import joblib
        except ImportError as exc:
            raise RuntimeError("candidate replay requires joblib") from exc
        payload: dict[str, Any] = joblib.load(artifact.path)
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported candidate artifact schema")
        if payload.get("trainer_id") != artifact.trainer_id:
            raise ValueError("candidate trainer identity mismatch")
        if payload.get("feature_schema") != artifact.feature_schema:
            raise ValueError("candidate feature schema mismatch")
        if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("candidate feature order mismatch")
        self._artifact = artifact
        self._pipeline = payload["pipeline"]
        self._constraints = (
            {constraints.instrument: constraints}
            if isinstance(constraints, InstrumentConstraints)
            else {
                symbol.strip().upper(): value for symbol, value in constraints.items()
            }
        )

    async def generate_signal(
        self,
        feature: FeatureSnapshot,
        market: MarketSnapshot,
        *,
        generated_at: datetime,
        source_mode: SourceMode = SourceMode.REPLAY,
    ) -> CandidateSignal | None:
        if (
            not feature.fresh
            or not feature.verify_checksum()
            or feature.candle_count < 50
            or feature.symbol != market.instrument
        ):
            return None
        constraints = self._constraints.get(market.instrument)
        if constraints is None:
            return None
        probabilities = self._pipeline.predict_proba(
            [feature_vector(feature, market.last or ((market.bid + market.ask) / 2))]
        )[0]
        classes = tuple(
            int(value) for value in self._pipeline.named_steps["classifier"].classes_
        )
        best_index = max(range(len(probabilities)), key=probabilities.__getitem__)
        label = classes[best_index]
        confidence = float(probabilities[best_index])
        if label == 0 or confidence < self._artifact.config.minimum_confidence:
            return None
        side: Literal["buy", "sell"] = "buy" if label == 1 else "sell"
        midpoint = (market.bid + market.ask) / Decimal("2")
        reference = _round_tick(
            midpoint,
            constraints.tick_size,
            ROUND_HALF_EVEN,
        )
        stop_fraction = self._artifact.config.stop_distance_bps / Decimal("10000")
        target_fraction = self._artifact.config.take_profit_distance_bps / Decimal(
            "10000"
        )
        if side == "buy":
            stop = _round_tick(
                reference * (Decimal("1") - stop_fraction),
                constraints.tick_size,
                ROUND_FLOOR,
            )
            target = _round_tick(
                reference * (Decimal("1") + target_fraction),
                constraints.tick_size,
                ROUND_CEILING,
            )
        else:
            stop = _round_tick(
                reference * (Decimal("1") + stop_fraction),
                constraints.tick_size,
                ROUND_CEILING,
            )
            target = _round_tick(
                reference * (Decimal("1") - target_fraction),
                constraints.tick_size,
                ROUND_FLOOR,
            )
        age = Decimal(str((generated_at - market.observed_at).total_seconds()))
        spread = (market.ask - market.bid) / midpoint * Decimal("10000")
        identity = hashlib.sha256(
            (
                f"{self._artifact.sha256}:{feature.input_event_id}:"
                f"{feature.checksum}:{side}"
            ).encode()
        ).hexdigest()
        return CandidateSignal(
            trace_id=f"alpha-model:{feature.input_event_id}",
            signal_id=f"sig_{identity[:24]}",
            strategy_id=ALPHA_SHADOW_MODEL,
            strategy_version="1.0.0",
            venue=market.venue,
            market_type=market.market_type,
            instrument=market.instrument,
            side=side,
            reference_price=reference,
            stop_price=stop,
            take_profit_price=target,
            confidence=Decimal(str(confidence)),
            spread_bps=spread,
            expected_slippage_bps=spread / Decimal("2"),
            data_age_seconds=age,
            source_mode=source_mode,
            requested_risk_fraction=self._artifact.config.requested_risk_fraction,
        )

    async def generate(
        self,
        feature: FeatureSnapshot,
        market: MarketSnapshot,
        *,
        generated_at: datetime,
        source_mode: SourceMode = SourceMode.REPLAY,
    ) -> CandidateForRiskEvent | None:
        if source_mode not in {SourceMode.REPLAY, SourceMode.PAPER_LIVE}:
            raise ValueError("alpha candidate admits only replay or paper_live")
        candidate = await self.generate_signal(
            feature,
            market,
            generated_at=generated_at,
            source_mode=source_mode,
        )
        if candidate is None:
            return None
        return CandidateForRiskEvent(
            trace_id=candidate.trace_id,
            market_event_id=feature.input_event_id,
            provider="alpha_shadow",
            model_role=None,
            model=ALPHA_SHADOW_MODEL,
            model_digest=self._artifact.sha256,
            candidate=candidate,
            market=market,
            created_at=generated_at,
        )


class SklearnShadowCandidateProducer:
    """Run one digest-bound candidate through the canonical paper boundary."""

    def __init__(
        self,
        artifact_path: Path,
        *,
        expected_sha256: str,
        constraints: InstrumentConstraints | Mapping[str, InstrumentConstraints],
    ) -> None:
        artifact = load_candidate_artifact(
            artifact_path,
            expected_sha256=expected_sha256,
        )
        self._artifact = artifact
        self._producer = SklearnReplayCandidateProducer(artifact, constraints)

    async def generate(
        self,
        feature: FeatureSnapshot,
        market: MarketSnapshot,
        *,
        generated_at: datetime,
        source_mode: SourceMode = SourceMode.PAPER_LIVE,
    ) -> CandidateForRiskEvent | None:
        if source_mode is not SourceMode.PAPER_LIVE:
            raise ValueError("alpha-shadow provider admits only paper_live mode")
        return await self._producer.generate(
            feature,
            market,
            generated_at=generated_at,
            source_mode=source_mode,
        )


def load_candidate_artifact(
    path: Path,
    *,
    expected_sha256: str,
) -> CandidateArtifact:
    """Load metadata only after the exact staged artifact digest is verified."""

    if path.is_symlink():
        raise ValueError("candidate artifact must be a regular non-symlink file")
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError("candidate artifact must be a regular non-symlink file")
    if (
        len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
        or _sha256_file(resolved) != expected_sha256
    ):
        raise ValueError("candidate artifact hash mismatch")
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("candidate artifact loading requires joblib") from exc
    payload: dict[str, Any] = joblib.load(resolved)
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported candidate artifact schema")
    if payload.get("trainer_id") != SklearnCandidateTrainer.trainer_id:
        raise ValueError("candidate trainer identity mismatch")
    if payload.get("feature_schema") != FEATURE_SCHEMA:
        raise ValueError("candidate feature schema mismatch")
    if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("candidate feature order mismatch")
    config_payload = payload.get("training_config")
    if not isinstance(config_payload, dict):
        raise ValueError("candidate training configuration is missing")
    config = CandidateTrainingConfig(
        horizon_bars=int(config_payload["horizon_bars"]),
        buy_return_threshold=float(config_payload["buy_return_threshold"]),
        sell_return_threshold=float(config_payload["sell_return_threshold"]),
        minimum_confidence=float(config_payload["minimum_confidence"]),
        stop_distance_bps=Decimal(str(config_payload["stop_distance_bps"])),
        take_profit_distance_bps=Decimal(
            str(config_payload["take_profit_distance_bps"])
        ),
        requested_risk_fraction=Decimal(str(config_payload["requested_risk_fraction"])),
        min_training_samples=int(config_payload["min_training_samples"]),
        random_state=int(config_payload["random_state"]),
        n_estimators=int(config_payload["n_estimators"]),
    )
    symbols = tuple(str(value) for value in payload.get("symbols", ()))
    datasets = tuple(str(value) for value in payload.get("dataset_sha256s", ()))
    if not symbols or not datasets or any(len(value) != 64 for value in datasets):
        raise ValueError("candidate artifact provenance is incomplete")
    return CandidateArtifact(
        path=resolved,
        sha256=expected_sha256,
        trainer_id=SklearnCandidateTrainer.trainer_id,
        feature_schema=FEATURE_SCHEMA,
        training_samples=int(payload.get("training_samples", 0)),
        symbols=symbols,
        dataset_sha256s=datasets,
        config=config,
    )


def feature_vector(
    feature: FeatureSnapshot,
    price: Decimal,
) -> tuple[float, ...]:
    if price <= 0:
        raise ValueError("feature-vector price must be positive")
    return (
        float(feature.rsi_14 / Decimal("100")),
        float(feature.trend_strength),
        float(feature.volatility),
        float(feature.atr / price),
        float(feature.ema_9 / price - Decimal("1")),
        float(feature.ema_21 / price - Decimal("1")),
        float(feature.ema_50 / price - Decimal("1")),
        float(feature.session_vwap / price - Decimal("1")),
        float(feature.support / price - Decimal("1")),
        float(feature.resistance / price - Decimal("1")),
        math.log1p(float(feature.liquidity_proxy)),
    )


def _round_tick(value: Decimal, tick: Decimal, rounding: str) -> Decimal:
    return (value / tick).to_integral_value(rounding=rounding) * tick


def _market_event(
    candle: HistoricalCandle,
    *,
    symbol: str,
    interval: str,
    sequence: int,
    ingest_run_id: str,
) -> MarketEvent:
    return MarketEvent.create(
        trace_id=f"{ingest_run_id}:{sequence}",
        event_type="market.candle",
        venue="binance",
        market_type="spot",
        instrument_id=symbol,
        exchange_ts=candle.close_time,
        received_ts=candle.close_time + timedelta(milliseconds=50),
        source_mode=SourceMode.REPLAY,
        ingest_run_id=ingest_run_id,
        payload=CandlePayload(
            interval=interval,
            open_time=candle.open_time,
            close_time=candle.close_time,
            open=candle.open,
            high=candle.high,
            low=candle.low,
            close=candle.close,
            volume=candle.volume,
            quote_volume=candle.quote_volume,
            trade_count=candle.trade_count,
        ),
        sequence_start=sequence,
        sequence_end=sequence,
    )
