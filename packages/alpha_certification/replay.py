"""Exact canonical feature/candidate/risk replay with conservative fills."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext
import hashlib
import math
from statistics import mean, pstdev
from typing import Protocol

from packages.domain import CandlePayload, MarketEvent, SourceMode
from packages.execution import MarketSnapshot
from packages.risk import (
    CandidateSignal,
    InstrumentConstraints,
    PortfolioState,
    RiskDecision,
    RiskEngine,
    RiskPolicy,
)
from workers.candidate.runtime import CandidateProducer
from workers.features import FeatureSnapshot, IncrementalFeatureEngine

from .data import HistoricalCandle


_ZERO = Decimal("0")
_TEN_THOUSAND = Decimal("10000")


class ReplayCandidateProducer(Protocol):
    async def generate_signal(
        self,
        feature: FeatureSnapshot,
        market: MarketSnapshot,
        *,
        generated_at: datetime,
    ) -> CandidateSignal | None: ...


class CanonicalCandidateAdapter:
    """Use an actual canonical worker candidate producer during replay."""

    def __init__(self, producer: CandidateProducer) -> None:
        self._producer = producer

    async def generate_signal(
        self,
        feature: FeatureSnapshot,
        market: MarketSnapshot,
        *,
        generated_at: datetime,
    ) -> CandidateSignal | None:
        event = await self._producer.generate(
            feature,
            market,
            generated_at=generated_at,
            source_mode=SourceMode.REPLAY,
        )
        return None if event is None else event.candidate


@dataclass(frozen=True, slots=True)
class ReplayCostModel:
    taker_fee_bps: Decimal = Decimal("4")
    spread_bps: Decimal = Decimal("4")
    slippage_bps: Decimal = Decimal("3")

    def __post_init__(self) -> None:
        for value in (
            self.taker_fee_bps,
            self.spread_bps,
            self.slippage_bps,
        ):
            if not value.is_finite() or value < 0:
                raise ValueError("replay costs must be finite and non-negative")

    @property
    def roundtrip_bps(self) -> Decimal:
        return Decimal("2") * (
            self.taker_fee_bps + self.spread_bps / Decimal("2") + self.slippage_bps
        )

    def to_payload(self) -> dict[str, str]:
        return {
            "taker_fee_bps": str(self.taker_fee_bps),
            "spread_bps": str(self.spread_bps),
            "slippage_bps": str(self.slippage_bps),
            "roundtrip_bps": str(self.roundtrip_bps),
        }


@dataclass(frozen=True, slots=True)
class ReplayConfig:
    symbol: str
    interval: str
    interval_seconds: int
    constraints: InstrumentConstraints
    risk_policy: RiskPolicy
    initial_equity: Decimal = Decimal("10000")
    costs: ReplayCostModel = ReplayCostModel()
    max_holding_bars: int = 12
    account_id: str = "alpha-replay"

    def __post_init__(self) -> None:
        if self.symbol.strip().upper() != self.constraints.instrument:
            raise ValueError("replay symbol and constraints must match")
        if self.interval_seconds < 1:
            raise ValueError("interval_seconds must be positive")
        if self.initial_equity <= 0:
            raise ValueError("initial_equity must be positive")
        if self.max_holding_bars < 1:
            raise ValueError("max_holding_bars must be positive")


@dataclass(frozen=True, slots=True)
class ReplayTrade:
    signal_id: str
    side: str
    signal_time: datetime
    entry_time: datetime
    exit_time: datetime
    entry_price: Decimal
    exit_price: Decimal
    quantity: Decimal
    gross_pnl: Decimal
    fees: Decimal
    net_pnl: Decimal
    exit_reason: str
    holding_bars: int

    def to_payload(self) -> dict[str, object]:
        return {
            "signal_id": self.signal_id,
            "side": self.side,
            "signal_time": self.signal_time.isoformat(),
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "entry_price": str(self.entry_price),
            "exit_price": str(self.exit_price),
            "quantity": str(self.quantity),
            "gross_pnl": str(self.gross_pnl),
            "fees": str(self.fees),
            "net_pnl": str(self.net_pnl),
            "exit_reason": self.exit_reason,
            "holding_bars": self.holding_bars,
        }


@dataclass(frozen=True, slots=True)
class ReplayMetrics:
    bars: int
    candidates: int
    approved_decisions: int
    rejected_decisions: int
    stale_pending_orders: int
    trades: int
    net_return: float
    buy_hold_return: float
    max_drawdown: float
    profit_factor: float | None
    annualized_sharpe: float
    win_rate: float
    expectancy: float
    duplicate_signals: int
    duplicate_fills: int
    reconciliation_mismatches: int
    next_bar_fill_violations: int

    def to_payload(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class ReplayResult:
    metrics: ReplayMetrics
    trades: tuple[ReplayTrade, ...]
    equity_curve: tuple[Decimal, ...]
    trace_digest: str
    cost_model: ReplayCostModel


@dataclass(slots=True)
class _PendingOrder:
    candidate: CandidateSignal
    decision: RiskDecision
    signal_time: datetime


@dataclass(slots=True)
class _OpenPosition:
    candidate: CandidateSignal
    signal_time: datetime
    entry_time: datetime
    entry_price: Decimal
    quantity: Decimal
    entry_fee: Decimal
    holding_bars: int = 0


class CanonicalAlphaReplay:
    """Replay exact canonical boundaries while keeping execution conservative."""

    def __init__(self, config: ReplayConfig) -> None:
        self.config = config

    async def run(
        self,
        candles: tuple[HistoricalCandle, ...],
        producer: ReplayCandidateProducer,
        *,
        warmup_candles: tuple[HistoricalCandle, ...] = (),
    ) -> ReplayResult:
        if len(candles) < 2:
            raise ValueError("canonical alpha replay requires at least two candles")
        self._validate_ordered(candles, label="replay")
        self._validate_ordered(warmup_candles, label="warmup")
        if warmup_candles and warmup_candles[-1].close_time > candles[0].open_time:
            raise ValueError(
                "warmup candles must close before the replay window begins"
            )
        with localcontext() as context:
            context.prec = 50
            return await self._run(
                candles,
                producer,
                warmup_candles=warmup_candles,
            )

    async def _run(
        self,
        candles: tuple[HistoricalCandle, ...],
        producer: ReplayCandidateProducer,
        *,
        warmup_candles: tuple[HistoricalCandle, ...],
    ) -> ReplayResult:
        feature_engine = IncrementalFeatureEngine(max_age_seconds=90)
        for sequence, candle in enumerate(warmup_candles, 1):
            event = self._market_event(candle, sequence)
            feature_engine.update(
                event,
                evaluated_at=candle.close_time + timedelta(milliseconds=50),
            )

        risk_engine = RiskEngine(self.config.risk_policy)
        realized = _ZERO
        pending: _PendingOrder | None = None
        position: _OpenPosition | None = None
        trades: list[ReplayTrade] = []
        processed_candles: list[HistoricalCandle] = []
        equity_curve: list[Decimal] = [self.config.initial_equity]
        seen_signals: set[str] = set()
        fill_ids: set[str] = set()
        duplicate_signals = 0
        duplicate_fills = 0
        next_bar_violations = 0
        candidates = 0
        approved = 0
        rejected = 0
        stale_pending = 0
        day_key = candles[0].open_time.date()
        week_key = candles[0].open_time.isocalendar()[:2]
        day_base = self.config.initial_equity
        week_base = self.config.initial_equity
        consecutive_losses = 0

        sequence_offset = len(warmup_candles)
        for sequence, candle in enumerate(candles, sequence_offset + 1):
            processed_candles.append(candle)
            mark_before = self._mark_equity(realized, position, candle.open)
            current_day = candle.open_time.date()
            current_week = candle.open_time.isocalendar()[:2]
            if current_day != day_key:
                day_key = current_day
                day_base = mark_before
            if current_week != week_key:
                week_key = current_week
                week_base = mark_before

            if pending is not None and position is None:
                position = self._fill_pending(pending, candle)
                if position is None:
                    stale_pending += 1
                else:
                    fill_id = self._fill_id(
                        pending.candidate.signal_id,
                        candle.open_time,
                        "entry",
                    )
                    if fill_id in fill_ids:
                        duplicate_fills += 1
                    fill_ids.add(fill_id)
                    if position.entry_time < pending.signal_time:
                        next_bar_violations += 1
                    realized -= position.entry_fee
                pending = None

            if position is not None:
                exit_values = self._exit_for_candle(position, candle)
                if exit_values is not None:
                    exit_price, reason = exit_values
                    gross = self._gross_pnl(position, exit_price)
                    exit_fee = self._fee(exit_price, position.quantity)
                    realized += gross - exit_fee
                    trade = ReplayTrade(
                        signal_id=position.candidate.signal_id,
                        side=position.candidate.side,
                        signal_time=position.signal_time,
                        entry_time=position.entry_time,
                        exit_time=candle.close_time,
                        entry_price=position.entry_price,
                        exit_price=exit_price,
                        quantity=position.quantity,
                        gross_pnl=gross,
                        fees=position.entry_fee + exit_fee,
                        net_pnl=gross - position.entry_fee - exit_fee,
                        exit_reason=reason,
                        holding_bars=position.holding_bars + 1,
                    )
                    trades.append(trade)
                    consecutive_losses = (
                        0 if trade.net_pnl > 0 else consecutive_losses + 1
                    )
                    fill_id = self._fill_id(
                        position.candidate.signal_id,
                        candle.close_time,
                        "exit",
                    )
                    if fill_id in fill_ids:
                        duplicate_fills += 1
                    fill_ids.add(fill_id)
                    position = None
                else:
                    position.holding_bars += 1

            mark = self._mark_equity(realized, position, candle.close)
            equity_curve.append(max(_ZERO, mark))
            if mark <= 0:
                break

            event = self._market_event(candle, sequence)
            evaluated_at = candle.close_time + timedelta(milliseconds=50)
            feature = feature_engine.update(event, evaluated_at=evaluated_at)
            market = self._market_snapshot(candle.close, candle.close_time)
            if position is None and pending is None:
                candidate = await producer.generate_signal(
                    feature,
                    market,
                    generated_at=evaluated_at,
                )
                if candidate is not None:
                    candidates += 1
                    if candidate.signal_id in seen_signals:
                        duplicate_signals += 1
                    seen_signals.add(candidate.signal_id)
                    portfolio = self._portfolio(
                        realized=realized,
                        position=position,
                        mark_price=candle.close,
                        evaluated_at=evaluated_at,
                        day_base=day_base,
                        week_base=week_base,
                        state_sequence=sequence,
                        consecutive_losses=consecutive_losses,
                    )
                    decision = risk_engine.evaluate(
                        candidate,
                        portfolio,
                        self.config.constraints,
                        evaluated_at=evaluated_at,
                    )
                    if decision.approved:
                        approved += 1
                        pending = _PendingOrder(
                            candidate=candidate,
                            decision=decision,
                            signal_time=candle.close_time,
                        )
                    else:
                        rejected += 1

        if pending is not None:
            stale_pending += 1
            pending = None
        if position is not None:
            last = processed_candles[-1]
            exit_price = self._exit_price(
                last.close,
                side="sell" if position.candidate.side == "buy" else "buy",
            )
            gross = self._gross_pnl(position, exit_price)
            exit_fee = self._fee(exit_price, position.quantity)
            realized += gross - exit_fee
            trades.append(
                ReplayTrade(
                    signal_id=position.candidate.signal_id,
                    side=position.candidate.side,
                    signal_time=position.signal_time,
                    entry_time=position.entry_time,
                    exit_time=last.close_time,
                    entry_price=position.entry_price,
                    exit_price=exit_price,
                    quantity=position.quantity,
                    gross_pnl=gross,
                    fees=position.entry_fee + exit_fee,
                    net_pnl=gross - position.entry_fee - exit_fee,
                    exit_reason="end_of_data",
                    holding_bars=position.holding_bars,
                )
            )
            equity_curve[-1] = max(_ZERO, self.config.initial_equity + realized)
            position = None

        reconciliation_mismatches = int(position is not None or pending is not None)
        metrics = self._metrics(
            candles=tuple(processed_candles),
            trades=trades,
            equity_curve=equity_curve,
            candidates=candidates,
            approved=approved,
            rejected=rejected,
            stale_pending=stale_pending,
            duplicate_signals=duplicate_signals,
            duplicate_fills=duplicate_fills,
            reconciliation_mismatches=reconciliation_mismatches,
            next_bar_violations=next_bar_violations,
        )
        trace_digest = self._trace_digest(trades, metrics)
        return ReplayResult(
            metrics=metrics,
            trades=tuple(trades),
            equity_curve=tuple(equity_curve),
            trace_digest=trace_digest,
            cost_model=self.config.costs,
        )

    @staticmethod
    def _validate_ordered(
        candles: tuple[HistoricalCandle, ...],
        *,
        label: str,
    ) -> None:
        if any(
            current.open_time <= previous.open_time
            for previous, current in zip(candles, candles[1:], strict=False)
        ):
            raise ValueError(f"{label} candles must be strictly ordered")

    def _market_event(
        self,
        candle: HistoricalCandle,
        sequence: int,
    ) -> MarketEvent:
        return MarketEvent.create(
            trace_id=f"alpha-replay:{self.config.symbol}:{sequence}",
            event_type="market.candle",
            venue=self.config.constraints.venue,
            market_type=self.config.constraints.market_type,
            instrument_id=self.config.symbol,
            exchange_ts=candle.close_time,
            received_ts=candle.close_time + timedelta(milliseconds=50),
            source_mode=SourceMode.REPLAY,
            ingest_run_id=f"alpha-replay-{self.config.symbol}-{self.config.interval}",
            payload=CandlePayload(
                interval=self.config.interval,
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

    def _market_snapshot(
        self,
        midpoint: Decimal,
        observed_at: datetime,
    ) -> MarketSnapshot:
        half_spread = self.config.costs.spread_bps / Decimal("20000")
        return MarketSnapshot(
            venue=self.config.constraints.venue,
            market_type=self.config.constraints.market_type,
            instrument=self.config.symbol,
            bid=midpoint * (Decimal("1") - half_spread),
            ask=midpoint * (Decimal("1") + half_spread),
            last=midpoint,
            observed_at=observed_at,
        )

    def _fill_pending(
        self,
        pending: _PendingOrder,
        candle: HistoricalCandle,
    ) -> _OpenPosition | None:
        entry = self._exit_price(candle.open, side=pending.candidate.side)
        candidate = pending.candidate
        if candidate.side == "buy":
            valid = candidate.stop_price < entry and (
                candidate.take_profit_price is None
                or candidate.take_profit_price > entry
            )
        else:
            valid = candidate.stop_price > entry and (
                candidate.take_profit_price is None
                or candidate.take_profit_price < entry
            )
        if not valid:
            return None
        quantity = pending.decision.approved_quantity
        # The canonical risk engine sizes against reference-to-stop distance.
        # Reject an adverse next-open gap rather than silently exceeding that
        # authorization. Execution costs remain explicit in realized PnL.
        actual_stop_risk = abs(candle.open - candidate.stop_price) * quantity
        if actual_stop_risk > pending.decision.approved_risk_amount:
            return None
        return _OpenPosition(
            candidate=candidate,
            signal_time=pending.signal_time,
            entry_time=candle.open_time,
            entry_price=entry,
            quantity=quantity,
            entry_fee=self._fee(entry, quantity),
        )

    def _exit_for_candle(
        self,
        position: _OpenPosition,
        candle: HistoricalCandle,
    ) -> tuple[Decimal, str] | None:
        candidate = position.candidate
        target = candidate.take_profit_price
        if candidate.side == "buy":
            stop_hit = candle.low <= candidate.stop_price
            target_hit = target is not None and candle.high >= target
            if stop_hit:
                return self._exit_price(candidate.stop_price, side="sell"), "stop"
            if target_hit and target is not None:
                return self._exit_price(target, side="sell"), "target"
            if position.holding_bars + 1 >= self.config.max_holding_bars:
                return self._exit_price(candle.close, side="sell"), "time"
        else:
            stop_hit = candle.high >= candidate.stop_price
            target_hit = target is not None and candle.low <= target
            if stop_hit:
                return self._exit_price(candidate.stop_price, side="buy"), "stop"
            if target_hit and target is not None:
                return self._exit_price(target, side="buy"), "target"
            if position.holding_bars + 1 >= self.config.max_holding_bars:
                return self._exit_price(candle.close, side="buy"), "time"
        return None

    def _exit_price(self, price: Decimal, *, side: str) -> Decimal:
        cost_bps = self.config.costs.spread_bps / Decimal("2")
        cost_bps += self.config.costs.slippage_bps
        fraction = cost_bps / _TEN_THOUSAND
        return (
            price * (Decimal("1") + fraction)
            if side == "buy"
            else price * (Decimal("1") - fraction)
        )

    def _fee(self, price: Decimal, quantity: Decimal) -> Decimal:
        return price * quantity * self.config.costs.taker_fee_bps / _TEN_THOUSAND

    @staticmethod
    def _gross_pnl(position: _OpenPosition, exit_price: Decimal) -> Decimal:
        if position.candidate.side == "buy":
            return (exit_price - position.entry_price) * position.quantity
        return (position.entry_price - exit_price) * position.quantity

    def _mark_equity(
        self,
        realized: Decimal,
        position: _OpenPosition | None,
        mark_price: Decimal,
    ) -> Decimal:
        equity = self.config.initial_equity + realized
        if position is not None:
            equity += self._gross_pnl(position, mark_price)
            equity -= self._fee(mark_price, position.quantity)
        return equity

    def _portfolio(
        self,
        *,
        realized: Decimal,
        position: _OpenPosition | None,
        mark_price: Decimal,
        evaluated_at: datetime,
        day_base: Decimal,
        week_base: Decimal,
        state_sequence: int,
        consecutive_losses: int,
    ) -> PortfolioState:
        equity = self._mark_equity(realized, position, mark_price)
        return PortfolioState(
            account_id=self.config.account_id,
            state_id=f"alpha-replay-state-{state_sequence}",
            equity=max(Decimal("0.00000001"), equity),
            cash=max(_ZERO, self.config.initial_equity + realized),
            daily_pnl=equity - day_base,
            weekly_pnl=equity - week_base,
            consecutive_losses=consecutive_losses,
            open_positions=1 if position is not None else 0,
            gross_exposure=(
                _ZERO if position is None else position.quantity * mark_price
            ),
            reconciled_at=evaluated_at,
            kill_switch_active=False,
        )

    def _metrics(
        self,
        *,
        candles: tuple[HistoricalCandle, ...],
        trades: list[ReplayTrade],
        equity_curve: list[Decimal],
        candidates: int,
        approved: int,
        rejected: int,
        stale_pending: int,
        duplicate_signals: int,
        duplicate_fills: int,
        reconciliation_mismatches: int,
        next_bar_violations: int,
    ) -> ReplayMetrics:
        equity = [float(value) for value in equity_curve]
        returns = [
            current / previous - 1
            for previous, current in zip(equity, equity[1:], strict=False)
            if previous > 0
        ]
        peak = equity[0]
        maximum_drawdown = 0.0
        for value in equity:
            peak = max(peak, value)
            if peak > 0:
                maximum_drawdown = max(maximum_drawdown, 1 - value / peak)
        net_values = [float(trade.net_pnl) for trade in trades]
        gross_profit = sum(value for value in net_values if value > 0)
        gross_loss = abs(sum(value for value in net_values if value <= 0))
        profit_factor = (
            gross_profit / gross_loss if gross_loss else None if gross_profit else 0.0
        )
        periods_per_year = 365 * 24 * 60 * 60 / self.config.interval_seconds
        sharpe = (
            mean(returns) / pstdev(returns) * math.sqrt(periods_per_year)
            if len(returns) > 1 and pstdev(returns) > 0
            else 0.0
        )
        roundtrip = self.config.costs.roundtrip_bps / _TEN_THOUSAND
        buy_hold = float(candles[-1].close / candles[0].open - Decimal("1") - roundtrip)
        return ReplayMetrics(
            bars=len(candles),
            candidates=candidates,
            approved_decisions=approved,
            rejected_decisions=rejected,
            stale_pending_orders=stale_pending,
            trades=len(trades),
            net_return=(
                equity[-1] / float(self.config.initial_equity) - 1 if equity else 0.0
            ),
            buy_hold_return=buy_hold,
            max_drawdown=maximum_drawdown,
            profit_factor=profit_factor,
            annualized_sharpe=sharpe,
            win_rate=(
                sum(value > 0 for value in net_values) / len(net_values)
                if net_values
                else 0.0
            ),
            expectancy=mean(net_values) if net_values else 0.0,
            duplicate_signals=duplicate_signals,
            duplicate_fills=duplicate_fills,
            reconciliation_mismatches=reconciliation_mismatches,
            next_bar_fill_violations=next_bar_violations,
        )

    @staticmethod
    def _fill_id(signal_id: str, at: datetime, leg: str) -> str:
        payload = f"{signal_id}:{at.astimezone(UTC).isoformat()}:{leg}".encode()
        return "fill_" + hashlib.sha256(payload).hexdigest()[:24]

    @staticmethod
    def _trace_digest(
        trades: list[ReplayTrade],
        metrics: ReplayMetrics,
    ) -> str:
        import json

        payload = {
            "metrics": metrics.to_payload(),
            "trades": [trade.to_payload() for trade in trades],
        }
        encoded = json.dumps(
            payload,
            allow_nan=False,
            default=str,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()
