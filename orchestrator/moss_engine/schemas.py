"""Strongly-typed data archetypes for the Moss Signal Engine ecosystem.

Every dataclass is designed for JSON serialization over NATS JetStream.
Repository Integration Contract:
  - MOSS v1.0.28 pillar payloads → PillarOutput, CompositeSignal
  - Skills_Registry metadata → SkillMeta, SkillState
  - Krypt-Trader events → WhaleEvent, MomentumScanResult, FadeSignal, ReconciliationReport
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ── Version constant pinned to MOSS Trading Skills Factory v1.0.28 ──
MOSS_FACTORY_VERSION: str = "1.0.28"


# ══════════════════════════════════════════════════════════════════════
# SECTION A:  SkillRegistry data archetypes (Skills_Registry.git pattern)
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SkillMeta:
    """Immutable metadata for a registered skill/brain.

    Mirrors Skills_Registry.git schema: name, version, status,
    execution metrics, and operational flags.
    """
    name: str
    version: str = MOSS_FACTORY_VERSION
    status: str = "idle"                     # idle | active | paused | error | killed
    register_ts: int = field(default_factory=lambda: int(time.time() * 1000))
    description: str = ""
    tags: Tuple[str, ...] = ()
    # Execution metrics (mutable via SkillState)
    total_invocations: int = 0
    total_errors: int = 0
    last_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    # Operational flags
    hot_swap_enabled: bool = True
    kill_switch_armed: bool = False
    max_daily_loss_pct: float = 5.0
    max_take_profit_pct: float = 50.0


@dataclass
class SkillState:
    """Mutable runtime state for a registered skill.

    Persisted inside SkillRegistry._states[name].
    """
    status: str = "idle"
    total_invocations: int = 0
    total_errors: int = 0
    last_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    cumulative_pnl: float = 0.0
    daily_pnl: float = 0.0
    daily_trailing_high: float = 0.0
    daily_drawdown_pct: float = 0.0
    last_execution_ts: int = 0
    consecutive_losses: int = 0
    kill_switch_active: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ══════════════════════════════════════════════════════════════════════
# SECTION B:  MossCompositeEngine 5-pillar archetypes (MOSS v1.0.28)
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PillarOutput:
    """Normalized output of one MOSS 5-pillar signal pillar.

    Each pillar normalizes to a score in [-1.0, +1.0] and a confidence
    in [0.0, 1.0], with a weight for composite blending.
    """
    pillar_name: str                   # trend | momentum | mean_reversion | volume | volatility
    score: float                       # -1.0 .. +1.0
    confidence: float                  # 0.0 .. 1.0
    weight: float                      # blend weight (sums to 1.0 across pillars)
    raw_values: Dict[str, float] = field(default_factory=dict)
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CompositeSignal:
    """Aggregated 5-pillar composite signal from MossCompositeEngine.

    Designed for NATS JetStream serialization on `signals.raw`.
    """
    symbol: str
    composite_score: float             # weighted sum of pillar scores
    composite_confidence: float        # weighted average confidence
    direction: int                      # +1 buy, -1 sell, 0 hold
    pillars: List[PillarOutput] = field(default_factory=list)
    contrarian_fade_invoked: bool = False
    whale_skew: float = 0.0
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> bytes:
        data = asdict(self)
        data["pillars"] = [p.to_dict() if hasattr(p, "to_dict") else p for p in self.pillars]
        return json.dumps(data, default=str).encode()


# ══════════════════════════════════════════════════════════════════════
# MODULE 1:  KryptCryptoCore archetypes (Krypt-Trader.git pattern)
# ══════════════════════════════════════════════════════════════════════

@dataclass
class WhaleEvent:
    """A single large-taker-trade event from Binance WebSocket aggTrades.

    Krypt-Trader: Whale Tracker / Order Flow Imbalance (OFI) pattern.
    Trades exceeding USD 100,000 are flagged as urgent directional vectors.
    """
    symbol: str
    price: float
    quantity: float
    notional_usd: float
    side: str                           # "buy" | "sell"
    is_urgent: bool                     # notional_usd >= 100_000
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MomentumScanResult:
    """15-Minute Crypto Momentum Cluster Scanner result.

    Krypt-Trader: 15m momentum scanner — triggers when current bar
    volume > 2.0x rolling 20-period average + expansionary price velocity.
    """
    symbol: str
    timeframe: str                      # "15m"
    current_volume: float
    avg_volume_20: float
    volume_ratio: float                 # current / avg_20
    price_velocity: float               # (close - open) / open
    momentum_triggered: bool            # volume_ratio >= 2.0 and |price_velocity| > threshold
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FadeSignal:
    """Contrarian Fade Mode signal — crowd-fading heuristic.

    Krypt-Trader: When RSI hits extreme boundaries AND price breaches
    outer Bollinger Bands, invert the composite direction to fade the crowd.
    """
    symbol: str
    rsi_extreme: bool                   # RSI < 20 or RSI > 80
    bb_breach: bool                     # price beyond outer BB
    original_direction: int             # +1 or -1
    faded_direction: int                # -original_direction
    fade_confidence: float              # 0.0 .. 1.0
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ══════════════════════════════════════════════════════════════════════
# MODULE 3:  Safety / Disaster Recovery archetypes (Krypt-Trader.git)
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ReconciliationReport:
    """Boot-time book reconciliation result.

    Krypt-Trader: reconcile_ledger_with_exchange() compares internal DB
    cache against exchange account snapshot, generating fixing trades if
    orphan states are detected after a network disconnect.
    """
    matched_positions: int
    orphan_positions: int
    generated_fixing_trades: int
    discrepancies: List[Dict[str, Any]] = field(default_factory=list)
    reconciliation_ts: int = field(default_factory=lambda: int(time.time() * 1000))
    success: bool = True

    def to_json(self) -> bytes:
        return json.dumps(asdict(self), default=str).encode()


@dataclass
class GuardrailStatus:
    """Hardware guardrails + master kill-switch status.

    Krypt-Trader: hard daily trailing stop-loss, rigid max take-profit,
    and boolean master kill-switch that zero-initializes all outputs.
    """
    master_kill_switch: bool = False
    daily_pnl: float = 0.0
    daily_trailing_high: float = 0.0
    daily_drawdown_pct: float = 0.0
    max_daily_loss_pct: float = 5.0
    max_take_profit_pct: float = 50.0
    consecutive_losses: int = 0
    max_consecutive_losses: int = 4
    is_triggered: bool = False          # True if any guardrail breached
    trigger_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ══════════════════════════════════════════════════════════════════════
# SECTION D:  ReflectiveEvolutionLoop archetypes (MOSS 7 Principles)
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ReflectionVerdict:
    """Evaluation of a single execution segment by the MOSS 7 Reflection Principles.

    Principles:
      1. Outcome Accountability — own the segment's PnL
      2. Causal Attribution — identify which parameters drove results
      3. Regime Awareness — was the segment in a known market regime?
      4. Risk Proportionality — was drawdown proportional to signal confidence?
      5. Signal Coherence — were pillar scores internally consistent?
      6. Execution Fidelity — did fills match signals?
      7. Adaptive Capacity — is the parameter space still explorable?
    """
    segment_id: str
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    principle_scores: Dict[str, float] = field(default_factory=dict)
    underperforming_params: List[str] = field(default_factory=list)
    regime_at_segment: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvolutionReport:
    """Result of a weekly evolutionary reflection cycle.

    Applies tactical micro-adjustments to parameters, bounded to ±30%
    of original base values.  Core risk variables (stop-loss, max position)
    are frozen and never adjusted.
    """
    cycle_id: str
    segment_count: int
    overall_sharpe: float
    overall_max_drawdown: float
    overall_win_rate: float
    verdicts: List[ReflectionVerdict] = field(default_factory=list)
    adjusted_parameters: Dict[str, float] = field(default_factory=dict)
    frozen_parameters: Dict[str, float] = field(default_factory=dict)
    original_parameters: Dict[str, float] = field(default_factory=dict)
    evolution_ts: int = field(default_factory=lambda: int(time.time() * 1000))

    def to_json(self) -> bytes:
        data = asdict(self)
        data["verdicts"] = [v.to_dict() if hasattr(v, "to_dict") else v for v in self.verdicts]
        return json.dumps(data, default=str).encode()
