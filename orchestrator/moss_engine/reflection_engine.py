"""Reflective Evolution Loop — MOSS weekly reflection.

MOSS 7 Principles:
  1. Outcome Accountability — own every execution segment PnL
  2. Causal Attribution — identify which params drove outcomes
  3. Regime Awareness — was the segment in an expected regime?
  4. Risk Proportionality — was drawdown proportional to signal confidence?
  5. Signal Coherence — were pillar scores internally consistent?
  6. Execution Fidelity — did fills match signals?
  7. Adaptive Capacity — is the parameter space still explorable?

Reflection cycle:
  - Segmentation: split history into epochs (daily, weekly, ...)
  - Evaluation: Sharpe, max drawdown, win rate per epoch
  - Verdict: apply MOSS 7 principles → detect underperforming params
  - Evolution: apply tactical micro-adjustments (±30% bounds)

Dimensions for adjustment:
  - Pillar blend weights
  - SuperTrend multiplier, RSI thresholds, OBV lookback

Frozen dimensions (core risk variables)
  - Stop-loss, max position/notional, kill-switch
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from collections import defaultdict
import numpy as np

from .schemas import (
    CompositeSignal,
    EvolutionReport,
    ReflectionVerdict,
    SkillMeta,
    SkillState,
)
from .skill_registry import SkillRegistry

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────
BASE_PARAMS: Dict[str, float] = {
    "trend_weight":          0.3,
    "momentum_weight":       0.25,
    "mean_reversion_weight": 0.2,
    "volume_weight":         0.15,
    "volatility_weight":     0.1,
    "supert_multiplier":     2.0,
    "volume_ano_ratio":      2.0,
    "rsi_overbought":        70.0,
    "rsi_oversold":          30.0,
    "atr_breakout_ratio":    1.3,
}

FROZEN_PARAMS: Tuple[str, ...] = (
    "sl_atr_mult",
    "max_position_cap",
    "max_notional_pct",
)

ADJUSTMENT_BOUNDS: Dict[str, Tuple[float, float]] = {
    # name: (min, max) for ±30% micro-adjustment
    "trend_weight":          (0.21, 0.39),      # ±30%
    "momentum_weight":       (0.175, 0.325),
    "mean_reversion_weight": (0.14, 0.26),
    "volume_weight":         (0.105, 0.195),
    "volatility_weight":     (0.07, 0.13),
    "supert_multiplier":     (1.4, 2.6),
    "volume_ano_ratio":      (1.4, 2.6),
    "rsi_overbought":        (63.0, 83.7),
    "rsi_oversold":          (17.1, 41.1),
    "atr_breakout_ratio":    (0.91, 1.69),
}

# Safe normalize: clamp adjustment
CLAMP_ADJ: Dict[str, Callable[[float, float], float]] = {
    name: (lambda val, base: max(min_val, min(max_val, val)))
    for name, (min_val, max_val) in ADJUSTMENT_BOUNDS.items()
}

REGIME_THRESHOLDS = {
    "sharpe_ratio":          0.5,       # > 0.5 is good
    "max_drawdown_pct":     5.0,       # < 5% is acceptable
    "win_rate":             0.5,       # > 0.5 is winning
    "signal_coherence":     0.8,       # std < 20%
}


class ReflectiveEvolutionLoop:
    """MOSS weekly reflection & evolution engine.

    Applies MOSS 7 Principles, assesses Sharpe/MaxDD/WinRate,
    and evolves parameters within ±30% bounds.
    """

    def __init__(self, registry: SkillRegistry, lookback_epochs: str = "1w") -> None:
        self._registry = registry
        self.segment_data: Dict[str, List[Any]] = {}  # symbol_interval → [CompositeSignal]
        self.evolution_log: deque = deque(maxlen=100)
        self._version = "1.0.0"

    # ── Segmentation & Buffering ─────────

    def buffer_signal(self, signal: CompositeSignal) -> None:
        """Ingest a new signal into epoch segmentation."""
        key = f"{signal.symbol}_{signal.interval}"
        if key not in self.segment_data:
            self.segment_data[key] = []
        self.segment_data[key].append(signal)

    def _segment_epoch(self, signals: List[CompositeSignal]) -> List[List[CompositeSignal]]:
        """Split epoch buffer into segments."""
        # Split into daily segments
        if not signals:
            return []
        signals_sorted = sorted(signals, key=lambda s: s.timestamp_ms)
        segments = []
        segment_start = signals_sorted[0].timestamp_ms
        segment_key = 0
        daily_ms = 86400_000

        daily_segments: Dict[int, List] = defaultdict(list)
        for signal in signals_sorted:
            day_id = (signal.timestamp_ms - segment_start) // daily_ms
            daily_segments[day_id].append(signal)

        for day_signals in daily_segments.values():
            segments.append(day_signals)

        return segments

    # ── Evaluation Metrics ─────────────

    def _compute_sharpe_ratio(self, pnls: List[float]) -> float:
        """Sharpe ratio from PnL trajectory."""
        returns = np.diff(pnls) / (np.array(pnls)[:-1] + 1e-8)
        return float(np.mean(returns) / np.std(returns)) if np.std(returns) > 1e-8 else 0.0

    def _compute_max_drawdown(self, pnls: List[float]) -> float:
        """Max drawdown as percentage."""
        if not pnls:
            return 0.0
        max_pnl = np.maximum.accumulate(pnls)
        denominator = max_pnl.copy()
        denominator[denominator == 0] = 1e-8
        dd = (max_pnl - pnls) / denominator
        return float(np.max(dd) * 100.0)

    def _compute_signal_coherence(self, segments: List[CompositeSignal]) -> float:
        """Coherence = stddev of pillar scores normalized."""
        pillar_scores = defaultdict(list)
        for seg in segments:
            for pillar in seg.pillars:
                pillar_scores[pillar.pillar_name].append(pillar.score)
        stdev_avg = np.mean([np.std(scores) for scores in pillar_scores.values()])
        return float(1.0 - np.tanh(stdev_avg / 0.2))

    # ── MOSS 7 Principles Verdict ──────

    def _apply_reflection_principles(
        self,
        segment_signals: List[CompositeSignal],
        segment_pnls: List[float],
        params_at_segment: Dict[str, float]
    ) -> ReflectionVerdict:
        """Evaluate one segment via the MOSS 7 Principles."""
        
        # Metrics
        sharpe = self._compute_sharpe_ratio(segment_pnls)
        max_dd = self._compute_max_drawdown(segment_pnls)
        win_rate = len([p for p in segment_pnls if p > 0]) / len(segment_pnls) if segment_pnls else 0.0
        signal_coherence = self._compute_signal_coherence(segment_signals)

        # MOSS 7 verdicts
        principle_scores = {}
        underperforming_params: List[str] = []
        
        # Principle 1: Outcome Accountability
        retrospective_pnl = sum(segment_pnls)
        principle_scores["accountability"] = float(np.tanh(retrospective_pnl / 100.0))
        
        # Principle 2: Causal Attribution
        pnls_array = np.array(segment_pnls)
        sharpness = np.abs(np.gradient(pnls_array)).mean()
        principle_scores["attribution"] = float(np.tanh(sharpness / 0.5))
        
        # Principle 3: Regime was it in?
        regime_at_segment = self._detect_regime(segment_signals)
        regime_key = regime_at_segment.get("name", "unknown").lower() if regime_at_segment else "unknown"
        principle_scores["regime_aware"] = float(0.5 if regime_key == "calm" else 0.9 if regime_key == "trend" else 0.7)
        
        # Principle 4: Drawdown proportional to signal confidence?
        avg_conf = np.mean([sig.composite_confidence for sig in segment_signals])
        conf_dd_proportional = float(np.tanh(avg_conf * (1 - max_dd / REGIME_THRESHOLDS["max_drawdown_pct"])))
        principle_scores["risk_proportional"] = conf_dd_proportional
        
        # Principle 5: Signal Coherence
        principle_scores["coherence"] = signal_coherence
        
        # Principle 6: Execution Fidelity
        # Assume ideal execution (we'd inject trade logs in full version)
        principle_scores["execution_fidelity"] = 1.0
        
        # Principle 7: Adaptive Capacity
        frozen_parms = [name for name in FROZEN_PARAMS if name in params_at_segment]
        principle_scores["adaptive_capacity"] = float(1.0 if not frozen_parms else 0.3)

        # Determine underperforming
        for p, threshold in REGIME_THRESHOLDS.items():
            if p.startswith("sharpe") and sharpe < threshold:
                underperforming_params.append("blend_weights")
            if p.startswith("max_drawdown") and max_dd > threshold:
                underperforming_params.extend(["supert_multiplier", "atr_breakout_ratio"])
            if p.startswith("win_rate") and win_rate < threshold:
                underperforming_params.extend(["momentum_weight", "rsi_threshold"])
        
        return ReflectionVerdict(
            segment_id=hex(hash(tuple(segment_signals[0].timestamp_ms for _ in range(10)))),
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_dd,
            win_rate=win_rate,
            principle_scores=principle_scores,
            underperforming_params=list(set(underperforming_params)),
            regime_at_segment=regime_key,
        )

    def _detect_regime(self, signals: List[CompositeSignal]) -> Optional[Dict[str, Any]]:
        """Simplistic regime detector."""
        closes = [s.pillars["mean_reversion"].raw_values["bb_sma"] for s in signals]
        vol = np.std(closes[-20:]) / np.mean(closes[-20:])
        
        if vol < 0.015:
            return {"name": "calm"}
        elif any(s.composite_score > 0.6 for s in signals):
            return {"name": "trend", "direction": +1}
        elif any(s.composite_score < -0.6 for s in signals):
            return {"name": "trend", "direction": -1}
        else:
            return {"name": "mean_revert"}

    # ── Evolution: Apply ±30% Micro-Adjustments ──

    def evolve_parameters(self) -> EvolutionReport:
        """Reflect on all segments, evolve parameters."""
        epoch_params = BASE_PARAMS.copy()
        verdicts = []
        adjusted = BASE_PARAMS.copy()
        original = BASE_PARAMS.copy()

        # Segment by asset_interval→epoch→segments
        for key, signals in self.segment_data.items():
            segments = self._segment_epoch(signals)
            for seg_id, segment in enumerate(segments):
                # N last signals → PnL trajectory placeholder
                segment_pnls = [np.random.uniform(-1, 1) for _ in range(max(1, len(segment) - 1))]
                verdict: ReflectionVerdict = self._apply_reflection_principles(
                    segment,
                    segment_pnls,
                    epoch_params
                )
                verdicts.append(verdict)
                
                # Adjust params for next epoch
                for param in verdict.underperforming_params:
                    if param in adjusted:
                        base = BASE_PARAMS[param]
                        val = adjusted[param]
                        min_adj, max_adj = ADJUSTMENT_BOUNDS[param]
                        adjusted_move = np.random.choice([-0.1, 0.1]) * base
                        val = CLAMP_ADJ[param](val + adjusted_move, base)
                        adjusted[param] = val

        # Repack report
        report = EvolutionReport(
            cycle_id=f"ev_{int(time.time())}",
            segment_count=len(verdicts),
            overall_sharpe=float(np.mean([v.sharpe_ratio for v in verdicts])),
            overall_max_drawdown=float(np.mean([v.max_drawdown_pct for v in verdicts])),
            overall_win_rate=float(np.mean([v.win_rate for v in verdicts])),
            verdicts=verdicts,
            adjusted_parameters={k: round(v, 6) for k, v in adjusted.items()},
            original_parameters=original,
            frozen_parameters={k: BASE_PARAMS[k] for k in FROZEN_PARAMS},
        )
        return report

    # ── Utility ───────────────────

    def inject_into_registry(self, registry: SkillRegistry) -> None:
        registry.register(
            name="moss_reflection",
            skill_cls_or_instance=self,
            version=self._version,
            description="MOSS weekly reflection & parameter evolution",
            tags=("moss", "evolution", "risk"),
        )

    @property
    def version(self) -> str:
        return self._version
