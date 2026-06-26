"""Brain #10: Polymarket Alpha Engine — 5-Formula Mathematical Pipeline.

Incorporates 5 core formulas derived from 72M trade analysis:
  1. Bayesian Evidence Tracking   (Bayesian LR shortcut)
  2. Longshot Bias Correction      (price < 0.10 → 57% floor; price > 0.80 → +2% lift)
  3. Expected Value & ROI Filter   (EV = prob×payout − cost; negative EV → skip)
  4. Quarter-Kelly Position Sizing  (0.25× Kelly, 5% hard cap)
  5. Nash Router                    (70% MAKER default; TAKER if edge > 20%; MAKER edge +1.12%)

All empirical constants are loaded from config/polymarket.yaml with safe defaults.
Weight in Go orchestrator: 0.05 (prediction-market signal, not primary).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from collections import deque
from dataclasses import dataclass, field
from math import exp, log
from pathlib import Path
from typing import Any

import numpy as np

from .base_brain import BaseBrain, BrainSignal

logger = logging.getLogger(__name__)

# ── Config Loading ──────────────────────────────────────────────

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
_CONFIG_FILE = _CONFIG_DIR / "polymarket.yaml"


def _load_polymarket_config() -> dict[str, Any]:
    """Load config/polymarket.yaml with graceful fallback to defaults."""
    if _CONFIG_FILE.exists():
        try:
            import yaml
            with _CONFIG_FILE.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                return data
        except Exception as exc:
            logger.warning(f"[polymarket_alpha] Config load failed: {exc}, using defaults")
    return {}


@dataclass
class _PolymarketDefaults:
    """Hardcoded empirical defaults from 72M trade analysis.
    Overridden by config/polymarket.yaml if present.
    """
    # Formula 2: Longshot Bias
    longshot_low_threshold: float = 0.10
    longshot_high_threshold: float = 0.80
    longshot_low_penalty: float = 0.57
    longshot_high_boost: float = 0.02

    # Formula 4: Quarter-Kelly
    kelly_quarter_factor: float = 0.25
    kelly_hard_cap: float = 0.05

    # Formula 5: Nash Router
    maker_default_probability: float = 0.70
    maker_edge_pct: float = 0.0112
    taker_threshold_pct: float = 0.20

    # Formula 1: Bayesian
    bayesian_jump_threshold: float = 0.20

    # Brain weight for Go orchestrator
    brain_weight: float = 0.05


# ── Data Payloads ────────────────────────────────────────────────

@dataclass
class EvidenceSnapshot:
    """Point-in-time evidence for a prediction market."""
    source: str
    headline: str
    impact: float          # -1.0 (strong NO) to +1.0 (strong YES)
    reliability: float     # 0.0 to 1.0
    timestamp_ms: int = 0


@dataclass
class MarketState:
    """Current state of a Polymarket contract."""
    question: str
    yes_price: float       # 0.0 to 1.0
    volume_24h: float = 0.0
    liquidity: float = 0.0
    open_interest: float = 0.0


@dataclass
class AlphaResult:
    """Full alpha pipeline output for a single market."""
    question: str
    posterior_yes: float
    adjusted_prob: float
    expected_value: float
    roi: float
    kelly_fraction: float
    kelly_capped: float
    route: str             # "MAKER" or "TAKER"
    maker_probability: float
    should_trade: bool
    evidence_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Formula Implementations ─────────────────────────────────────

class BayesianEvidenceTracker:
    """Formula 1: Bayesian Likelihood-Ratio shortcut.

    Instead of full Bayesian update:
        LR = Π reliability_i × (1 + |impact_i|)
        posterior = prior × LR / (prior × LR + (1 − prior))

    A sudden LR jump > bayesian_jump_threshold triggers re-evaluation.
    """

    def __init__(self, cfg: _PolymarketDefaults) -> None:
        self._jump_threshold = cfg.bayesian_jump_threshold

    def update(
        self,
        prior: float,
        evidence: list[EvidenceSnapshot],
    ) -> tuple[float, dict[str, Any]]:
        lr = 1.0
        for e in evidence:
            r = max(0.01, min(1.0, e.reliability))
            lr *= r * (1.0 + abs(e.impact))

        prev_posterior = prior
        numerator = prior * lr
        posterior = numerator / (numerator + (1.0 - prior))

        jump = abs(posterior - prev_posterior)
        meta: dict[str, Any] = {
            "likelihood_ratio": round(lr, 6),
            "evidence_count": len(evidence),
            "jump": round(jump, 6),
            "jump_triggered": jump > self._jump_threshold,
        }
        return float(np.clip(posterior, 0.001, 0.999)), meta


class LongshotBiasCorrector:
    """Formula 2: Empirical longshot bias correction.

    From 72M trades:
      - price < 0.10 → actual win rate = 57% of quoted price
      - price > 0.80 → actual win rate = quoted price + 2%
    """

    def __init__(self, cfg: _PolymarketDefaults) -> None:
        self._low_thresh = cfg.longshot_low_threshold
        self._high_thresh = cfg.longshot_high_threshold
        self._low_penalty = cfg.longshot_low_penalty
        self._high_boost = cfg.longshot_high_boost

    def correct(self, raw_prob: float) -> tuple[float, dict[str, Any]]:
        adjusted = raw_prob
        correction = "none"

        if raw_prob < self._low_thresh:
            adjusted = raw_prob * self._low_penalty
            correction = f"longshot_penalty (×{self._low_penalty})"
        elif raw_prob > self._high_thresh:
            adjusted = raw_prob + self._high_boost
            correction = f"favorite_boost (+{self._high_boost})"

        adjusted = float(np.clip(adjusted, 0.001, 0.999))
        meta: dict[str, Any] = {
            "raw_prob": round(raw_prob, 6),
            "adjusted_prob": round(adjusted, 6),
            "correction": correction,
        }
        return adjusted, meta


class EVROIFilter:
    """Formula 3: Expected Value & ROI filter.

    EV  = adjusted_prob × payout − cost
    ROI = EV / cost
    Trade only if EV > 0.
    """

    def evaluate(
        self,
        adjusted_prob: float,
        yes_price: float,
    ) -> tuple[float, float, dict[str, Any]]:
        payout = 1.0
        cost = yes_price
        ev = adjusted_prob * payout - cost
        roi = ev / cost if cost > 0.001 else 0.0
        meta: dict[str, Any] = {
            "payout": payout,
            "cost": round(cost, 6),
            "ev": round(ev, 6),
            "roi": round(roi, 6),
            "tradeable": ev > 0,
        }
        return float(ev), float(roi), meta


class QuarterKellySizer:
    """Formula 4: Quarter-Kelly position sizing.

    f_kelly = (b·p − q) / b   where b = profit_per_unit / loss_per_unit
    f_safe  = 0.25 × f_kelly, capped at kelly_hard_cap (default 5%).
    """

    def __init__(self, cfg: _PolymarketDefaults) -> None:
        self._quarter = cfg.kelly_quarter_factor
        self._cap = cfg.kelly_hard_cap

    def size(
        self,
        prob: float,
        yes_price: float,
    ) -> tuple[float, float, dict[str, Any]]:
        b = (1.0 - yes_price) / yes_price if yes_price > 0.001 else 0.0
        p = min(max(prob, 0.001), 0.999)
        q = 1.0 - p

        f_kelly = (b * p - q) / b if b > 0 else 0.0
        f_kelly = max(0.0, f_kelly)
        f_safe = min(self._cap, self._quarter * f_kelly)

        meta: dict[str, Any] = {
            "full_kelly": round(f_kelly, 6),
            "quarter_factor": self._quarter,
            "hard_cap": self._cap,
            "kelly_capped": round(f_safe, 6),
        }
        return f_kelly, float(f_safe), meta


class NashRouter:
    """Formula 5: Nash Equilibrium order routing.

    Default 70% MAKER → when edge > taker_threshold%, switch TAKER.
    MAKER edge = +1.12% (from empirical analysis).
    """

    def __init__(self, cfg: _PolymarketDefaults) -> None:
        self._maker_default = cfg.maker_default_probability
        self._maker_edge = cfg.maker_edge_pct
        self._taker_thresh = cfg.taker_threshold_pct

    def route(
        self,
        ev: float,
        roi: float,
    ) -> tuple[str, float, dict[str, Any]]:
        maker_prob = self._maker_default
        route = "MAKER"

        if roi > self._taker_thresh:
            route = "TAKER"
            maker_prob = 1.0 - self._maker_default

        # Apply MAKER edge bonus when routing MAKER
        effective_edge = self._maker_edge if route == "MAKER" else 0.0
        meta: dict[str, Any] = {
            "route": route,
            "maker_probability": round(maker_prob, 4),
            "effective_edge_pct": round(effective_edge * 100, 4),
            "taker_threshold_pct": round(self._taker_thresh * 100, 2),
            "roi": round(roi, 6),
        }
        return route, float(maker_prob), meta


# ── The Brain ────────────────────────────────────────────────────

class PolymarketBrain(BaseBrain):
    """Brain #10: Polymarket Alpha Engine — 5-Formula Mathematical Pipeline.

    Processes prediction-market contracts through Bayesian evidence tracking,
    longshot bias correction, EV/ROI filtering, Quarter-Kelly sizing, and
    Nash-equilibrium order routing.

    Push methods (for external data injection):
      - push_evidence(source, headline, impact, reliability)
      - push_market(question, yes_price, volume_24h, liquidity)
    """

    @property
    def brain_id(self) -> str:
        return "polymarket_alpha"

    def __init__(self) -> None:
        raw_cfg = _load_polymarket_config()
        self._cfg = _PolymarketDefaults(**{
            k: v for k, v in raw_cfg.items()
            if k in _PolymarketDefaults.__dataclass_fields__
        })

        self._bayesian = BayesianEvidenceTracker(self._cfg)
        self._longshot = LongshotBiasCorrector(self._cfg)
        self._ev_filter = EVROIFilter()
        self._kelly = QuarterKellySizer(self._cfg)
        self._router = NashRouter(self._cfg)

        self._evidence_buffer: deque[EvidenceSnapshot] = deque(maxlen=200)
        self._market_buffer: deque[MarketState] = deque(maxlen=100)
        self._result_buffer: deque[AlphaResult] = deque(maxlen=50)

        self._weight = self._cfg.brain_weight

    async def warmup(self) -> None:
        logger.info(
            f"[polymarket_alpha] Ready "
            f"(kelly_q={self._cfg.kelly_quarter_factor}, "
            f"kelly_cap={self._cfg.kelly_hard_cap}, "
            f"maker_edge={self._cfg.maker_edge_pct:.4f})"
        )

    async def compute_score(self, symbol: str) -> BrainSignal:
        if not self._market_buffer:
            return BrainSignal(
                brain_id=self.brain_id,
                symbol=symbol,
                score=0.0,
                confidence=0.1,
                weight=self._weight,
                direction=0,
                metadata={"status": "no_market_data"},
            )

        market = self._market_buffer[-1]
        result = self._run_pipeline(market)

        # Convert AlphaResult → BrainSignal
        # score: positive = tradeable YES edge, negative = tradeable NO edge
        # Scale by |EV| so stronger edges → stronger signals
        if result.should_trade:
            score = float(np.clip(result.expected_value * 2.0, -1.0, 1.0))
        else:
            score = 0.0

        direction = 1 if result.should_trade and result.adjusted_prob > 0.5 else (
            -1 if result.should_trade and result.adjusted_prob < 0.5 else 0
        )

        confidence = 0.3
        if result.should_trade and result.evidence_count >= 2:
            confidence = min(0.85, 0.4 + result.evidence_count * 0.08)
        elif result.should_trade:
            confidence = 0.35

        return BrainSignal(
            brain_id=self.brain_id,
            symbol=symbol,
            score=score,
            confidence=confidence,
            weight=self._weight,
            direction=direction,
            metadata={
                "posterior_yes": round(result.posterior_yes, 6),
                "adjusted_prob": round(result.adjusted_prob, 6),
                "ev": round(result.expected_value, 6),
                "roi": round(result.roi, 6),
                "kelly_capped": round(result.kelly_capped, 6),
                "route": result.route,
                "should_trade": result.should_trade,
                "evidence_count": result.evidence_count,
            },
        )

    def _run_pipeline(self, market: MarketState) -> AlphaResult:
        """Execute the full 5-formula pipeline on a single market."""
        evidence = [e for e in self._evidence_buffer]
        prior = market.yes_price

        # F1: Bayesian Evidence Tracking
        posterior, bayes_meta = self._bayesian.update(prior, evidence)

        # F2: Longshot Bias Correction
        adjusted, longshot_meta = self._longshot.correct(posterior)

        # F3: EV & ROI Filter
        ev, roi, ev_meta = self._ev_filter.evaluate(adjusted, market.yes_price)

        # F4: Quarter-Kelly Sizing (only if EV+)
        kelly_raw, kelly_capped, kelly_meta = self._kelly.size(adjusted, market.yes_price)

        # F5: Nash Router
        route, maker_prob, route_meta = self._router.route(ev, roi)

        should_trade = ev > 0

        result = AlphaResult(
            question=market.question,
            posterior_yes=round(posterior, 6),
            adjusted_prob=adjusted,
            expected_value=ev,
            roi=roi,
            kelly_fraction=kelly_raw,
            kelly_capped=kelly_capped,
            route=route,
            maker_probability=maker_prob,
            should_trade=should_trade,
            evidence_count=len(evidence),
            metadata={
                "bayesian": bayes_meta,
                "longshot": longshot_meta,
                "ev_roi": ev_meta,
                "kelly": kelly_meta,
                "router": route_meta,
            },
        )
        self._result_buffer.append(result)
        return result

    # ── Push Methods (external data injection) ──────────────────

    def push_evidence(
        self,
        source: str,
        headline: str,
        impact: float,
        reliability: float,
    ) -> None:
        """Push a new piece of evidence for Bayesian tracking."""
        import time
        self._evidence_buffer.append(EvidenceSnapshot(
            source=source,
            headline=headline,
            impact=float(np.clip(impact, -1.0, 1.0)),
            reliability=float(np.clip(reliability, 0.0, 1.0)),
            timestamp_ms=int(time.time() * 1000),
        ))

    def push_market(
        self,
        question: str,
        yes_price: float,
        volume_24h: float = 0.0,
        liquidity: float = 0.0,
    ) -> None:
        """Push current market state for the latest contract."""
        self._market_buffer.append(MarketState(
            question=question,
            yes_price=float(np.clip(yes_price, 0.001, 0.999)),
            volume_24h=volume_24h,
            liquidity=liquidity,
        ))

    # ── Demo / Standalone Mode ──────────────────────────────────

    def run_demo(self) -> list[AlphaResult]:
        """Run the 4-stage demo scenario (standalone, no NATS needed)."""
        # Stage 1: Baseline — Fed Rate Cut, initial odds 35%
        self.push_market("Fed Rate Cut July 2026", yes_price=0.35)
        r1 = self._run_pipeline(self._market_buffer[-1])

        # Stage 2: Weak jobs report evidence
        self.push_evidence("BLS", "Non-farm payrolls +89K (miss)", impact=0.60, reliability=0.80)
        r2 = self._run_pipeline(self._market_buffer[-1])

        # Stage 3: Whale inflow — large YES bets
        self.push_evidence("Polymarket Order Flow", "Large YES bets $500K+", impact=0.35, reliability=0.65)
        r3 = self._run_pipeline(self._market_buffer[-1])

        # Stage 4: Emergency BJ news
        self.push_evidence("Reuters", "Emergency BJ meeting scheduled", impact=0.90, reliability=0.92)
        r4 = self._run_pipeline(self._market_buffer[-1])

        results = [r1, r2, r3, r4]

        print("\n╔══════════════════════════════════════════════════════════════╗")
        print("║   POLYMARKET ALPHA ENGINE — 5-Formula Pipeline v1.0       ║")
        print("╠══════════════════════════════════════════════════════════════╣")
        print("║  F1: Bayesian Evidence   F2: Longshot Bias Correction    ║")
        print("║  F3: EV & ROI Filter       F4: Quarter-Kelly (0.25, 5%)   ║")
        print("║  F5: Nash Router (MAKER 70%, TAKER if edge > 20%)         ║")
        print("╚══════════════════════════════════════════════════════════════╝\n")

        headers = "Stage | Posterior | Adj Prob |  Net EV  |   ROI   | Kelly Cap | Route"
        print(headers)
        print("-" * len(headers))

        stage_names = ["Baseline", "Weak jobs", "Whale inflow", "Emergency news"]
        for name, r in zip(stage_names, results):
            print(
                f"{name:<16} | {r.posterior_yes:.6f} | "
                f"{r.adjusted_prob:.6f} | {r.expected_value:+.6f} | "
                f"{r.roi:+.4f} | {r.kelly_capped:.6f} | {r.route}"
            )

        print()
        return results


# ── Standalone Runner ────────────────────────────────────────────

if __name__ == "__main__":
    brain = PolymarketBrain()
    brain.run_demo()
