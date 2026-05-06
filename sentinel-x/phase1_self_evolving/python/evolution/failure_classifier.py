"""
Autonomous Failure Classifier — Phase 1 Self-Evolving Core.

Detects WHY a trade failed without human intervention:
  - bad_prediction:  signal was directionally wrong
  - regime_mismatch: correct signal, wrong market environment
  - bad_execution:   signal correct but slippage/timing killed PnL
  - model_drift:     historically accurate signal now degraded
  - data_contamination: outlier/stale data corrupted signal chain
  - shallow_reasoning: LLM used vague language without data backing
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from core.nim_client import nim_json

log = logging.getLogger(__name__)


class FailureType(str, Enum):
    BAD_PREDICTION       = "bad_prediction"
    REGIME_MISMATCH      = "regime_mismatch"
    BAD_EXECUTION        = "bad_execution"
    MODEL_DRIFT          = "model_drift"
    DATA_CONTAMINATION   = "data_contamination"
    SHALLOW_REASONING    = "shallow_reasoning"
    UNKNOWN              = "unknown"


@dataclass
class FailureDiagnosis:
    failure_type:       FailureType
    confidence:         float          # [0, 1]
    root_cause:         str
    affected_agents:    list[str]
    severity:           float          # [0, 1] — how bad was this failure?
    recommended_action: str


_CLASSIFIER_SYSTEM = """You are a trading system failure analyst. Given trade metadata and outcome,
classify the failure with surgical precision. Return JSON:
{
  "failure_type": "bad_prediction"|"regime_mismatch"|"bad_execution"|"model_drift"|"data_contamination"|"shallow_reasoning",
  "confidence": <0-1>,
  "root_cause": "<1 sentence — specific, data-driven>",
  "affected_agents": ["<agent>", ...],
  "severity": <0-1>,
  "recommended_action": "<specific remediation>"
}

Classification rules:
- bad_prediction:      pnl < -1%, signal was in wrong direction, no regime mismatch
- regime_mismatch:     signal was correct in previous regime but current regime changed
- bad_execution:       entry signal was correct but fill price / timing caused the loss
- model_drift:         PSI > 0.15 on any feature OR win rate dropped > 15% in last 20 trades
- data_contamination:  outlier in input (volume spike > 10x, price gap > 5%)
- shallow_reasoning:   agent argument had no specific numbers/data points"""


async def classify_failure(
    trade_metadata: dict[str, Any],
    outcome: dict[str, Any],
    recent_win_rates: dict[str, float],
    psi_scores: dict[str, float],
) -> FailureDiagnosis:
    """Classify a trade failure without human intervention."""

    # Fast-path heuristics before calling NIM (reduces GPU cost)
    fast_type = _fast_classify(outcome, recent_win_rates, psi_scores)

    context = (
        f"Trade: {trade_metadata.get('symbol')} {trade_metadata.get('side')}\n"
        f"Entry: {trade_metadata.get('entry_price'):.4f} → "
        f"Exit: {outcome.get('exit_price', 0):.4f}\n"
        f"PnL: {outcome.get('pnl_pct', 0):.2%}\n"
        f"Slippage: {outcome.get('slippage_bps', 0):.1f} bps\n"
        f"Regime at entry: {trade_metadata.get('regime', 'unknown')}\n"
        f"Bull score: {trade_metadata.get('bull_score', 0):.2f} | "
        f"Bear score: {trade_metadata.get('bear_score', 0):.2f}\n"
        f"Confidence at entry: {trade_metadata.get('confidence_pct', 0):.1f}%\n"
        f"Recent win rates: {recent_win_rates}\n"
        f"PSI scores: {psi_scores}\n"
        f"Fast-path hint: {fast_type.value}"
    )

    result = await nim_json(
        [
            {"role": "system", "content": _CLASSIFIER_SYSTEM},
            {"role": "user", "content": context},
        ],
        temperature=0.05,
    )

    return FailureDiagnosis(
        failure_type=FailureType(result.get("failure_type", "unknown")),
        confidence=float(result.get("confidence", 0.5)),
        root_cause=result.get("root_cause", ""),
        affected_agents=result.get("affected_agents", []),
        severity=float(result.get("severity", 0.5)),
        recommended_action=result.get("recommended_action", ""),
    )


def _fast_classify(
    outcome: dict,
    win_rates: dict[str, float],
    psi_scores: dict[str, float],
) -> FailureType:
    """Sub-millisecond heuristic classifier — no LLM needed for obvious cases."""
    pnl = outcome.get("pnl_pct", 0)
    slippage = outcome.get("slippage_bps", 0)
    volume_ratio = outcome.get("volume_ratio", 1.0)

    if volume_ratio > 10.0:
        return FailureType.DATA_CONTAMINATION
    if max(psi_scores.values(), default=0) > 0.15:
        return FailureType.MODEL_DRIFT
    if slippage > 50 and pnl < 0:
        return FailureType.BAD_EXECUTION
    avg_wr = sum(win_rates.values()) / max(len(win_rates), 1)
    if avg_wr < 0.40:
        return FailureType.MODEL_DRIFT
    return FailureType.UNKNOWN
