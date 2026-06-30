---
title: "Scalping Engine"
type: entity
tags: [scalping, cpu-only, low-latency, momentum, rejection, fast-path]
created: 2026-07-01
updated: 2026-07-01
status: active
---

# Scalping Engine

Ultra-fast CPU-only trading decision engine. Target: <5ms logic, no LLM/VLM/RAG, deterministic output.

## Definition

A pure-computation decision engine that makes trading decisions using only the last 5 candles, current price, and basic market structure. All logic is CPU-only with zero I/O, zero network calls, and zero model inference. Designed for sub-second scalping where latency is the primary constraint.

## Intuition

When a candle closes, the scalping engine:
1. **Classifies volatility** from recent candle ranges (high/medium/low)
2. **Detects momentum breakout** — current candle breaks previous range with conviction
3. **Detects rejection** — long wicks at support/resistance levels
4. **Checks trend alignment** — recent price action matches the broader trend
5. **Detects volume spikes** — current volume > 1.5x average
6. **Fuses signals** into a single BUY/SELL/HOLD decision with confidence score

## Output Format

Strict JSON — no explanations, no extra text:

```json
{
  "action": "BUY | SELL | HOLD",
  "confidence": 0-100,
  "size_pct": 0-5,
  "reason": "MOMENTUM_BREAKOUT"
}
```

## Signal Detection

### Momentum Breakout → BUY/SELL
- Current candle closes above/below previous candle's range
- Candle body ratio > 50% (strong conviction)
- Price change > 0.3% (breakout threshold)
- Also detects 3-candle acceleration patterns

### Sharp Rejection → BUY/SELL
- Long wick (>60% of range) at support/resistance level
- Pin bar pattern (body < 20% of range)
- Bullish rejection = long lower wick at support
- Bearish rejection = long upper wick at resistance

### Trend Alignment
- Checks if last 3 candles align with the broader trend direction
- Adds +15 confidence when aligned, -15 when conflicting

### Volume Confirmation
- Current volume > 1.5x average of last 4 candles
- Multiplies signal score by 1.2x

## Signal Fusion

Score-based fusion with thresholds:

| Signal | Score Impact |
|--------|:------------:|
| Momentum breakout | ±30 |
| Rejection at S/R | ±25 |
| Trend aligned | ±15 |
| Volume spike | ×1.2 multiplier |
| High volatility | Size reduced 50% |
| Low volatility | HOLD (no trade) |
| Conflicting signals | HOLD |

**Threshold:** confidence must exceed 50 (configurable via `CONFIDENCE_THRESHOLD`)

**Position sizing:** `size = (confidence/100) × max_size`, reduced by 50% in high volatility. Clamped to [0.5%, 5%].

## Volatility Classification

| Level | ATR/Price | Behavior |
|-------|:---------:|----------|
| High | >2% | Reduce size 50%, widen stops |
| Medium | 0.5-2% | Normal trading |
| Low | <0.5% | HOLD — no trade (insufficient movement) |

## Latency Profile

| Stage | Time |
|-------|:----:|
| Input parsing | <0.1ms |
| Volatility classification | <0.1ms |
| Momentum detection | <0.5ms |
| Rejection detection | <0.5ms |
| Trend alignment | <0.1ms |
| Volume spike | <0.1ms |
| Signal fusion | <0.1ms |
| **Total** | **<2ms** |

## Pipeline Integration

The scalping engine is integrated into the LangGraph pipeline as a **fast-path node**:

```
feature_engine → scalping_node
                        │
                ┌───────┴───────┐
                │ conf > 80%?   │
                ▼ YES           ▼ NO
          risk_gate       [VLM → RAG → swarm] → risk_gate
```

- **Fast path** (<5ms): scalping confidence > threshold → skip VLM/RAG/swarm
- **Heavy path** (~500ms): scalping confidence ≤ threshold → full agent analysis

The threshold is configurable via:
- `SCALPING_CONFIDENCE_THRESHOLD` env var (default: 80)
- `TradingState.scalping_confidence_threshold` (per-pipeline override)

## NATS Events

When the scalping fast-path triggers, the bridge publishes:
- `ScalpDecisionEvent` to `signals.scalp.<symbol>` — with action/confidence/size/reason/latency
- `TradeSignalEvent` to `signals.raw.scalping_engine.<symbol>` — mapped to standard signal format

## Source File

`orchestrator/scalping_engine.py`

## Related

- [[multi-agent-pipeline]] — dual-path architecture that uses the scalping engine
- [[nats-langgraph-bridge]] — publishes scalping events to NATS
- [[nats-event-system]] — ScalpDecisionEvent and ScalpExecutedEvent event types
- [[feature-engine]] — provides volatility/trend inputs to scalping node
- [[risk-engine]] — risk gate that validates scalping signals before execution
