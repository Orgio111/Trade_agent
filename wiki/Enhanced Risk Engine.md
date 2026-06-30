# Enhanced Risk Engine

## Overview
Hard filter that sits between AutoGen consensus and execution. Can override BUY/SELL → HOLD.

## Components

### 1. Minimum Confidence
```python
if signal.confidence < min_confidence (default 0.6):
    reject("Low confidence")
```
**HOLD always passes** — no confidence check.

### 2. Daily Drawdown Limit
```python
if state.daily_pnl_pct <= -max_daily_drawdown (default -3%):
    reject("Daily drawdown exceeded")
```

### 3. Max Concurrent Positions
```python
if state.open_positions >= max_concurrent (default 3):
    reject("Max positions")
```

### 4. Circuit Breaker (Kill Streak)
```python
if state.loss_streak >= kill_streak (default 5):
    activate_circuit_breaker()  # 5min cooldown
    reject("Circuit breaker triggered")
```
Auto-resets after 5 minutes.

### 5. ATR Volatility Filter
```python
atr_pct = indicators.get("atr_pct", 0)
atr_mean = indicators.get("atr_mean_pct", atr_pct)
if atr_pct >= atr_mean * atr_threshold_mult (default 2.5):
    reject("ATR vol filter")
```
**Uses `>=`** — exactly at threshold = blocked.

### 6. Kelly Criterion Position Sizing
```python
# After N trades with win_rate, avg_win, avg_loss
kelly_f = win_rate - (1 - win_rate) / (avg_win / avg_loss)
size_pct = min(kelly_f * equity, kelly_max_pct)  # capped at 5%
```
Updated on each trade via `on_trade_result(pnl_pct, is_win)`.

## RiskDecision Output
```python
RiskDecision(
    allow: bool,
    reason: str,
    adjusted_signal: Signal | None,
    max_size: float  # Kelly-capped
)
```

## Status API
```python
engine.get_status() -> {
    "circuit_breaker_active": bool,
    "kelly_stats": {...},
    "circuit_breaker_until": datetime | None
}
```

## Config
```yaml
risk:
  max_position_pct: 0.05
  max_daily_drawdown: 0.03
  kill_streak: 5
  min_confidence: 0.6
  atr_filter:
    enabled: true
    atr_threshold_mult: 2.5
```

## Related
- [[AutoGen Trading Pipeline]]
- [[Model Loader VRAM Strategy]]
- [[Backtest Engine]]