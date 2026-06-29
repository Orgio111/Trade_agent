---
title: "Backtesting Pipeline"
type: concept
tags: [backtesting, pipeline, historical-data, validation]
created: 2026-06-30
updated: 2026-06-30
status: stable
---

# Backtesting Pipeline

> This page is an alias for [[brain-backtest-infrastructure]]. The backtesting pipeline IS the brain backtest infrastructure — they are the same system.

See [[brain-backtest-infrastructure]] for the full architecture, metrics, and usage.

## How we use it

- `orchestrator/brain_backtest.py` — main backtest engine
- CLI: `python -m orchestrator.brain_backtest --symbol BTCUSDT --days 90`
- Validates brain weight changes before deployment
- Referenced by: [[signal-aggregation-logic]], [[brain-ecosystem]]

## Related

- [[brain-backtest-infrastructure]] — canonical page
- [[brain-ecosystem]] — brain weights being backtested
- [[continual-learning-pipeline]] — uses backtest for model validation

## Sources

- Internal code: `orchestrator/brain_backtest.py`
