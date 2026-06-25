---
title: FinRL Kelly Brain
type: entity
tags: [brain, rl, ppo, kelly, position-sizing]
created: 2026-06-26
updated: 2026-06-26
weight: 0.10
brain_id: finrl_kelly
source_file: finrl_brain.py
status: active
---

# FinRL Kelly Brain

## Overview

**Brain #6** — Position sizing brain. Uses PPO reinforcement learning (stable_baselines3) for sizing conviction, with Kelly Criterion as fallback. **Critical:** This brain's score = sizing amplification (0..+1), NOT direction. It amplifies or dampens the directional consensus from other brains.

## Architecture

```
push_trade_result → trade_history (deque 500) → PPO agent ──── 50% ──┐
                                                 Kelly Criterion ── 50% ──┤→ blend → score
                                                 ↘ (no PPO): 100% half-Kelly
```

## Parameters

| Parameter | Default | Env var |
|-----------|---------|---------|
| max_risk_pct | 0.015 (1.5%) | `MAX_RISK_PCT` |
| risk_pct_per_trade | 0.015 | `RISK_PCT_PER_TRADE` |
| model_dir | `models/finrl/` | `FINRL_MODEL_DIR` |
| model_path | auto-discover | `FINRL_MODEL_PATH` |
| auto_train_threshold | 50 trades | — |
| trade_history_maxlen | 500 | — |
| PPO/Kelly blend | 50/50 | — |

## Kelly formula

```
f* = (p * b - q) / b
where: p = win_rate, q = 1-p, b = avg_win / avg_loss
Half-Kelly applied: f* / 2
```

## Outputs

- `score`: ∈ [0, +1] (sizing conviction — 0 = no position, 1 = full conviction)
- `confidence`: ∈ [0, 1]
- `metadata`: kelly_fraction, ppo_size, trade_count, win_rate

## Fallback chain

1. **PPO + half-Kelly blend** (50/50, model loaded + ≥50 trade history)
2. **Half-Kelly only** (PPO unavailable)
3. **Default risk** (no trade history → `risk_pct_per_trade`)

## Fail modes

- `stable_baselines3` not installed → Kelly only
- PPO checkpoint corrupt → warning, retrain or Kelly only
- < 50 trades in history → no auto-train, Kelly only
- Win rate = 0 or avg_loss = 0 → Kelly undefined → default risk

## Edge & known weaknesses

- **Edge:** RL sizing adapts to non-stationary markets (Kelly is stationary assumption)
- **Edge:** Half-Kelly is conservative — prevents over-betting
- **Weakness:** PPO needs 50+ trades to auto-train — cold start problem
- **Weakness:** Corrupted checkpoint is a known issue (audit warnings)
- **Weakness:** 50/50 blend is hardcoded — should be regime-adaptive
- **10× opportunity:** Online PPO training after each trade result (already partially implemented); make blend ratio regime-aware (more Kelly in ranging, more PPO in trending)

## Related

- [[freqai-brain]] — provides directional signal that FinRL sizes
- [[timesfm-brain]] — primary directional input for FinRL amplification
- [[llm-regime-brain]] — regime should modulate PPO/Kelly blend ratio

## Sources

- Internal code: `orchestrator/brains/finrl_brain.py`
