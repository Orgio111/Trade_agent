---
title: "Neural Network Brain"
type: entity
tags: [brain, neural-network, lstm, transformer, deep-learning, proposed]
created: 2026-06-26
updated: 2026-06-26
sources: [[codecrafters-build-your-own-x]]
status: stable
---

# Neural Network Brain

## Definition

Proposed brain #10 for the Trade_agent orchestrator. A custom-trained deep learning model (LSTM/GRU or Transformer) that learns temporal patterns from OHLCV data and technical indicators to predict price direction.

## Status

**Status: STABLE — Implemented and tested. Supports both LSTM and Transformer architectures.**

This brain has been implemented, tested, and benchmarked. Key results:
- **Architecture selection**: `CUSTOM_NN_ARCHITECTURE` env var → `lstm` (default) or `transformer`
- **Inference latency**: 14.5ms LSTM (warm), ~15ms Transformer (warm), 6.1s (cold with data fetch)
- **Auto-train**: Working — trains both LSTM and Transformer from accumulated OHLCV data
- **Attention weights**: Transformer provides attention weight matrices showing which timesteps matter
- **Score**: -0.095 (Transformer), -0.1447 (LSTM) on test run
- **Confidence**: 25% (low — model needs more training data)
- **Fallback**: Rule-based RSI/MACD/BB scoring when model unavailable

## Specifications

| Field | Value |
|---|---|
| brain_id | `custom_nn` |
| weight | 0.05 |
| output | score(-1..+1), confidence, attention_top_positions (Transformer) |
| architecture | LSTM(12→64→1) or Transformer(12→64, 4 heads, 2 layers) — selectable via env var |
| training_data | 180d BTCUSDT 1h from Binance |
| inference_latency | 14.5ms LSTM, ~15ms Transformer (warm), 6.1s (cold) |
| model_path | `models/nn_brain/custom_nn_latest_{architecture}.pt` |
| fallback | Rule-based RSI/MACD/BB scoring |
| attention | Transformer captures multi-head attention weights for interpretability |

## How it fits in the brain ecosystem

See [[brain-ecosystem]] for the full weight table. This brain sits at the supplementary tier (0.05). Total ensemble: 12 brains, weight sum = 1.00.

## Key differentiators from existing brains

| Brain | Method | This brain's advantage |
|---|---|---|
| freqai | XGBoost (tabular) | Captures sequential dependencies XGBoost misses |
| timesfm | Foundation model (zero-shot) | Trained on our specific data regime |
| finrl | PPO RL (policy learning) | Supervised signal, not action selection |
| llm_regime | LLM classification | Faster inference, no API dependency |

## Learning path (from [[codecrafters-build-your-own-x]])

1. **Quick prototype** (1-2 days): DataCamp LSTM tutorial
2. **Sequence patterns** (3 hours): Music Gen LSTM (Keras)
3. **Deep understanding** (15 hours): Andrej Karpathy "Neural Networks: Zero to Hero"
4. **Framework internals** (20+ hours): Build Deep Learning From Scratch

## Related

- [[custom-trading-brain-architecture]] — full design document
- [[freqai-brain]] — complementary tabular model
- [[timesfm-brain]] — complementary foundation model
- [[finrl-brain]] — complementary RL model
- [[brain-ecosystem]] — full brain weight table
- [[nn-brain-development-guide]] — step-by-step playbook for building this brain
- [[brain-backtest-infrastructure]] — evaluation framework
- [[codecrafters-build-your-own-x]] — learning resources

## Open questions

- ~~LSTM vs Transformer: start with LSTM for prototyping, migrate to Transformer?~~ → **Both implemented!** Select via `CUSTOM_NN_ARCHITECTURE` env var
- Sequence length: 50 candles (current), could test 100 or 200
- Custom loss function: penalize drawdown more than MSE?
- Integration with [[continual-learning-pipeline]] for auto-retraining?
- Attention weight analysis: can we use attention patterns to dynamically adjust brain weights?
- Transformer vs LSTM performance comparison on live paper trading?
