---
title: "Custom Trading Brain Architecture"
type: concept
tags: [neural-network, brain, architecture, transformer, lstm, deep-learning]
created: 2026-06-26
updated: 2026-06-26
sources: [[codecrafters-build-your-own-x]]
status: stable
---

# Custom Trading Brain Architecture

## Definition

A proposed new brain for the Trade_agent orchestrator that uses deep learning (LSTM/GRU or Transformer) to predict price movements from OHLCV data and technical indicators. Designed to complement the existing 9 brains by adding learned temporal pattern recognition.

## Intuition

The existing brains use classical techniques: rule-based scoring ([[freqai-brain]]), foundation models ([[timesfm-brain]]), LLM classification ([[llm-regime-brain]]), and statistical methods ([[statarb-brain]]). None of them *learn* temporal patterns from our specific trading data. A custom neural network brain would fill this gap by training on historical BTCUSDT data to recognize regime-dependent patterns that rules cannot capture.

## Architecture Design

### Option A: LSTM/GRU (Simpler, faster to prototype)
```
Input (128-dim) → LSTM(128, 2 layers) → FC(64) → ReLU → FC(32) → Output(score, confidence)
```
- **Pros**: Fast inference (~50ms), handles sequential data, well-understood
- **Cons**: Prone to overfitting on market noise, limited long-range memory

### Option B: Transformer (More powerful, recommended)
```
Input (128-dim) → MultiHeadAttention(4 heads, 128-dim) → LayerNorm → 
FFN(512) → LayerNorm → Output(score, confidence)
```
- **Pros**: Attention weights show *which timeframes matter*, better long-range patterns, state-of-the-art
- **Cons**: More data needed, slower inference, more complex implementation

### Recommendation: **Start with LSTM, migrate to Transformer**

The [[codecrafters-build-your-own-x]] research and expert analysis recommend:
1. LSTM for rapid prototyping (1-2 days)
2. Transformer for production (1-2 weeks after LSTM validates)

## Observation Space (128-dim, compatible with existing TradingEnvironment)

```
[0:7]     — OHLCV + returns (existing)
[7:30]    — Technical indicators: RSI, MACD, BB, ATR, volume profile
[30:38]   — Position state (existing)
[38:40]   — ML signal (existing)
[40:50]   — NEW: LSTM/GRU hidden states (10-dim compressed)
[64:96]   — MTF features (existing)
[96:128]  — Deep encoder (existing)
```

## Training Pipeline

### Data Sources
- **Primary**: Binance REST API — 180 days BTCUSDT 1h candles via `DataLoader.from_binance_api()`
- **Fallback**: Synthetic GBM data via `generate_training_data()`
- **Live inference**: Real-time WebSocket feed via `DataPipeline`

### Data Splits
- 70% training (126 days)
- 15% testing (27 days)
- 15% validation (27 days)

### Training Parameters
| Parameter | LSTM | Transformer |
|---|---|---|
| Algorithm | PPO (stable-baselines3) | PPO |
| Learning rate | 3e-4 | 1e-4 |
| Batch size | 64 | 128 |
| Timesteps | 50,000 initial | 200,000 |
| Hidden dim | 128 | 128 |
| Layers | 2 | 4 |
| Attention heads | N/A | 4 |

### Evaluation Metrics
| Metric | Target | Method |
|---|---|---|
| Train accuracy | >55% | Direction prediction |
| Test accuracy | >52% | Validation set |
| Overfitting gap | <15pp | Train - Test |
| Sharpe ratio | >0.5 | Brain backtest |
| Profit factor | >1.2 | Brain backtest |

## Integration with Existing System

### Brain Interface
```python
class CustomNNBrain(BaseBrain):
    brain_id = "custom_nn"  # weight: 0.05
    
    async def compute_score(self, symbol: str) -> BrainSignal:
        # Fetch OHLCV, compute features, run model, return signal
        ...
```

### Weight Allocation
- Current total: 0.90 (9 brains)
- New brain: 0.05
- New total: 0.95
- Remaining gap: 0.05 (safety margin)

### Model Persistence
- Save to `models/nn_brain/custom_nn_latest.pt`
- Auto-load on warmup()
- Auto-retrain via [[continual-learning-pipeline]] integration

## Fallback Strategy
When model is unavailable (cold start, retraining):
- Rule-based scoring using RSI + MACD + BB (same as [[freqai-brain]] fallback)
- Score = 0.0, confidence = 0.1

## Learning Resources (from [[codecrafters-build-your-own-x]])

| Resource | Purpose | Time Investment |
|---|---|---|
| Andrej Karpathy "Neural Networks: Zero to Hero" | Deep understanding of backprop, transformers | 15 hours |
| DataCamp LSTM tutorial | Quick prototype | 2 hours |
| Music Gen LSTM (Keras) | Sequence prediction patterns | 3 hours |
| Build Deep Learning From Scratch | PyTorch internals | 20+ hours |

**Key insight from Karpathy course**: LSTM/GRU should be considered transitional. The course builds toward Transformer attention mechanisms, which are significantly better at capturing long-range dependencies in volatile crypto markets.

## Contradictions / updates
- Expert analysis suggests moving away from LSTM toward Transformer for production trading. LSTM is prone to overfitting on market noise. We start with LSTM for prototyping but plan migration to Transformer.

## Related
- [[nn-brain-development-guide]] — step-by-step implementation playbook
- [[freqai-brain]] — uses XGBoost, the model this brain would complement
- [[timesfm-brain]] — foundation model forecaster, this brain is a custom trained alternative
- [[finrl-brain]] — RL brain, this brain is a supervised/attention alternative
- [[brain-backtest-infrastructure]] — used for evaluating this brain's performance
- [[codecrafters-build-your-own-x]] — source of learning resources
- [[neural-network-brain]] — entity page for this brain

## Open questions
- Should we use Transformer from the start, or validate with LSTM first?
- What sequence length (lookback) works best for BTCUSDT? 50, 100, or 200 candles?
- How to handle regime changes without retraining? (attention mechanism may solve this)
- Custom loss function: should we penalize drawdown more than standard MSE?
