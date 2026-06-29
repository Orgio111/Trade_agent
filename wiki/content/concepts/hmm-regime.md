---
title: "HMM Regime Detection"
type: concept
tags: [hmm, regime, hidden-markov-model, classification, market-state]
created: 2026-06-30
updated: 2026-06-30
status: stable
---

# HMM Regime Detection

## Definition

Hidden Markov Model for probabilistic market regime classification. Detects 4 latent regimes — Bull (trending up), Bear (trending down), Range (low volatility), and High-Vol (chaotic) — from multi-dimensional observation features extracted from OHLCV data. Replaces simple ADX-only regime detection with a probabilistic approach that provides regime probabilities and transition confidence.

## Intuition

Markets don't follow a single pattern — they switch between trending, ranging, and volatile states. An HMM treats these as hidden states that emit observable features (returns, volatility, volume). By fitting the model on historical data and decoding with Viterbi, we get the most likely current regime and its probability. This is more robust than threshold-based rules because it considers the full state transition dynamics.

## Architecture

```
OHLCV DataFrame
    │
    ▼
Feature Extraction (5-dim)
  ├── 1-period normalized return
  ├── 5-period normalized return
  ├── ATR/close (volatility)
  ├── Volume ratio (vs SMA-20)
  └── Normalized RSI
    │
    ▼
GaussianHMM (4 states, diagonal covariance)
    │
    ├── Viterbi decoding → most likely state sequence
    └── Posterior probabilities → regime confidence
    │
    ▼
Post-hoc labeling (by emission means):
  Highest mean return → Bull
  Lowest mean return → Bear
  Lowest volatility → Range
  Highest volatility → High-Vol
    │
    ▼
Regime + Confidence + Transition Matrix
```

## How we use it

- Source: `orchestrator/hmm_regime.py` — `HMMRegimeDetector` class
- Used by: `orchestrator/ensemble_meta.py` as one of the signal sources in the ensemble
- Used by: `orchestrator/brains/llm_regime_brain.py` as a fallback when LLM is unavailable
- Training: `hmm.fit(df)` on historical OHLCV; prediction: `hmm.predict(df)` returns regime + confidence
- Signal generation: `hmm.get_regime_signal(df)` returns direction (long/short/hold) based on regime

## Strengths & weaknesses

**Strengths:**
- Probabilistic — gives regime probabilities, not just classification
- Captures state transitions (regime persistence, switching frequency)
- 5-dimensional features capture multiple market aspects simultaneously

**Weaknesses:**
- Requires sufficient history (lookback=50 candles minimum)
- Assumes Gaussian emissions (may not capture fat tails well)
- Regime labels are assigned post-hoc, not learned — sensitive to initial conditions
- Fallback to ADX-based detection when HMM is not fitted

## Related

- [[attention-based-regime-detection]] — alternative regime detection using Transformer attention
- [[ensemble-meta-model]] — consumes HMM regime as one signal source
- [[brain-ecosystem]] — LLM Regime Brain uses this as fallback

## Sources

- Internal code: `orchestrator/hmm_regime.py`
