---
title: "Attention-Based Regime Detection"
type: concept
tags: [regime, transformer, attention, market-structure, deep-learning]
created: 2026-06-26
updated: 2026-06-26
sources: [[custom-trading-brain-architecture]], [[codecrafters-build-your-own-x]]
status: research
---

# Attention-Based Regime Detection

## Definition

Using Transformer self-attention weight patterns to automatically classify market regimes (trending, ranging, volatile, crisis). The attention matrix encodes which historical timesteps the model considers most relevant for its current prediction — and the structure of this matrix changes systematically across different market conditions.

## Core Insight

The Transformer in [[neural-network-brain]] already captures attention weights via `forward_with_attention()`. These weights are not just a byproduct of prediction — they are a **diagnostic signal** about market structure.

### Attention Entropy as Regime Metric

**Shannon entropy** of the attention distribution measures how "focused" or "spread out" the model's attention is:

```
H = -Σ p(i) * log(p(i))
```

| Entropy | Attention Pattern | Market Regime | Action |
|---------|-------------------|---------------|--------|
| **Low** (< 2.0) | Concentrated on recent bars | Strong trend | Follow trend, increase size |
| **Medium** (2.0–3.5) | Balanced focus | Ranging / consolidating | Mean-revert, reduce size |
| **High** (> 3.5) | Diffuse / spread across all bars | Volatile / crisis / transition | Defensive, tighten stops, reduce size |

### Attention Pattern Shape

Beyond entropy, the **shape** of the attention matrix reveals regime structure:

- **Trending regime**: Attention is localized to the last 5-10 bars (recent momentum matters). The attention matrix shows a strong diagonal band.
- **Ranging regime**: Attention is spread across the lookback window with periodic peaks (support/resistance levels repeat). The matrix shows vertical stripes at key price levels.
- **Crisis regime**: Attention becomes erratic — no clear pattern, high entropy, attention jumps between distant bars (flash crashes, news events).
- **Transition regime**: Attention entropy is increasing — the model is losing confidence in recent patterns, suggesting an impending regime change.

## Current System's Regime Detection

| Method | File | Speed | Accuracy | Weakness |
|--------|------|-------|----------|----------|
| **HMM** | `hmm_regime.py` | Fast (~1ms) | Medium | Requires retraining, fixed 4 states |
| **LLM** | `llm_regime_brain.py` | Slow (~2-8s) | High | API dependency, latency |
| **ADX** | `feature_engine.py` | Fast (~1ms) | Low | Only trend strength, not regime |
| **Wyckoff** | `market_structure.py` | Medium (~10ms) | Medium | Pattern-dependent, subjective |
| **Attention** (proposed) | `custom_nn_brain.py` | Fast (~15ms) | TBD | Requires trained Transformer |

### Integration Opportunity

The attention-based detector would be a **5th regime signal** in `ensemble_meta.py`:

```
Current ensemble sources:
  ml_rf (1.0)       — ML/RF predictions
  ppo_rl (1.5)      — PPO RL actions
  regime (0.8)      — HMM regime signal ← REPLACE with attention-based
  rule_smc (0.7)    — Smart Money Concepts

Proposed new source:
  attention_regime (1.0) — Attention entropy + pattern clustering
```

## Proposed Architecture

### Phase 1: Entropy-Based Regime Filter (Simple, Fast)

Add to `TransformerPredictor`:

```python
def get_regime_from_attention(self, features: np.ndarray) -> dict:
    """Classify regime from attention patterns.
    
    Returns:
        dict with regime, confidence, entropy, attention_type
    """
    attn_weights = self.get_attention_weights(features)  # (nhead, seq, seq)
    if attn_weights is None:
        return {"regime": "unknown", "confidence": 0.0}
    
    # Average across heads
    avg_attn = np.mean(attn_weights, axis=0)  # (seq, seq)
    
    # Compute entropy of last row (which positions matter most)
    last_row = avg_attn[-1]
    last_row = last_row / (last_row.sum() + 1e-10)
    entropy = -np.sum(last_row * np.log(last_row + 1e-10))
    
    # Classify regime
    if entropy < 2.0:
        regime = "trending"
        confidence = min(0.9, 0.6 + (2.0 - entropy) * 0.2)
    elif entropy < 3.5:
        regime = "ranging"
        confidence = 0.5
    else:
        regime = "volatile"
        confidence = min(0.9, 0.6 + (entropy - 3.5) * 0.2)
    
    # Check for recent-focus pattern (trending indicator)
    recent_focus = float(avg_attn[-1, -5:].sum())
    if recent_focus > 0.5 and regime == "trending":
        confidence = min(0.95, confidence + 0.1)
    
    return {
        "regime": regime,
        "confidence": round(confidence, 3),
        "entropy": round(float(entropy), 3),
        "recent_focus": round(recent_focus, 3),
    }
```

### Phase 2: Attention Pattern Clustering (Advanced)

1. **Collect** attention matrices over time (store last 200 windows)
2. **Flatten** each (seq × seq) matrix to a feature vector
3. **Reduce dimensionality** with PCA (keep 10-20 components)
4. **Cluster** with K-Means (k=4: bull, bear, ranging, volatile)
5. **Map** cluster centers to regime labels based on correlation with price action

This approach discovers regimes empirically rather than imposing fixed categories.

### Phase 3: Dynamic Brain Weight Adjustment

Use attention entropy to **dynamically adjust brain weights** in the Go orchestrator:

```python
# When attention entropy is high (volatile regime):
#   - Increase weight of statarb_brain (mean reversion works in ranging)
#   - Decrease weight of custom_nn_brain (model confused)
#   - Decrease weight of finrl_kelly (position sizing risky)

# When attention entropy is low (trending regime):
#   - Increase weight of custom_nn_brain (model confident)
#   - Increase weight of timesfm_brain (forecasting works)
#   - Decrease weight of statarb_brain (mean reversion fails in trends)
```

## Validation Plan

| Step | Method | Duration | Success Criteria |
|------|--------|----------|------------------|
| 1 | Backtest entropy vs realized volatility | 1 week | Correlation > 0.6 |
| 2 | Backtest entropy vs regime transitions | 2 weeks | Detects >70% of transitions |
| 3 | Paper trade with attention regime filter | 1 month | Sharpe improvement > 0.2 |
| 4 | Live paper trade with dynamic weights | 2 months | Consistent alpha |

## Risks & Limitations

1. **Attention ≠ Causation** (Jain & Wallace, 2019): Attention weights are diagnostic, not causal. A high-entropy attention pattern doesn't *cause* volatility — it reflects the model's uncertainty about it.
2. **Look-ahead bias**: Attention weights are computed from the same data used for prediction. Need to ensure regime classification doesn't leak future information.
3. **Overfitting**: If the Transformer is overfit, attention patterns may reflect noise rather than genuine market structure. Regularization and proper train/test splits are critical.
4. **Latency**: Computing attention weights adds ~15ms per inference. At 15s cadence this is negligible, but for HFT it matters.

## Related

- [[neural-network-brain]] — the Transformer that produces attention weights
- [[custom-trading-brain-architecture]] — full brain design document
- [[llm-regime-brain]] — current LLM-based regime detection
- [[hmm-regime]] — current HMM-based regime detection
- [[ensemble-meta-model]] — where regime signals are fused
- [[codecrafters-build-your-own-x]] — learning resources for Transformer implementation

## Open Questions

- What entropy threshold best separates trending vs ranging vs volatile for BTCUSDT?
- Should we use per-head entropy (4 values) or average entropy (1 value)?
- How often should attention patterns be reclustered? (daily? weekly?)
- Can we detect regime transitions *before* they happen using entropy acceleration?
- Should attention regime be a separate brain or a feature of custom_nn_brain?
