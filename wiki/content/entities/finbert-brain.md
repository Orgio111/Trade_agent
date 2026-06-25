---
title: FinBERT NLP Brain
type: entity
tags: [brain, nlp, sentiment, ollama, nim, openrouter]
created: 2026-06-26
updated: 2026-06-26
weight: 0.10
brain_id: finbert_nlp
source_file: finbert_brain.py
status: active
---

# FinBERT NLP Brain

## Overview

**Brain #5** — NLP sentiment analyzer. Auto-fetches crypto news from CryptoCompare API. Runs 2 concurrent Ollama models for consensus scoring. Outputs bounded sentiment between -1.0 (panic) and +1.0 (euphoria).

## Architecture

```
CryptoCompare fetch → SENTIMENT_PROMPT → Ollama model 1 (qwen2.5:3b) ──┐
push_headlines ──────→ SENTIMENT_PROMPT → Ollama model 2 (phi3:mini) ──┤→ average → score
                                       ↘ Tier 2: NIM ────────────────┤
                                         Tier 3: OpenRouter:free ───┘
                                         ↘ keyword_sentiment_scan
```

## Parameters

| Parameter | Default | Env var |
|-----------|---------|---------|
| ollama_models | `qwen2.5:3b,phi3:mini` | `OLLAMA_SENTIMENT_MODELS` |
| nim_model | `meta/llama-3.1-8b-instruct` | `NIM_SENTIMENT_MODEL` |
| openrouter_model | `google/gemma-4-31b-it:free` | `OPENROUTER_SENTIMENT_MODEL` |
| timeout | 8.0s | `SENTIMENT_TIMEOUT` |
| headline_cache_ttl | 300s (5 min) | — |
| concurrent_ollama | 2 | — |

## Outputs

- `score`: ∈ [-1, +1] (panic..euphoria)
- `confidence`: ∈ [0, 1]
- `metadata`: sentiment (very_bullish..very_bearish), key_signal, models_used

## Fallback chain

1. **2× Ollama concurrent** (consensus from 2 models)
2. **NIM** (cloud fallback)
3. **OpenRouter**:free** (rate-limited cloud)
4. **Keyword sentiment scan** (bullish/bearish word matching on headlines)

## Fail modes

- All Ollama models offline → cloud tiers
- CryptoCompare API down → cached headlines (5min TTL)
- Both Ollama models return conflicting sentiment → average (may cancel out)
- LLM returns invalid JSON → keyword fallback

## Edge & known weaknesses

- **Edge:** Dual-model consensus reduces hallucination — one model's noise is averaged out
- **Edge:** CryptoCompare auto-fetch means no manual sentiment input needed
- **Weakness:** CryptoCompare free tier has rate limits
- **Weakness:** Averaging 2 models can produce neutral signal when they disagree → low actionable value
- **10× opportunity:** Fine-tune FinBERT/sentiment model on crypto-specific data; add social media (X/Twitter) feed

## Related

- [[llm-regime-brain]] — same tier infrastructure, different purpose
- [[statarb-brain]] — funding rate sentiment complements NLP sentiment
- [[onchain-brain]] — whale flows + NLP provide multi-modal conviction

## Sources

- Internal code: `orchestrator/brains/finbert_brain.py`
