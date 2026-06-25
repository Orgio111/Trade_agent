---
title: LLM Regime Brain
type: entity
tags: [brain, llm, regime, ollama, nim, openrouter]
created: 2026-06-26
updated: 2026-06-26
weight: 0.15
brain_id: llm_regime
source_file: llm_regime_brain.py
status: active
---

# LLM Regime Brain

## Overview

**Brain #3** — Dual-tier LLM regime classifier. Feeds Binance market data (price change, volume ratio, RSI, headlines) into a structured JSON prompt. Classifies market into 5 regimes: `trending_up`, `trending_down`, `ranging`, `volatile`, `crisis`.

## Architecture

```
Binance fetch → REGIME_PROMPT → Tier 1: Ollama (qwen2.5:3b) ──── OK → parse JSON
                                 Tier 2: NIM (meta/llama-3.1-8b-instruct) ── OK → parse JSON
                                 Tier 3: OpenRouter (google/gemma-4-31b-it:free) → parse JSON
                                 ↘ All fail: keyword_regime_match (crash/bull/bear scan)
```

## Parameters

| Parameter | Default | Env var |
|-----------|---------|---------|
| ollama_url | `http://localhost:11434` | `OLLAMA_URL` |
| ollama_model | `qwen2.5:3b` | `OLLAMA_REGIME_MODEL` |
| nim_url | `https://integrate.api.nvidia.com` | `NIM_URL` |
| nim_model | `meta/llama-3.1-8b-instruct` | `NIM_REGIME_MODEL` |
| openrouter_model | `google/gemma-4-31b-it:free` | `OPENROUTER_REGIME_MODEL` |
| timeout | 8.0s | `LLM_REGIME_TIMEOUT` |
| cache_ttl | 300s (5 min) | — |

## Key: NIM env

Checks `NIM_API_KEY` first, then `NVIDIA_API_KEY` as fallback.

## Outputs

- `score`: ∈ [-1, +1] (regime mapped: trending_up=+0.5, crisis=-0.9, etc.)
- `confidence`: ∈ [0, 1]
- `metadata`: regime, rationale, tier_used

## Fallback chain

1. **Ollama** (local, lowest latency)
2. **NIM** (cloud, free credits)
3. **OpenRouter**:free** (cloud, rate-limited)
4. **Keyword match** (regex on headlines → regime guess)

## Fail modes

- Ollama offline → tier 2/3
- NIM 403/404 → tier 3
- OpenRouter 429 → OK (key valid, rate-limited)
- All tiers fail → keyword fallback
- LLM returns invalid JSON → retry or keyword fallback

## Edge & known weaknesses

- **Edge:** LLM captures qualitative context (news, narrative) that no TA can
- **Edge:** 3-tier fallback makes it resilient — almost never fully offline
- **Weakness:** LLM inference latency (8s timeout) — not suitable for HFT
- **Weakness:** Small models hallucinate regime classifications
- **Weakness:** Cache (5min TTL) may miss rapid regime shifts
- **10× opportunity:** Fine-tune a small classifier on manually-labeled regime data; replace generic LLM

## Related

- [[finbert-brain]] — same tier architecture, different purpose (sentiment vs regime)
- [[timesfm-brain]] — regime context can modulate TimesFM weight in orchestrator
- [[freqai-brain]] — RSI fed into regime prompt

## Sources

- Internal code: `orchestrator/brains/llm_regime_brain.py`
