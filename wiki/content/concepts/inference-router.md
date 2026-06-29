---
title: "Inference Router"
type: concept
tags: [inference, routing, multi-provider, cache, cost-tracking]
created: 2026-06-30
updated: 2026-06-30
status: stable
---

# Inference Router

## Definition

Multi-provider intelligent inference orchestration that routes requests across Groq, NVIDIA NIM, OpenRouter, vLLM, and LocalOllama with semantic caching (40-60% cost reduction), task-aware provider selection, deterministic fallback chains, circuit breaking, and per-request cost/latency tracking.

## Architecture

```
InferenceTask(task_type, messages, model)
    │
    ▼
Semantic Cache Lookup (Qdrant + in-memory)
    │ (miss)
    ▼
Task-type → Provider Chain Selection
    ├── reasoning:   groq → nvidia_nim → openrouter
    ├── analysis:    nvidia_nim → groq → openrouter
    ├── fast:        groq → nvidia_nim → openrouter
    ├── classify:    openrouter → nvidia_nim → groq
    └── urgent:      groq → openrouter → nvidia_nim
    │
    ▼
Provider Call (with timeout + circuit breaker)
    │ (success)                    │ (failure)
    ▼                              ▼
Cache + Track Cost          Try Next Provider
```

## Task Types

| Type | Optimal Provider | Use Case |
|------|-----------------|----------|
| `reasoning` | Groq (Mixtral-8x7B) | Deep reasoning, strategy decisions |
| `analysis` | NVIDIA NIM (Nemotron-70B) | Market/portfolio analysis |
| `fast` | Groq (Llama3-8B) | Latency-sensitive execution |
| `classification` | OpenRouter (free) | Light text classification |
| `coding` | OpenRouter (deepseek-coder) | Code generation |
| `embedding` | NVIDIA NIM (NV-EmbedQA-E5) | Embedding |
| `urgent` | Groq (Llama3-8B-8192) | Ultra low-latency scalping |

## How we use it

- Source: `inference/router.py` — `InferenceRouter` class
- Used by: `orchestrator/brains/` — all LLM-consuming brains route through this
- Used by: `orchestrator/inference_integration.py` — brain-to-inference mapping
- Providers: `inference/providers/` — Groq, NVIDIA NIM, OpenRouter, vLLM, LocalOllama

## Related

- [[vllm-inference-provider]] — self-hosted GPU provider
- [[brain-ecosystem]] — brains that consume inference
- [[brain-backtest-infrastructure]] — backtesting uses inference for LLM brains

## Sources

- Internal code: `inference/router.py`
