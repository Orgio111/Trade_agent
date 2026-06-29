---
title: "Model Sequential Loading"
type: concept
tags: [model-loading, vram, ollama, sequential, gpu-memory]
created: 2026-06-30
updated: 2026-06-30
status: stable
---

# Model Sequential Loading

## Definition

A VRAM-aware model loading strategy for consumer GPUs (RTX 4050 6GB) that ensures only one large model is loaded at a time. Models are loaded on-demand and unloaded before the next model is loaded, with an always-warm fast model (phi3:3.8b) kept resident for <100ms scalping latency.

## Intuition

6GB VRAM can't hold multiple 8B models simultaneously. Sequential loading means: phi3:3.8b is always warm (2.5GB), and when qwen3:8b (4.5GB) is needed, phi3 is temporarily swapped out. This requires careful VRAM budgeting and Ollama's model management API to unload/load without crashing the inference server.

## VRAM Budget (RTX 4050 6GB)

| Model | VRAM | Latency | Role |
|-------|:----:|:-------:|------|
| phi3:3.8b | 2.5 GB | <100ms | Always-warm fast scalper |
| qwen3:8b | 4.5 GB | 200-400ms | On-demand reasoning |
| deepseek-r1:8b | 4.7 GB | 250-500ms | Async deep analysis |
| qwen2.5:3b | 2.0 GB | 80-150ms | Fallback fast |

## Loading Sequence

```
1. phi3:3.8b loaded (always resident)     → 2.5GB used
2. Need qwen3:8b → unload phi3            → 0GB used
3. Load qwen3:8b                          → 4.5GB used
4. Done with qwen3 → unload               → 0GB used
5. Reload phi3:3.8b                       → 2.5GB used
```

## How we use it

- Implemented in: `inference/providers/local_ollama.py` — `LocalOllamaProvider`
- Ollama manages VRAM via `ollama pull` / `ollama rm` / model auto-loading
- Referenced by: [[local-trading-ai-architecture]] — RTX 4050 optimized design

## Related

- [[local-trading-ai-architecture]] — overall local AI system design
- [[vllm-inference-provider]] — alternative GPU inference (requires more VRAM)
- [[brain-ecosystem]] — brains that consume model inference

## Sources

- Internal design: AGENTS.md §ODOO ERP → TRADING AI INTEGRATION
