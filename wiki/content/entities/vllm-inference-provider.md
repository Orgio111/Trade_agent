---
title: vLLM Inference Provider
type: entity
tags: [inference, vllm, gpu, llm, self-hosted, nvidia]
created: 2026-06-30
updated: 2026-06-30
source_file: vllm.py
status: active
---

# vLLM Inference Provider

## Overview

Self-hosted GPU inference via vLLM — runs local LLMs (Qwen2.5, Mistral, DeepSeek) on NVIDIA GPUs with zero API cost. One of five inference providers in the [[inference-router]] system, serving as the high-throughput local option.

## Architecture

```
InferenceRouter ──→ vLLM Server (localhost:8000) ──→ GPU (CUDA)
     │                      │
     │              OpenAI-compatible API
     │              /v1/chat/completions
     ▼
  Task routing → model selection → vLLM inference
```

## File

`inference/providers/vllm.py` — `vLLMProvider(BaseProvider)`

## Quick Usage

```python
from inference.providers import vLLMProvider

provider = vLLMProvider(base_url="http://localhost:8000/v1")
result = await provider.infer(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Analyze BTC trend"}],
)
print(result.content)
```

## Setup

```bash
# Start vLLM server (one-time)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90
```

## Performance (single request, batch=1)

| GPU | 7B model | 13B model | 70B model |
|-----|:--------:|:---------:|:---------:|
| RTX 4090 (24GB) | 50-150ms | 150-400ms | N/A (OOM) |
| A100 80GB | 20-80ms | 40-120ms | 100-300ms |
| H100 80GB | 15-50ms | 30-80ms | 50-200ms |

## Model Shortcuts

```python
models = {
    "reasoning": "Qwen/Qwen2.5-7B-Instruct",
    "analysis": "Qwen/Qwen2.5-7B-Instruct",
    "fast": "Qwen/Qwen2.5-1.5B-Instruct",
    "coding": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    "classification": "Qwen/Qwen2.5-1.5B-Instruct",
}
```

## How it fits in the inference ecosystem

```
5 providers:
  GroqProvider        — ultra-fast cloud (LPU)
  NvidiaNIMProvider   — high-quality reasoning (DeepSeek V4)
  OpenRouterProvider  — 200+ models, universal fallback
  vLLMProvider        — self-hosted GPU (zero cost) ← THIS PROVIDER
  LocalOllamaProvider — CPU/GPU hybrid (RTX 4050 optimized)
```

The InferenceRouter (`inference/router.py`) selects providers based on task type, latency requirements, and availability. vLLM is preferred for high-throughput batch inference when a GPU is available.

## K8s Deployment

See `deployment/k8s/gpu-node-pool.yaml` for the full vLLM GPU Deployment, Service, and PersistentVolumeClaim.

## Related

- [[local-trading-ai-architecture]] — overall local AI system design with Ollama
- [[brain-ecosystem]] — brains that consume inference
- [[odoo-erp-trading-integration]] — Odoo brain uses LocalOllamaProvider for ERP analysis

## Sources

- Internal code: `inference/providers/vllm.py`
- Internal design: AGENTS.md §vLLM GPU INFERENCE PROVIDER
