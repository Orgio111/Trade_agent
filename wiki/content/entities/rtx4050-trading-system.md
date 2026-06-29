---
title: RTX 4050 Trading System
type: entity
tags: [hardware, rtx4050, ollama, local-deployment, gpu-optimization]
created: 2026-06-30
updated: 2026-06-30
sources: [[local-trading-ai-architecture]]
status: stable
---

# RTX 4050 Trading System

## Specification

| Component | Spec | Notes |
|-----------|------|-------|
| GPU | NVIDIA RTX 4050 Laptop | 6GB GDDR6, 2560 CUDA cores |
| CPU | Intel i5-13420H | 8 cores (4P+4E), 4.6 GHz boost |
| RAM | 16GB DDR5 | 4800 MHz |
| OS | Windows 11 / WSL2 Ubuntu 22.04 | Dual boot supported |
| Storage | 512GB+ NVMe | For models + ChromaDB + logs |

## Model Capacity (4-bit quantization)

| Model | Parameters | VRAM (4-bit) | Load Time | Inference (1k tokens) |
|-------|------------|--------------|-----------|----------------------|
| phi3:3.8b | 3.8B | ~2.5GB | ~500ms | ~80ms |
| qwen3:8b | 8B | ~5.2GB | ~1000ms | ~250ms |
| deepseek-r1:8b | 8B | ~5.2GB | ~1000ms | ~300ms |
| mistral:7b | 7B | ~4.5GB | ~800ms | ~200ms |
| moondream | 1.6B | ~1.8GB | ~300ms | ~150ms |
| nomic-embed-text | 137M | ~0.5GB | ~200ms | ~20ms |

## VRAM Management Strategy

**Rule**: Only ONE 8B-class model in VRAM at a time.

```
Default State:
├── phi3:3.8b (LOADED, warm)          → 2.5GB
├── nomic-embed-text (LOADED)         → 0.5GB
├── Available VRAM                    → 3.0GB
└── System/OS/Display                 → ~1.5GB overhead      → 1.5GB
                                            = 6GB total
```

**Loading Sequence**:
1. phi3:3.8b loads at startup (always warm)
2. nomic-embed-text loads for embeddings
3. On demand: qwen3:8b loads → phi3 unloads → inference → phi3 reloads
4. On demand: deepseek-r1:8b loads in background (async)
5. On demand: mistral:7b loads → execute tools → unload

## Latency Profiles

| Path | Models | Target Latency | Typical |
|------|--------|----------------|---------|
| Scalping | phi3:3.8b (warm) | <200ms end-to-end | ~185ms |
| Normal | qwen3:8b (load) | <3s cold, <500ms warm | 2.8s / 450ms |
| Macro | deepseek-r1:8b (async) | N/A (background) | >2s |
| Vision | moondream (on-demand) | <1s | ~800ms |
| Execution | mistral:7b (load) | <1s | ~900ms |

## Ollama Configuration

```yaml
# ~/.ollama/config.yaml or environment variables
OLLAMA_NUM_PARALLEL: 1          # Sequential only
OLLAMA_MAX_LOADED_MODELS: 1     # Enforce single 8B model
OLLAMA_FLASH_ATTENTION: 1       # Enable flash attention
OLLAMA_KV_CACHE_TYPE: f16       # Half-precision KV cache
```

## Deployment Checklist

- [ ] Ollama installed and running (`ollama serve`)
- [ ] Models pulled: `ollama pull phi3:3.8b qwen3:8b deepseek-r1:8b mistral:7b moondream nomic-embed-text`
- [ ] Python environment with: fastapi, chromadb, websockets, python-binance, pandas, numpy
- [ ] ChromaDB persistence directory configured
- [ ] Binance API keys (testnet first) or paper trading mode
- [ ] WebSocket endpoint for candle feed (Binance/Bybit/Deribit)

## Integration Points

- **Brain Registry**: Complements existing 11-brain orchestrator as a local inference fallback
- **Signal Aggregation**: Uses same weighted aggregation logic from [[signal-aggregation-logic]]
- **Memory**: ChromaDB for vector memory, incremental state for candle linking
- **Risk**: Hard-coded rules (no LLM) matching existing risk engine

## Known Limitations

1. **No concurrent 8B models** — sequential loading adds 500-1000ms latency on model switch
2. **Phi3 reasoning ceiling** — 3.8B model may hallucinate on complex multi-step reasoning
3. **No multi-GPU** — single RTX 4050 only
4. **Windows WSL2 overhead** — ~5-10% GPU performance loss vs native Linux
5. **Model quantization quality** — 4-bit may degrade on low-probability tokens