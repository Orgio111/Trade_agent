---
title: Local AI Deployment Guide
type: playbook
tags: [deployment, ollama, rtx4050, trading-ai, windows, wsl2]
created: 2026-06-30
updated: 2026-06-30
sources: [[local-trading-ai-architecture], [rtx4050-trading-system]]
status: stable
---

# Local AI Deployment Guide

## When to Use

Deploy a fully autonomous trading AI on local hardware (RTX 4050 / 16GB RAM) using Ollama. No cloud dependencies after initial model download.

## Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| GPU | RTX 3050 (4GB) | RTX 4050+ (6GB+) |
| RAM | 12GB | 16GB+ |
| Storage | 50GB free | 100GB+ NVMe |
| OS | Windows 10/11, Linux | Windows 11 + WSL2 or native Linux |
| Network | For model download only | Offline-capable after setup |

## How to Run

### 1. Install Ollama

**Windows (PowerShell Admin):**
```powershell
winget install Ollama.Ollama
# Or download from https://ollama.ai/download
```

**Linux/WSL2:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. Pull Models (Sequential)

```bash
# Embeddings (required first)
ollama pull nomic-embed-text

# Core models (order by priority)
ollama pull phi3:3.8b         # Always-warm scalping model
ollama pull qwen3:8b          # Main reasoning
ollama pull mistral:7b        # Tool execution
ollama pull deepseek-r1:8b    # Deep macro analysis
ollama pull moondream         # Vision (optional)

# Verify
ollama list
```

### 3. Configure Ollama for RTX 4050

**Windows (Environment Variables):**
```powershell
$env:OLLAMA_NUM_PARALLEL = "1"
$env:OLLAMA_MAX_LOADED_MODELS = "1"
$env:OLLAMA_FLASH_ATTENTION = "1"
$env:OLLAMA_KV_CACHE_TYPE = "f16"
$env:OLLAMA_GPU_LAYERS = "35"
# Make permanent:
[Environment]::SetEnvironmentVariable("OLLAMA_NUM_PARALLEL", "1", "User")
# ... repeat for others
```

**Linux/WSL2 (~/.bashrc or /etc/environment):**
```bash
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=f16
export OLLAMA_GPU_LAYERS=35
```

### 4. Pre-warm Phi3 (Always-Warm Model)

```bash
# Keep phi3 loaded in background
ollama run phi3:3.8b "System ready. Respond with: OK" &
# Verify
ollama ps
```

### 5. Install Python Dependencies

```bash
# Create venv
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux:
source .venv/bin/activate

# Install
pip install --upgrade pip
pip install fastapi uvicorn websockets chromadb python-binance pandas numpy pyyaml ollama
```

### 6. Project Structure

```
trading_ai/
├── config.yaml          # All parameters
├── main.py              # FastAPI orchestrator
├── models.py            # Pydantic models
├── state.py             # IncrementalState
├── router.py            # OllamaRouter
├── risk.py              # RiskEngine
├── memory.py            # LocalMemory (ChromaDB)
├── execution.py         # ExecutionEngine
├── features.py          # Feature extraction
├── vision.py            # Vision pipeline
└── requirements.txt
```

### 7. Configuration (config.yaml)

```yaml
# config.yaml
system:
  symbol: "BTCUSDT"
  timeframe: "1m"
  paper_mode: true
  exchange: "binance_testnet"

model:
  warm_model: "phi3:3.8b"
  reasoning_model: "qwen3:8b"
  execution_model: "mistral:7b"
  vision_model: "moondream"
  deep_model: "deepseek-r1:8b"
  embedding_model: "nomic-embed-text"

risk:
  max_position_pct: 0.05
  max_daily_drawdown: 0.03
  max_concurrent: 3
  kill_streak: 5

memory:
  chroma_path: "./memory"
  buffer_size: 200
  regime_states: 5

execution:
  maker_preference: true
  default_type: "LIMIT"
```

### 8. Run the System

```bash
# Terminal 1: Start orchestrator
python main.py

# Terminal 2: Test with simulated candles (or connect real WS)
python test_client.py
```

### 9. Health Checks

```bash
# Ollama status
ollama ps
curl http://localhost:11434/api/tags

# ChromaDB
python -c "import chromadb; c=chromadb.PersistentClient('./memory'); print(c.list_collections())"

# FastAPI
curl http://localhost:8000/health

# Model latency test
python -c "
import ollama, time
for m in ['phi3:3.8b', 'qwen3:8b']:
    start=time.time()
    r=ollama.generate(m, 'test', options={'num_predict': 10})
    print(f'{m}: {time.time()-start:.2f}s')
"
```

## Verification

### Expected Results

| Check | Pass Criteria |
|-------|---------------|
| `ollama ps` | phi3:3.8b shows as loaded |
| `nvidia-smi` | VRAM ~3-4GB used (phi3 + embeddings + OS) |
| Latency test | phi3 <100ms, qwen3 <300ms |
| ChromaDB | Collections: trades, patterns |
| FastAPI | `/health` returns 200 OK |

### Smoke Test

```python
# test_client.py
import asyncio, websockets, json, time

async def test():
    async with websockets.connect("ws://localhost:8000/candle") as ws:
        # Send fake candle
        candle = {
            "timestamp": int(time.time()*1000),
            "open": 50000, "high": 50100, "low": 49900, "close": 50050,
            "volume": 100.5, "symbol": "BTCUSDT"
        }
        await ws.send(json.dumps(candle))
        response = await ws.recv()
        print(json.loads(response))

asyncio.run(test())
```

Expected: Response with `signal`, `execution`, `state` within 200ms (warm phi3).

## Common Issues

| Issue | Fix |
|-------|-----|
| OOM on model load | Reduce `OLLAMA_GPU_LAYERS` to 30, enable CPU offload |
| Phi3 not staying warm | Increase `OLLAMA_MAX_LOADED_MODELS` to 2, keep phi3 small |
| Slow first inference | Run warmup: `ollama run phi3:3.8b "warmup"` |
| ChromaDB memory | Limit buffer_size in config.yaml |
| Windows WSL2 GPU | Ensure NVIDIA drivers in WSL (`nvidia-smi` in WSL) |

## Maintenance

```bash
# Weekly: Update models
ollama pull phi3:3.8b && ollama pull qwen3:8b

# Monthly: Clean ChromaDB
python -c "
import chromadb
c = chromadb.PersistentClient('./memory')
# Archive old trades, keep last 10k
"

# Monitor disk
du -sh ./memory
```

## Related

- [[local-trading-ai-architecture]] — architecture concept
- [[rtx4050-trading-system]] — hardware entity
- [[incremental-candle-state]] — memory concept