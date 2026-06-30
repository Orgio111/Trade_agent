# Model Loader VRAM Strategy

## Problem
RTX 4050 6GB VRAM — cannot fit all Ollama models simultaneously.

## Solution: Sequential Loading
Only **one model in VRAM at a time**. Ollama auto-swaps, but we track explicitly.

## Model Assignments
| Agent | Model | Size | Role |
|-------|-------|------|------|
| Coordinator | `phi3:mini` | 2.2GB | Pinned (always in VRAM) |
| QuantAgent | `qwen2.5:3b` | 1.9GB | Quantitative analysis |
| PatternAgent | `moondream` | 1.7GB | Vision/patterns |
| MacroAgent | `qwen2.5:3b` | 1.9GB | Macro context |
| RiskAgent | `qwen2:1.5b` | 0.9GB | Risk filter |

## Key Features

### GPU Pinned Models
```python
ModelLoader(gpu_pinned=["phi3:mini"])
```
Pinned models stay loaded (not unloaded during sequential swaps).

### Keep-Alive
```python
ModelLoader(keep_alive="5m")
```
Ollama `keep_alive` parameter — model stays in VRAM for 5 minutes after last use.

### Warmup Methods
```python
# Single model warmup
loader.warmup_model("qwen2.5:3b")

# Batch warmup all assigned models
loader.warm_all()  # Sequential, respects VRAM budget
```

### Load Logic
```python
def load(model, warmup=True):
    if model in _loaded: return True
    if vram_used + model_vram > vram_budget:
        unload_lru()  # Except pinned
    ollama.pull(model)
    if warmup: tiny_inference(model)
    _loaded[model] = LoadInfo(...)
    return True
```

## Usage in AutoGenExecutor
```python
self.model_loader = ModelLoader(
    gpu_pinned=["phi3:mini"],
    keep_alive="5m",
)

def warmup(self):
    for m in [MODEL_QUANT, MODEL_PATTERN, MODEL_MACRO, MODEL_RISK, MODEL_COORDINATOR]:
        self.model_loader.warmup_model(m)
```

## Status
```python
loader.get_status() -> {
    "loaded": ["phi3:mini", "qwen2.5:3b"],
    "pinned": ["phi3:mini"],
    "warmup_times_ms": {"qwen2.5:3b": 11158, ...},
    "vram_used_mb": 4100,
    "vram_budget_mb": 5120
}
```

## Related
- [[AutoGen Trading Pipeline]]
- [[Enhanced Risk Engine]]
- [[Backtest Engine]]