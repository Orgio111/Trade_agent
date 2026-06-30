# Troubleshooting

## Common Issues

### ModuleNotFoundError: No module named 'orchestrator'
**Cause**: Running from wrong directory or missing PYTHONPATH.

**Fix**:
```bash
cd /path/to/Trade_agent
PYTHONPATH=. python -m orchestrator.autogen_trader single --paper
```

### ModuleNotFoundError: cannot import name 'MODEL_MACRO_RISK'
**Cause**: Old constant name in autogen_executor.py warmup().

**Fix**: Use correct names from autogen_team.py:
```python
MODEL_COORDINATOR = "phi3:mini"
MODEL_QUANT = "qwen2.5:3b"
MODEL_PATTERN = "moondream"
MODEL_MACRO = "qwen2.5:3b"
MODEL_RISK = "qwen2:1.5b"
```

### Ollama Connection Refused
**Cause**: Ollama not running.

**Fix**:
```bash
ollama serve
# Verify
curl http://localhost:11434/api/tags
```

### Model Not Found
**Cause**: Model not pulled.

**Fix**:
```bash
ollama pull phi3:mini
ollama pull qwen2.5:3b
ollama pull moondream
ollama pull qwen2:1.5b
```

### VRAM OOM (Out of Memory)
**Cause**: Too many models loaded simultaneously.

**Fix**: 
- ModelLoader sequential loading handles this automatically
- Reduce `gpu_pinned` list
- Increase `keep_alive` to reduce reload frequency

### PaperClient.place_order RuntimeWarning: coroutine never awaited
**Cause**: Calling async method from sync context.

**Fix**: Use sync wrapper (already implemented in autogen_executor.py):
```python
def _execute_signal(self, signal, risk_decision):
    return asyncio.run(self._execute_signal_async(signal, risk_decision))
```

### Circuit Breaker Stuck
**Cause**: 5 consecutive losses triggered 5min cooldown.

**Fix**: Wait 5 minutes or restart executor (resets internal state).

### Config Not Loading
**Cause**: config.yaml not found or wrong path.

**Fix**: Check `--config` path, or ensure config.yaml in working directory.

### ImportError: cannot import 'BinanceClient' from 'execution'
**Cause**: Local `orchestrator/execution.py` shadows root `execution.py`.

**Fix**: Use absolute import or rename local file.

## Debug Commands
```bash
# Test imports
PYTHONPATH=. python -c "from orchestrator.autogen_executor import AutoGenSignalExecutor; print('OK')"

# Test risk engine
PYTHONPATH=. python -c "from orchestrator.enhanced_risk import EnhancedRiskEngine; e=EnhancedRiskEngine(); print(e.get_status())"

# Test model loader
PYTHONPATH=. python -c "from orchestrator.model_loader import ModelLoader; m=ModelLoader(); print(m.gpu_pinned)"
```

## Logs
```bash
# Follow trading logs
tail -f logs/agent.log | grep -E "(BUY|SELL|REJECTED|APPROVED|ERROR)"

# Filter by cycle
grep "cycle_0001" logs/agent.log
```

## Related
- [[AutoGen Trading Pipeline]]
- [[CLI Reference]]