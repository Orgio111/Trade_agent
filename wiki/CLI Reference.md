# CLI Reference

## Commands

### `single` — One Trading Cycle
```bash
python -m orchestrator.autogen_trader single [OPTIONS]
```
| Option | Default | Description |
|--------|---------|-------------|
| `--paper` / `--live` | `--paper` | Paper vs live mode |
| `--config` | `config.yaml` | Config file path |
| `--balance` | `10000` | Initial balance (paper) |
| `--max-round` | `6` | AutoGen GroupChat rounds |
| `--no-warmup` | `false` | Skip model pre-warming |
| `--live-data` | `false` | Fetch real candle from Binance |
| `--symbol` | `BTCUSDT` | Trading symbol |
| `--interval` | `1m` | Candle interval |
| `--output` | `none` | Save result JSON to file |

### `daemon` — Continuous Loop
```bash
python -m orchestrator.autogen_trader daemon [OPTIONS]
```
| Option | Default | Description |
|--------|---------|-------------|
| `--interval` | `60` | Seconds between cycles |
| `--stop-on-error` | `false` | Exit on error |
| (plus all `single` options) | | |

### `backtest` — Historical Replay
```bash
python -m orchestrator.autogen_trader backtest --data FILE [OPTIONS]
```
| Option | Description |
|--------|-------------|
| `--data` | CSV/JSON path (required) |
| `--sample-rate` | Use every Nth candle (default 1) |
| `--max-cycles` | Limit cycles (0 = all) |
| `--output` | Save report JSON |

### `status` — Current State
```bash
python -m orchestrator.autogen_trader status
```
Shows cycle count, paper/live mode, account equity, risk status, warmup state.

## Environment
```bash
# Run from repo root with PYTHONPATH
cd /path/to/Trade_agent
PYTHONPATH=. python -m orchestrator.autogen_trader single --paper
```

## Example Output
```
  ✅ [cycle_0001] COMPLETED
  Action:    BUY
  Confidence: 0.95
  Latency:   autogen=37963ms  risk=0ms  exec=6ms  total=37969ms
  Risk:      APPROVED — All risk checks passed
  Execution: {'order_id': 'paper_1', 'filled_qty': 0.0079, 'avg_price': 60000.0, 'status': 'FILLED', 'mode': 'paper'}
```

## Related
- [[AutoGen Trading Pipeline]]
- [[Backtest Engine]]