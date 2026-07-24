# Sanitized Runtime Log Evidence

Audit date: 2026-07-24  
Values that could identify credentials, accounts, hosts outside the local runtime,
or user data are omitted.

## vLLM

```text
Engine initialization failed because free GPU memory (approximately 4.95 GiB)
was below the configured utilization target (approximately 5.1 GiB).
Container restarted repeatedly; the OpenAI-compatible model endpoint reset the
connection.
```

Impact: the local model provider was unavailable while the adjacent inference
service continued to report healthy.

## Market data

```text
Binance WebSocket connection attempts repeatedly returned HTTP 404.
The market-data status endpoint returned HTTP 500 due to response serialization.
Multiple reconnecting feed loops were visible.
```

Impact: there was no verified live candle/order-book path into the runtime.

## Orchestration and messaging

```text
Orchestrator: registered brains=12, active brains=0, nats_connected=false.
Realtime aggregator: no_signals_yet.
NATS: one legacy stream, zero messages, zero consumers.
```

Impact: HTTP liveness did not correspond to an operational trading pipeline.

## Execution and RL

```text
Execution integration logged Binance testnet user-data-stream errors, including
HTTP 410 and name-resolution failures.
RL model restore failed because the serialized model/config was incompatible with
the installed PyTorch/PPO loading behavior.
```

Impact: paper execution remained isolated and the RL loop had no verified restored
policy.

## Monitoring and dashboard

```text
Prometheus configuration referenced a recording-rules file that was absent.
Grafana logged deprecated Angular plugin errors and plugin-install permission
errors.
Dashboard container health probe targeted /health although the implemented route
was /api/health.
```

Impact: monitoring presented partial green signals while important dependencies
and end-to-end behavior were broken.

