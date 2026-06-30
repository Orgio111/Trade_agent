---
title: Ultra-Low Latency Monitoring
slug: ultra-low-latency-monitoring
category: entity
status: active
updated: 2026-07-01
tags: [gpu, monitoring, grafana, prometheus, scalping, vllm, rtx4050]
related:
  - "[[scalping-engine]]"
  - "[[local-inference-server]]"
  - "[[gpu-optimizer]]"
  - "[[multi-agent-pipeline]]"
  - "[[infrastructure-overview]]"
---

# Ultra-Low Latency Monitoring

> Grafana dashboard + Prometheus metrics for GPU utilization, VRAM, vLLM inference latency, and scalping engine timing.

## Overview

A dedicated monitoring stack for the ultra-low latency trading components — GPU, vLLM inference, and scalping engine. Provides real-time visibility into hardware utilization and sub-5ms decision latency.

## Dashboard: `QUANTEX Ultra-Low Latency Stack`

**UID:** `quantex-ultra-low-latency`
**Refresh:** 5s
**Time range:** Last 1 hour

### Panel Layout (16 panels, 4 rows)

```
Row 0 (y=0):   GPU Utilization │ VRAM Usage │ GPU Temperature │ GPU Power │ Status Checks
Row 1 (y=6):   VRAM Over Time (12w)         │ GPU Utilization + Temp Over Time (12w)
Row 2 (y=14):  vLLM Inference Latency (12w)  │ vLLM Tokens Generated (12w)
Row 3 (y=22):  vLLM Request Rate (8w)        │ vLLM Active (4w) │ (empty)
Row 4 (y=28):  Scalping Decision Latency (12w) │ Scalping Signal Types (12w)
Row 5 (y=36):  Scalping vs vLLM Latency (8w) │ Pipeline Latency (8w) │ Fast Path Ratio (8w)
```

### Panel Details

| Panel | Type | Metrics | Description |
|-------|------|---------|-------------|
| GPU Utilization | gauge | `gpu_utilization_pct` | NVIDIA GPU utilization %, yellow > 70%, red > 90% |
| VRAM Usage | bargauge | `gpu_vram_used_mb`, `gpu_vram_total_mb` | Used vs total VRAM, max=6144 (RTX 4050) |
| GPU Temperature | gauge | `gpu_temperature_celsius` | Temperature in °C, throttle risk > 85°C |
| GPU Power Draw | stat | `gpu_power_draw_watts` | Power consumption in watts |
| GPU Status Checks | stat | `gpu_status_checks_total{result}` | OK vs error count |
| VRAM Over Time | timeseries | `gpu_vram_used_mb`, `gpu_vram_total_mb` | VRAM trend — spikes = model loading |
| GPU Util + Temp | timeseries | `gpu_utilization_pct`, `gpu_temperature_celsius` | Combined utilization and thermal trend |
| vLLM Latency | timeseries | `vllm_inference_latency_ms` | Average and p95 latency per model |
| vLLM Tokens | timeseries | `vllm_tokens_generated_total` | Token generation rate (tokens/s) |
| vLLM Requests | timeseries | `vllm_requests_total` | Request rate by endpoint |
| vLLM Active | stat | `vllm_active_requests` | Concurrent inference requests |
| Scalping Latency | timeseries | `scalping_decision_latency_ms` | Avg, p95, p99 — target <5ms |
| Scalping Signals | timeseries | `scalping_signals_total` | Signal type detection rate |
| Scalping vs vLLM | bargauge | scalping avg vs vLLM avg | CPU vs GPU latency comparison |
| Pipeline Latency | timeseries | `pipeline_latency_ms` | Fast path vs heavy path total latency |
| Fast Path Ratio | gauge | `pipeline_fast_path_ratio` | % of runs taking scalping fast path |

### Template Variables

| Variable | Source | Description |
|----------|--------|-------------|
| `$gpu` | `label_values(gpu_utilization_pct, gpu)` | GPU name filter |
| `$model` | `label_values(vllm_inference_latency_ms, model)` | vLLM model filter |

## Prometheus Metrics

### GPU Metrics (from `gpu_optimizer.py`)

| Metric | Type | Labels | Source |
|--------|------|--------|--------|
| `gpu_utilization_pct` | Gauge | `gpu` | nvidia-smi query |
| `gpu_vram_used_mb` | Gauge | `gpu` | nvidia-smi query |
| `gpu_vram_total_mb` | Gauge | `gpu` | nvidia-smi query |
| `gpu_temperature_celsius` | Gauge | `gpu` | nvidia-smi query |
| `gpu_power_draw_watts` | Gauge | `gpu` | nvidia-smi query |
| `gpu_status_checks_total` | Counter | `result` | ok/error tracking |

**Export pattern:** All 5 GPU metrics + status counter exported in a single `GPU_PROMETHEUS_AVAILABLE` block inside `check_gpu_status()`. Graceful fallback if prometheus_client not installed.

### vLLM Inference Metrics (from `local_inference_server.py`)

| Metric | Type | Labels | Source |
|--------|------|--------|--------|
| `vllm_inference_latency_ms` | Histogram | `model` | `/generate` endpoint |
| `vllm_tokens_generated_total` | Counter | `model` | Token count per request |
| `vllm_requests_total` | Counter | `model`, `endpoint` | Request count (/generate, /generate/fast, /generate/stream) |
| `vllm_active_requests` | Gauge | — | Concurrent request tracker |

**Histogram buckets:** `[10, 25, 50, 100, 200, 500, 1000, 2000, 5000]` ms
**Active requests pattern:** `inc()` in try block, `dec()` in finally block.

### Scalping Metrics (from `scalping_engine.py`)

| Metric | Type | Labels | Source |
|--------|------|--------|--------|
| `scalping_decision_latency_ms` | Histogram | — | `decide()` timing |
| `scalping_signals_total` | Counter | `signal` | Signal type detection |

**Histogram buckets:** `[0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]` ms
**Signal types:** `momentum_breakout`, `rejection_support`, `rejection_resistance`, `volume_spike`, `trend_aligned`

### Pipeline Metrics (from `langgraph_pipeline.py`)

| Metric | Type | Labels | Source |
|--------|------|--------|--------|
| `pipeline_path_total` | Counter | `path` | Fast vs heavy path count |
| `pipeline_latency_ms` | Histogram | `path` | Total pipeline latency |
| `scalping_decisions_total` | Counter | `action` | BUY/SELL/HOLD distribution |

## Recording Rules

### GPU Rules (`gpu_metrics`, 30s interval)

| Rule | Expression |
|------|------------|
| `gpu_vram_usage_ratio` | `gpu_vram_used_mb / gpu_vram_total_mb` |
| `gpu_power_efficiency` | `gpu_utilization_pct / gpu_power_draw_watts` |
| `gpu_temp_avg_5m` | `avg_over_time(gpu_temperature_celsius[5m])` |

### Inference Rules (`inference_metrics`, 15s interval)

| Rule | Expression |
|------|------------|
| `vllm_avg_latency_fast_5m` | `rate(sum)/rate(count)` with NaN guard |
| `vllm_throughput_tps_5m` | `rate(vllm_tokens_generated_total[5m])` |
| `vllm_req_rate_5m` | `rate(vllm_requests_total[5m])` |

### Scalping Rules (`scalping_metrics`, 15s interval)

| Rule | Expression |
|------|------------|
| `scalping_avg_latency_5m` | `rate(sum)/rate(count)` with NaN guard |
| `scalping_signal_rate_5m` | `rate(scalping_signals_total[5m])` |
| `scalping_vs_vllm_latency_ratio` | `scalping_avg / vllm_avg` |

### Pipeline Rules (`pipeline_path_rules`, 15s interval)

| Rule | Expression |
|------|------------|
| `pipeline_fast_path_ratio` | `fast / (fast + heavy)` with NaN guard |
| `pipeline_avg_latency_fast_5m` | Fast path avg latency |
| `pipeline_avg_latency_heavy_5m` | Heavy path avg latency |
| `pipeline_throughput_1m` | `sum(rate(path_total[1m])) * 60` |

## Files

| File | Purpose |
|------|---------|
| `monitoring/grafana/dashboards/ultra-low-latency.json` | Grafana dashboard definition |
| `monitoring/recording_rules.yml` | Prometheus recording rules |
| `orchestrator/gpu_optimizer.py` | GPU metrics export |
| `orchestrator/local_inference_server.py` | vLLM metrics export |
| `orchestrator/scalping_engine.py` | Scalping metrics export |
| `orchestrator/langgraph_pipeline.py` | Pipeline metrics export |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Prometheus Scrape (15s)                    │
│                                                              │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │  nvidia  │  │  vLLM FastAPI│  │  ScalpingEngine       │  │
│  │  -smi    │  │  /metrics    │  │  prometheus_client     │  │
│  └────┬─────┘  └──────┬───────┘  └──────────┬────────────┘  │
│       │               │                     │                │
│       ▼               ▼                     ▼                │
│  gpu_*_mb         vllm_*_ms            scalping_*_ms        │
│  gpu_*_pct        vllm_*_total         scalping_*_total     │
│       │               │                     │                │
│       └───────┬───────┴─────────────────────┘                │
│               ▼                                              │
│  ┌─────────────────────┐                                    │
│  │  Recording Rules    │  ← Pre-computed ratios/latencies   │
│  │  (recording_rules.yml)│                                   │
│  └──────────┬──────────┘                                    │
│             ▼                                                │
│  ┌─────────────────────┐                                    │
│  │  Grafana Dashboard   │  ← 16 panels, 5s refresh          │
│  │  ultra-low-latency   │                                    │
│  └─────────────────────┘                                    │
└─────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

1. **Graceful fallback:** All metric imports use try/except — system works without prometheus_client
2. **Single export block:** GPU metrics consolidated into one `GPU_PROMETHEUS_AVAILABLE` check
3. **Active request tracking:** vLLM uses inc/dec pattern with try/finally for accurate concurrency
4. **NaN guards:** All recording rules use `or vector(0)` to prevent NaN in dashboards
5. **Label cardinality:** GPU label = fixed GPU name, model label = configured model (not user-provided)
6. **Histogram buckets:** Tuned for expected latency ranges — scalping <5ms, vLLM 50-500ms
