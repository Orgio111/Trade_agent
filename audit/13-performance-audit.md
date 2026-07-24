# Performance Audit

## Evidence boundary

No production load or destructive benchmark was run. The observed stack had no
end-to-end signals, so p50/p95/p99 trading latency, throughput, queue lag, fill
latency, and sustained capacity cannot be truthfully calculated.

Point-in-time observations:

- vLLM consumed approximately 98% CPU while repeatedly failing initialization.
- The MoE service consumed approximately 41% CPU and about 546 MB RAM.
- Several health requests completed between roughly 12 and 600 ms; frontend root
  was slower in isolated samples. These are not statistically valid benchmarks.
- Current source gives most services finite CPU/RAM limits, but no PID limits.
- The host had approximately 8.17 GiB RAM, while multiple databases, services, and
  a GPU model server were co-located.
- Docker stored 62.54 GB of images and 12.53 GB of fully reclaimable build cache.

## Bottlenecks

1. GPU admission failure creates a hot crash loop.
2. Market reconnect storms create log/CPU/network pressure without useful data.
3. Python swarm debates are documented at multi-second latency and are unsuitable
   for a strict one-minute close path without a hard deadline.
4. No queue-lag/backpressure evidence exists for NATS consumers.
5. No DB index/query-plan or connection-pool soak has been performed on canonical
   tables.
6. Multiple inactive services consume resources in the legacy topology.

## Required benchmark design

Use deterministic replay rather than live orders:

- 1, 10, and 100 symbols at 1m/5m equivalent event rates.
- Burst/reconnect/duplicate/out-of-order scenarios.
- p50/p95/p99 for validation, features, candidate, risk, intent, simulated fill,
  reconciliation.
- NATS pending/ack latency, DB transaction time, outbox age, CPU/RAM/disk growth.
- Model cold/warm latency and timeout/fallback rate.
- 24-hour normal soak plus a seven-day paper soak before autonomous consideration.

Success must be tied to an explicit candle-close deadline and safety behavior under
deadline miss. Speed may never bypass risk checks.

## Performance verdict

Performance readiness is **unverified**. Resource limits and replay tests are a
useful base, but the current crash/retry loops invalidate capacity inference.

