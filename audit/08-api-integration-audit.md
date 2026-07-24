# API and Integration Audit

## Integration inventory

| Integration | Intended use | Observed state | Authentication handling | Verdict |
|---|---|---|---|---|
| Binance public WebSocket | Candles/market data | Repeated HTTP 404 reconnects | Public | Broken in observed runtime |
| Binance testnet user stream | Execution/account events | HTTP 410 and DNS failures | Credentials present but not displayed | Broken/unverified |
| NATS JetStream | Event backbone | Healthy server; legacy zero-message stream | Internal network | Service up, application disconnected |
| Ollama | Canonical candidate generation | Canonical worker not deployed | Local | Source implemented, runtime unverified |
| vLLM | Local OpenAI-compatible inference | Crash-loop/connection reset | Local | Broken |
| Cloud LLM providers | Optional inference | Real secret names present | Shared environment | Not externally validated |
| Qdrant | Legacy trade memory | Green, zero points | Internal | Real but unused |
| Chroma | Knowledge/runtime memory | 1,066 knowledge; zero runtime | No compensating auth verified | Working knowledge path, security blocked |
| Prometheus | Metrics collection | Four targets up | Internal/host-exposed | Partial |
| Grafana | Monitoring UI | Healthy HTTP, plugin errors | Password configured | Partial |

No external secret-bearing validation calls were made. A real credential's presence
does not prove validity or authorization.

## [AUD-007] Market-data integration is broken and reports misleading health

- Severity: High
- Category: External API / data integrity
- Component: Legacy market data service
- File: `services/market_data/`, legacy Compose configuration
- Line: WebSocket URL/reconnect and status response model
- Runtime service: `quantex-market-data`
- Status: Confirmed
- Evidence: Logs repeatedly show Binance WebSocket HTTP 404; `/api/v1/status` returns HTTP 500 due serialization; downstream feature/order-book symbol sets were empty; NATS had no signal traffic.
- Impact: Trading decisions would use no data, stale data, or fallback/synthetic paths while container liveness remains green.
- Root cause: Endpoint/protocol drift, duplicated feed loops, and health probes that do not assert freshness or downstream delivery.
- Reproduction: Read the sanitized market-data log tail, call status, and inspect downstream symbol/NATS state.
- Recommended fix: Replace the legacy feed with the canonical public market producer; validate endpoint and closed-candle semantics; enforce sequence/freshness/duplicate checks; cap reconnects with jitter; make readiness require a recent validated event and active consumer.
- Validation after fix: A paper-only supervised test records ordered real exchange event IDs/timestamps through raw, validated, features, candidate, risk, and fill tables with freshness SLOs.
- Estimated effort: M
- Priority: P0

## API correctness

- The canonical control plane exposes only read-only status routes, which is the
  appropriate production direction.
- The running legacy orchestrator has 62 OpenAPI routes and no security scheme.
- Several services return 200 from shallow health routes despite disconnected
  dependencies or empty application state.
- Dashboard health configuration targets `/health`, while the route is
  `/api/health`.
- Frontend `/dashboard/api/health` returned timeout/502 during probes.
- Go realtime was internally healthy but had no signals and was not published on
  the expected host port in the observed stack.

## WebSocket/SSE and background jobs

WebSocket code exists for market feeds and frontend updates. Runtime verification
showed reconnect storms on the exchange side and no aggregated downstream signal.
Long-running worker loops exist in both canonical and legacy paths; no independent
production scheduler/cron or failed-job operator workflow was verified. Canonical
DLQ replay is implemented as an explicit script, which is safer than automatic
unbounded replay.

## Retry assessment

The canonical producer uses bounded backoff with jitter, but the observed legacy
market and vLLM paths exhibited repeated failures without converging. Retry budgets,
circuit-breaker state, and alert thresholds must be part of readiness, not merely
log output.

