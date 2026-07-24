# Observability Audit

Observability score: **25/100**.

## [AUD-010] Prometheus configuration is invalid and health endpoints are false-positive

- Severity: High
- Category: Monitoring correctness
- Component: Prometheus and service health
- File: `monitoring/prometheus.yml`, Compose mounts, service health routes
- Line: Rule-file declarations and health handlers
- Runtime service: `quantex-prometheus` and legacy services
- Status: Confirmed
- Evidence: `promtool check config` fails because `/etc/prometheus/recording_rules.yml` is referenced but absent; four targets report Up; orchestrator status reports services true while brains/NATS are disconnected; inference reports healthy while vLLM fails; dashboard probe path is wrong.
- Impact: Operators receive green signals during application failure and may enable or leave a broken system unattended.
- Root cause: Liveness, dependency readiness, business readiness, and release readiness are conflated; configuration is not validated before start.
- Reproduction: Run `promtool check config`; compare shallow health with NATS, brain, provider, freshness, and DB schema state.
- Recommended fix: Fail Prometheus startup/CI on invalid config; define separate liveness/readiness/business SLO endpoints; readiness must include schema, leases, stream/consumer, data freshness, provider state, reconciliation, and kill switch.
- Validation after fix: Inject each dependency failure and assert NotReady plus a routed alert with bounded detection time.
- Estimated effort: M
- Priority: P0

## [AUD-020] Telemetry coverage is insufficient for end-to-end trading safety

- Severity: Medium
- Category: Observability coverage
- Component: Metrics, logs, traces, alerts
- File: `monitoring/`, event definitions, service instrumentation
- Line: Multiple
- Runtime service: Entire stack
- Status: Confirmed
- Evidence: Only four Prometheus targets were scraped; no verified alerts; no end-to-end trace was observed despite canonical trace IDs; no metrics proved event age, consumer lag, rejection reason, fallback rate, reconciliation mismatch, fill latency, or kill-switch transitions.
- Impact: Failures can remain silent, and incident reconstruction cannot prove why an order did or did not occur.
- Root cause: Infrastructure metrics were implemented before a trading safety SLO/error-budget model.
- Reproduction: Query Prometheus targets/rules and search dashboards for the listed business invariants.
- Recommended fix: Define golden signals per event stage; propagate immutable trace IDs; add structured/redacted logs; alert on freshness, lag, DLQ, lease expiry, reconciliation, restart rate, disk, and unauthorized access.
- Validation after fix: A chaos/replay test generates one correlated trace from source event to paper fill/rejection and every injected fault creates the expected alert.
- Estimated effort: L
- Priority: P1

## Existing positive controls

- Canonical events include trace and context fields.
- Prometheus, Grafana, and NATS exporter are provisioned.
- Four endpoints were scrapeable.
- Container logs include actionable component errors.

These controls are foundations, not 24/7 operational evidence.

