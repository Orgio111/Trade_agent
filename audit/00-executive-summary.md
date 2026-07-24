# Executive Summary

Audit date: 2026-07-24 (Asia/Ulaanbaatar)  
Scope: complete repository, Git history indicators, dependency manifests, Docker
source and observed runtime, databases, APIs, trading/AI paths, tests, security,
observability, reliability, and production operations.  
Safety mode: read-only; paper/testnet only; no secret values retained.

## Project purpose and actual architecture

Trade_agent is a multi-language algorithmic-trading platform with a canonical
Python/NATS/PostgreSQL worker pipeline, a TypeScript dashboard, Go realtime
aggregation, legacy service APIs, local/cloud inference adapters, paper execution,
and a durable knowledge base.

The repository now contains a materially improved canonical paper-runtime design:
versioned events, inbox/outbox, leases, deterministic risk, reconciliation,
kill-switch state, migration roles, and a NATS recovery acceptance harness.
However, the 23 running containers were an older legacy topology. None of the
canonical workers was running, the canonical database schema was absent, the
legacy NATS stream had zero messages/consumers, market data was failing, and the
model runtime was crash-looping. Source capability and deployed reality therefore
diverge.

## Current operating state

Working at the audit point:

- Docker, NATS liveness, PostgreSQL connectivity, Redis, Qdrant, Chroma project
  knowledge, nginx/frontend, and several HTTP service liveness endpoints.
- 604 of 605 Python tests; canonical Ruff and MyPy; frontend lint/typecheck;
  npm audit; Go compile/vet.
- The isolated committed recovery artifact demonstrates NATS readiness failure and
  recovery for the canonical topology.

Not working or not production-verifiable:

- Live market ingestion, active brain publication, canonical NATS stream,
  market-to-risk-to-order-to-fill flow, local vLLM inference, canonical DB
  migrations/roles, Prometheus configuration, full E2E tests, backup/restore,
  alerting, Rust verification, and production authentication.

Simulated or mock:

- Paper broker and FIX simulator; synthetic/backtest generators; feature/rule
  fallbacks; selected agent/model fallbacks; no live-broker approval.

## Scores

| Measure | Score | Evidence basis |
|---|---:|---|
| Production readiness | **33/100** | Weighted score in [`15-production-readiness.md`](15-production-readiness.md) |
| Security | **20/100** | Critical unauthenticated surface, Chroma advisory, secret sprawl, hardening gaps |
| 24/7 reliability | **30/100** | Recovery harness exists; observed runtime has crash/retry loops and no DR |
| Code quality | **45/100** | Canonical gates pass; legacy full-repo gates fail materially |
| Test readiness | **50/100** | Broad suite, but one regression, no coverage, E2E timeout, Rust unverified |
| Observability | **25/100** | Four scrape targets up; config invalid and health semantics misleading |
| Docker/runtime | **25/100** | Canonical Compose improved; running deployment stale and insecurely exposed |
| Real-data integrity | **18/100** | Public market producer exists; observed real feed and end-to-end flow broken |

Findings: **3 Critical, 9 High, 10 Medium, 4 Low**.

## Top ten 24/7 blockers

1. Deployed runtime does not match the canonical worker architecture.
2. No verified real market-data-to-fill path; NATS carries no signals.
3. Unauthenticated state-changing legacy API routes are host-exposed.
4. Chroma 1.5.9 has a critical pre-auth network code-injection advisory.
5. Local vLLM is in a resource-driven restart loop.
6. Market-data WebSocket reconnects on HTTP 404 and status serialization fails.
7. Runtime PostgreSQL lacks canonical migrations and uses one superuser login.
8. No verified backup, restore, RPO, or RTO mechanism.
9. Prometheus configuration is invalid and service health is semantically false.
10. CI/test/governance gates are not green on the audited commit.

## Immediate P0 actions

1. Isolate the current runtime from non-loopback/public access and disable
   unauthenticated mutation routes.
2. Stop treating the 23-container legacy stack as a candidate production release;
   deploy the canonical paper topology only after validation.
3. Remove or network-isolate Chroma 1.5.9 until an upstream fix or safe replacement
   exists.
4. Rotate/reissue the historically committed TLS key/certificate and rotate all
   credentials whose exposure cannot be disproved; do not print them.
5. Keep live execution disabled. Require a supervised, paper-only market-to-fill
   acceptance run with deterministic risk and kill-switch tests.

## Decision

```text
PRODUCTION READY: NO
24/7 AUTONOMOUS OPERATION: NO
REAL DATA VERIFIED: PARTIAL
SECURITY APPROVED: NO
```

Production deployment is rejected. The canonical source is a viable remediation
base, but it is not the deployed system and has not demonstrated an order/fill
path. See [`16-remediation-backlog.md`](16-remediation-backlog.md).

