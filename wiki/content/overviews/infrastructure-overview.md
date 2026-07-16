---
title: Infrastructure Overview
type: overview
tags:
  - infrastructure
  - docker
  - deployment
  - local-first
  - paper-trading
created: 2026-06-30
updated: 2026-07-16
sources:
  - "[[canonical-local-paper-runtime-v1]]"
status: stable
---

# Infrastructure Overview

The canonical deployment is a local Docker Compose paper/replay runtime. PostgreSQL and NATS own durable trading state and transport; Ollama is host-local; the FastAPI control plane is read-only. Earlier Kubernetes, Go/Rust, cloud/vLLM, dashboard, and monitoring paths are retained only as legacy research infrastructure.

## Default topology

```text
host-local Ollama
       |
       v
+---------------+       +------------------+
| control-plane | ----> | PostgreSQL       |
| read-only API | ----> | NATS JetStream   |
+---------------+       +------------------+
                               |
market.raw.v1                  |
       |                       |
       v                       v
+--------------------+  +-----------------+  +------------------+
| market-data-worker |  | decision-worker |  | execution-worker |
| validate/reject    |  | hard-rule risk  |  | PaperBroker only |
+--------------------+  +-----------------+  +------------------+

Auxiliary local services: Redis (disposable state) and ChromaDB
(knowledge projection, never trading authority).
```

## Default Compose services

| Service | Exposure | Runtime role | Authority |
|---|---|---|---|
| `postgres` | loopback `5432` | Execution/risk ledger and migrations | Authoritative durable state |
| `migrate` | none | One-shot checksum-locked migration application | Schema owner |
| `nats` | loopback `4222`, `8222` | File-backed `QUANTEX_CORE` JetStream | Canonical event transport |
| `market-data-worker` | none | Raw market contract and quality gate | Market validation only |
| `decision-worker` | none | Candidate verification and deterministic risk | Risk authorization only |
| `execution-worker` | none | Exact approval to deterministic paper fill | Paper execution only |
| `control-plane` | loopback `8001` | Health/readiness/runtime status | Read-only observation |
| `redis` | loopback `6379` | Disposable local state infrastructure | Non-authoritative |
| `chroma` | loopback `8100` | Local knowledge vector projection | Non-authoritative |

The runtime containers share one non-root Python image. Its frozen `workers` dependency group installs only the canonical FastAPI/NATS/PostgreSQL runtime surface, keeping legacy ML and research packages outside the default image. Migrations complete before the control plane and workers start. No migration directory is mounted into the PostgreSQL entrypoint, which avoids accidental reapplication outside the checksum-locked runner.

## Locality and secrets

- Published infrastructure ports bind to `127.0.0.1`.
- Ollama endpoints admit only loopback or `host.docker.internal` HTTP URLs with no credentials, query, or path components.
- PostgreSQL credentials are injected at runtime; they are not copied into documentation or source defaults.
- Worker settings admit only `paper_live` and `replay`, do not load dotenv files, and sanitize public configuration summaries.
- The exact approved Ollama registry is documented in [[local-trading-ai-architecture]].

## Safe startup

```powershell
# Set POSTGRES_PASSWORD in the current local process or approved secret manager.
docker compose config --services
docker compose up -d
docker compose ps

Invoke-RestMethod http://127.0.0.1:8001/health
Invoke-RestMethod http://127.0.0.1:8001/ready
Invoke-RestMethod http://127.0.0.1:8001/api/v1/runtime
```

The default service list must contain only the nine services above. Do not add `--profile legacy` to the canonical startup. See [[local-ai-deployment-guide]] for model and readiness checks.

## Legacy profile quarantine

The `legacy` profile contains pre-canonical services, including overlapping orchestrators, market/feature/order-book services, inference routers, cloud/vLLM paths, legacy risk/execution, Go realtime aggregation, frontend/proxy, and observability. These services are preserved for audit and controlled migration; they are not part of the default authority graph.

Rules:

- Starting the legacy profile is an explicit engineering investigation, not a production deployment.
- Legacy mutation routes do not authorize paper or live execution in the canonical core.
- Cloud inference is default-off and cannot satisfy local-model provenance.
- Kubernetes and Terraform manifests describe the pre-canonical topology and are not a deployment target for [[canonical-local-paper-runtime-v1]].
- A legacy component can move into the default only through manifest ownership, canonical contracts, tests, resource evidence, and an ADR.

## Persistence

| Volume | Purpose | Recovery expectation |
|---|---|---|
| `postgres_data` | Risk/execution authority and schema history | Required for deterministic recovery |
| `nats_data` | JetStream messages and consumer state | Required for bounded redelivery |
| `redis_data` | Disposable cache/state | Rebuildable |
| `chroma_data` | Local knowledge projection | Rebuildable from governed sources |

Legacy volumes remain defined for profiled services but are not active default dependencies.

## Current operational gaps

- No default candidate producer connects validated market events to `signals.candidate.v1`.
- No bootstrap command seeds an audited active risk policy, reconciled portfolio snapshot, effective instrument constraints, or inactive kill-switch state.
- Inbox/outbox/offset schemas exist, but ACK and publication are not yet transactionally wired to them.
- There is no durable dead-letter replay service.
- Position/tax-lot projectors and startup reconciliation are incomplete.
- Protective stop/OCO and live-promotion infrastructure remain outside the runtime.

## Related

- [[canonical-local-paper-runtime-v1]] — authoritative default-runtime ADR
- [[deterministic-paper-core-v1]] — deterministic domain and paper execution semantics
- [[nats-event-system]] — canonical subject and delivery contract
- [[local-trading-ai-architecture]] — Ollama registry and authority boundary
- [[trade-project-full-integration-build-plan]] — phased migration and promotion gates
