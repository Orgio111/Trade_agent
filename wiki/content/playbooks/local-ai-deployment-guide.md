---
title: Local AI Deployment Guide
type: playbook
tags: [deployment, ollama, docker-compose, local-first, paper-trading]
created: 2026-06-30
updated: 2026-07-16
sources:
  - "[[canonical-local-paper-runtime-v1]]"
  - "[[local-trading-ai-architecture]]"
  - "[[infrastructure-overview]]"
status: stable
---

# Local AI Deployment Guide

## Purpose

Start and inspect the canonical local paper/replay runtime. This playbook never enables live trading, never starts a cloud inference provider, and never uses the legacy profile.

## Prerequisites

- Docker Desktop or Docker Engine with Compose v2.
- Host-local Ollama reachable at `http://127.0.0.1:11434`.
- Sufficient local disk for PostgreSQL, NATS, ChromaDB, and the approved models.
- A PostgreSQL password supplied through the current process or an approved local secret manager.
- No live broker key is required or used.

## 1. Install the exact local models

```powershell
ollama pull qwen3:8b
ollama pull phi3:3.8b
ollama pull deepseek-r1:8b
ollama pull moondream
ollama pull mistral
ollama pull nomic-embed-text

ollama list
```

Do not substitute cloud models or arbitrary aliases. The runtime registry and provenance contract are defined in [[local-trading-ai-architecture]]. On a 6 GB RTX 4050, keep concurrency conservative and benchmark actual warm/cold behavior locally.

## 2. Supply local configuration

Required:

- `POSTGRES_PASSWORD`

Optional safe overrides include `POSTGRES_PORT`, `REDIS_PORT`, `NATS_PORT`, `NATS_MONITOR_PORT`, `CONTROL_PLANE_PORT`, and `PAPER_ACCOUNT_ID`. The worker mode is `paper_live`; no live mode is admitted.

Keep secrets outside source control and do not print them into logs. Worker configuration does not load dotenv files on its own.
Variables used only by quarantined `legacy` services are not required to render or start the default graph. The legacy Grafana service independently fails at startup unless `GRAFANA_ADMIN_PASSWORD` is supplied.

## 3. Verify the default boundary

```powershell
docker compose config --services
```

Expected default services:

```text
postgres
redis
migrate
chroma
nats
control-plane
market-data-worker
decision-worker
execution-worker
```

If pre-canonical services appear without an explicit profile, stop and treat it as deployment drift. Do not use `--profile legacy` for this playbook.

## 4. Start the runtime

```powershell
docker compose up -d
docker compose ps
```

The one-shot migration service must complete successfully before the control plane and workers start. PostgreSQL, Redis, NATS, ChromaDB, and the control plane publish loopback ports only.

## 5. Inspect health and readiness

```powershell
Invoke-RestMethod http://127.0.0.1:8001/health
Invoke-RestMethod http://127.0.0.1:8001/ready
Invoke-RestMethod http://127.0.0.1:8001/api/v1/runtime
Invoke-RestMethod http://127.0.0.1:8222/healthz
```

Interpretation:

- `/health` proves only that the read-only FastAPI process is alive.
- `/ready` checks PostgreSQL, NATS, and the exact Ollama registry; it returns HTTP 503 when a dependency is unavailable.
- `execution_enabled=false` is correct while the kill switch is active or uninitialized.
- Even `ready=true` does not mean the full candidate-to-fill loop is available.

The API has no mutation endpoint. It cannot seed risk inputs or disable the kill switch.

## 6. Understand the intentional fail-closed state

The default runtime currently seeds none of the following:

- an active risk policy;
- a reconciled portfolio snapshot;
- effective instrument constraints;
- an inactive kill-switch row;
- a candidate producer.

Therefore the runtime can validate raw market envelopes, but it cannot autonomously produce an executable candidate. Any candidate submitted without all authoritative risk inputs fails closed. This is an implementation boundary, not an operator error.

## 7. Observe without mutating

```powershell
docker compose logs --tail 100 control-plane
docker compose logs --tail 100 market-data-worker
docker compose logs --tail 100 decision-worker
docker compose logs --tail 100 execution-worker
```

Canonical worker logs sanitize failure types and do not copy rejected NATS payloads. Avoid dumping environment variables or database connection strings during troubleshooting.

## 8. Stop safely

```powershell
docker compose down
```

This preserves PostgreSQL, NATS, Redis, and Chroma volumes. Removing volumes destroys durable local state and requires explicit operator intent; it is not part of routine shutdown.

## Verification checklist

- [ ] `docker compose config --services` contains only the nine default services.
- [ ] Ollama lists all six exact approved models.
- [ ] `migrate` completed successfully.
- [ ] `/health` responds on loopback.
- [ ] `/ready` reports dependency details without secrets.
- [ ] `execution_enabled` remains false until durable state is explicitly initialized.
- [ ] No live broker or cloud inference service is running.
- [ ] No legacy profile was started.

## Known gaps and next operations

Do not improvise database seed rows manually. The next required deployment feature is a tested bootstrap command that validates and records policy, portfolio, constraints, and kill-switch state with provenance and audit. Transactional inbox/outbox wiring and durable dead-letter replay must follow before reliability claims are raised.

## Related

- [[canonical-local-paper-runtime-v1]]
- [[infrastructure-overview]]
- [[local-trading-ai-architecture]]
- [[deterministic-paper-core-v1]]
- [[nats-event-system]]
