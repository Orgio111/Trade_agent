---
title: Canonical Local Paper Runtime v1
type: decision
tags:
  - architecture
  - local-first
  - paper-trading
  - docker-compose
  - nats
  - postgresql
  - ollama
created: 2026-07-16
updated: 2026-07-16
sources:
  - "[[trade-project-full-integration-build-plan]]"
  - "[[deterministic-paper-core-v1]]"
  - "[[nats-event-system]]"
  - "[[local-trading-ai-architecture]]"
status: stable
---

# Canonical Local Paper Runtime v1

> **2026-07-24 runtime update:** [[production-remediation-2026-07-24]]
> verifies the evolved 12-service graph, authoritative bootstrap, real public
> producer, feature/candidate workers, transactional outbox path, reconciliation,
> encrypted restore drill, and real-public-data paper fills. Historical nine-
> service tables below describe the v1 starting point.

## Context

The repository contained a verified deterministic paper/replay core, but the default deployment still exposed a larger pre-canonical topology with overlapping orchestration, inference, risk, and execution paths. That made the safe reference implementation optional while legacy services appeared authoritative. [[trade-project-full-integration-build-plan]] requires one default runtime with explicit ownership, durable boundaries, and no live or cloud path.

## Options considered

1. Keep the large Compose topology as the default and document which routes are safe.
2. Create a second Compose file for the canonical core while leaving the existing default unchanged.
3. Make the canonical local paper runtime the default and quarantine pre-canonical services behind an explicit profile.

## Decision

Choose option 3. Default `docker compose up` starts only local infrastructure and the canonical Python runtime:

| Service | Primary responsibility |
|---|---|
| `postgres` | Authoritative migrations, risk decisions, snapshots, constraints, intents, orders, and fills |
| `migrate` | Apply checksum-locked migrations before runtime services start |
| `nats` | File-backed JetStream transport for canonical v1 subjects |
| `market-data-worker` | Validate raw `MarketEvent` envelopes and publish accepted or rejected outcomes |
| `decision-worker` | Evaluate model-produced candidates with deterministic hard-rule risk against PostgreSQL state |
| `execution-worker` | Execute exact durable approvals through `PaperBroker` only and persist results |
| `control-plane` | Expose read-only liveness, readiness, and runtime status |
| `redis` | Local disposable state infrastructure; not authoritative in this hot path |
| `chroma` | Local project/runtime knowledge projection; not trading authority |

Ollama remains a host-local dependency rather than a Compose inference service. All pre-canonical orchestration, cloud inference, vLLM, Go/Rust runtime, dashboards, and monitoring services require the explicit `legacy` profile. Profile availability does not promote them into the canonical authority path.

The control plane and three workers share `infra/workers/Dockerfile`. Its frozen `workers` dependency group contains only the FastAPI, NATS, PostgreSQL, HTTP, and validation libraries required by the canonical runtime; it does not install the repository's forecasting, ML, broker-research, or legacy orchestration dependency surface. The image runs as the non-root `quantex` user.

## Canonical event flow

```text
external/local raw producer
  -> market.raw.v1
  -> market-data-worker
     -> market.validated.v1 | market.rejected.v1

separate candidate producer (not yet supplied by the default stack)
  -> signals.candidate.v1
  -> decision-worker
     -> risk.approved.v1 | risk.rejected.v1
  -> execution-worker (approved subject only)
     -> orders.intent.v1
     -> orders.updated.v1
```

The runtime deliberately has no implicit `market.validated.v1 -> signals.candidate.v1` transformation. A future candidate producer must be separately owned, model-provenance bound, and unable to bypass deterministic risk.

## Local model boundary

One immutable registry maps roles to the only approved Ollama model names:

| Role | Exact model |
|---|---|
| reasoning | `qwen3:8b` |
| fast | `phi3:3.8b` |
| research | `deepseek-r1:8b` |
| vision | `moondream` |
| tool formatting | `mistral` |
| embedding | `nomic-embed-text` |

Candidate and risk-decision envelopes retain `provider=ollama`, the exact role/model mapping, and a required 64-character model digest. Only the reasoning and fast roles may produce a candidate. The current contracts validate digest form and preserve it end to end; the missing candidate producer must capture and verify the digest against the local Ollama inventory. Provenance never grants risk or execution authority. No cloud provider is admitted.

## Durable authority and transport semantics

Migration `003_runtime_durability.sql` adds:

- event inbox, outbox, and consumer-offset tables as future durable processing scaffolding;
- immutable portfolio snapshots and versioned instrument constraints for risk input;
- candidate-event binding on signals and one immutable risk verdict per signal;
- derived position and tax-lot projection scaffolding tied to fills;
- append-only guards for signals, risk decisions, portfolio snapshots, and instrument constraints.

The decision worker loads exactly one active policy, a fresh reconciled portfolio snapshot, and one effective constraint version. It rejects missing, stale, ambiguous, or identity-mismatched inputs. The candidate and verdict are recorded atomically; redelivery reuses the first verdict only when signal, candidate hash, market event, and candidate event all match.

JetStream uses one bounded file-backed `QUANTEX_CORE` stream with exact subject ownership. Consumers use explicit manual ACK, bounded redelivery, NAK backoff, and terminal settlement after the delivery limit. Producers require deterministic `Nats-Msg-Id` values for server-window deduplication. These controls provide at-least-once transport with idempotent identities, not end-to-end exactly-once processing.

## Control-plane boundary

The FastAPI control plane exposes only:

- `GET /health`
- `GET /ready`
- `GET /api/v1/runtime`

It checks PostgreSQL migration/kill-switch state, NATS reachability, and the exact local Ollama registry. Readiness fails closed when a dependency is unavailable. `execution_enabled` is true only when all probes pass and the durable kill switch is explicitly inactive. The API cannot toggle the kill switch, create candidates, approve risk, place orders, or mutate broker state.

## Safety invariants

- Runtime modes are limited to `paper_live` and `replay`.
- `execution-worker` imports and accepts only `PaperBroker`; no live broker path is present.
- A risk approval must match the exact persisted decision before an intent can be created.
- Model output remains untrusted candidate input; deterministic risk owns authorization.
- Intent and result envelopes bind the exact risk event, decision, intent, order, and fills.
- Legacy mutation routes are not an alternate authority surface.
- This change does not modify any strategy or trading decision logic.

## Known gaps

1. The default stack has no candidate producer, so validated market events cannot yet become candidates.
   That producer must also bind the candidate digest to the model inventory observed at inference time.
2. Migrations intentionally seed no active risk policy, portfolio snapshot, instrument constraints, or inactive kill-switch row. The decision path therefore fails closed until an explicit bootstrap process exists.
3. The event inbox, outbox, and consumer-offset tables are scaffolding; worker callbacks do not yet wire ACK to a transactional inbox/outbox boundary.
4. Terminal JetStream failures are settled but are not copied to a durable dead-letter subject/table with an operator replay workflow.
5. Position/tax-lot projections are schema only; no canonical projector currently maintains them.
6. Protective stop/OCO lifecycle, startup reconciliation, immutable raw capture, and promotion-grade shadow/live gates remain incomplete.

## Rationale

Making the safe path the default removes deployment ambiguity without deleting legacy research. Exact event identities, deterministic risk, durable PostgreSQL authority, and paper-only execution preserve the local model boundary while allowing each missing capability to be added behind a testable contract.

## Consequences

- Operators receive a small, local-first default stack that cannot place live orders.
- The default runtime is intentionally not end-to-end productive until candidate generation and authoritative bootstrap inputs are implemented.
- Legacy services remain available for controlled investigation but must not be treated as production authority.
- `GET /ready` describes dependency readiness, not evidence that a complete candidate-to-fill loop is available.

## Revisit trigger

Revisit this decision when transactional inbox/outbox processing, durable dead-letter replay, seeded-and-audited risk inputs, a separately owned candidate producer, and restart reconciliation all pass fault-injection tests. Live execution still requires a separate explicit ADR and human authorization.

## Related

- [[deterministic-paper-core-v1]]
- [[infrastructure-overview]]
- [[nats-event-system]]
- [[local-trading-ai-architecture]]
- [[local-ai-deployment-guide]]
- [[multi-agent-pipeline]]
