---
title: NATS JetStream Event System
type: concept
tags: [nats, event-system, jetstream, durability, idempotency, local-first]
created: 2026-06-30
updated: 2026-07-16
sources:
  - "[[canonical-local-paper-runtime-v1]]"
  - "[[deterministic-paper-core-v1]]"
status: stable
---

# NATS JetStream Event System

## Definition

The canonical runtime uses one local, file-backed JetStream named `QUANTEX_CORE`. It transports exact v1 byte envelopes between independently owned workers. NATS owns delivery and bounded redelivery; domain parsing, risk authorization, PostgreSQL transactions, and paper execution remain outside the bus.

## Canonical subjects

| Subject | Producer | Consumer | Meaning |
|---|---|---|---|
| `market.raw.v1` | External/local raw producer | `market-data-worker` | Untrusted canonical `MarketEvent` input |
| `market.validated.v1` | `market-data-worker` | Future candidate producer | Contract- and quality-approved market event |
| `market.rejected.v1` | `market-data-worker` | Observability/replay tooling | Payload-free rejection identity and reason codes |
| `signals.candidate.v1` | Future candidate producer | `decision-worker` | Ollama-provenance-bound, non-authoritative candidate |
| `risk.approved.v1` | `decision-worker` | `execution-worker` | Persisted deterministic approval |
| `risk.rejected.v1` | `decision-worker` | Observability/replay tooling | Persisted deterministic rejection |
| `orders.intent.v1` | `execution-worker` | Observability/projectors | Persisted deterministic paper intent envelope |
| `orders.updated.v1` | `execution-worker` | Observability/projectors | Paper order/fill result envelope |

The stream owns the exact eight subjects. Startup fails if an existing stream's name, subjects, retention, storage, message size, duplicate window, or resource bounds drift.

## Stream bounds

| Setting | Value |
|---|---:|
| Storage | file |
| Retention | limits |
| Maximum age | 7 days |
| Maximum bytes | 512 MiB |
| Maximum message | 1 MiB |
| Maximum consumers | 32 |
| Replicas | 1 |
| Server duplicate window | 120 seconds |

This is a single-node local runtime. It favors bounded laptop operation and deterministic restart behavior over high availability.

## Delivery contract

Canonical consumers use:

- durable names with an exact filter subject;
- `DeliverPolicy.ALL` and explicit manual ACK;
- bounded ACK wait, pending count, payload size, and delivery attempts;
- exponential NAK delay for retryable handler failures;
- terminal settlement after the configured delivery limit;
- sanitized failure metadata that never copies message payloads.

ACK occurs only after the handler completes, including its downstream durable publication. Producer publications require a deterministic `Nats-Msg-Id`; duplicate server acknowledgements are observable.

## What this does not guarantee

JetStream remains at-least-once transport. The current worker callback flow does not atomically combine:

1. inbox admission;
2. PostgreSQL domain mutation;
3. outbox publication;
4. input ACK.

Migration `003_runtime_durability.sql` creates `event_inbox`, `event_outbox`, and `consumer_offsets`, but no worker wires them yet. A crash between a domain commit and output publication/ACK can therefore replay work. Deterministic event IDs, unique database bindings, exact decision lookup, intent IDs, and `Nats-Msg-Id` reduce duplicate effects, but they are not an end-to-end exactly-once claim.

Terminal failures are currently `term`-settled after notification. There is no durable dead-letter subject/table publisher or operator replay command yet.

## Envelope integrity

- Market events carry deterministic provenance-bound IDs and payload checksums.
- Candidate events bind the exact market event, candidate content, Ollama role/model/digest, trace, and timestamp.
- Risk events bind one candidate event and market event to one immutable verdict.
- Order-intent events bind the exact persisted approval and deterministic intent.
- Order-update events bind the risk event, intent event, order, and every fill.

Tampered or cross-mode envelopes fail validation and do not grant authority. See [[deterministic-paper-core-v1]].

## Legacy event quarantine

The historical `orchestrator/events.py`, hierarchical `signals.raw.*`, Go aggregation subjects, ACP v2 events, WebSocket subjects, and Rust consumers remain pre-canonical. They do not share the canonical v1 worker contracts and are started only through legacy-profile services. They may be migrated only through a language-neutral schema, golden fixtures, replay compatibility, and manifest ownership.

## Strengths and limitations

| Strength | Limitation |
|---|---|
| Exact subject authority and startup drift detection | Single NATS node; no high availability |
| Explicit ACK and bounded redelivery | Transactional inbox/outbox not wired |
| Deterministic producer deduplication IDs | Server dedupe window is finite |
| Domain authority stays outside transport | No durable DLQ/replay workflow |

## Related

- [[canonical-local-paper-runtime-v1]]
- [[deterministic-paper-core-v1]]
- [[infrastructure-overview]]
- [[multi-agent-pipeline]]
- [[trade-project-full-integration-build-plan]]
