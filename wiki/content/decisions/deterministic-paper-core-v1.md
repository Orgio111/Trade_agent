---
title: Deterministic Paper Core v1
type: decision
tags:
  - architecture
  - deterministic-systems
  - risk
  - execution
  - replay
  - paper-trading
created: 2026-07-15
updated: 2026-07-16
sources:
  - "[[trade-project-full-integration-build-plan]]"
  - "[[canonical-local-paper-runtime-v1]]"
  - "[[nats-event-system]]"
  - "[[broker-abstraction-layer]]"
status: stable
---

# Deterministic Paper Core v1

## Context

The repository previously had disconnected event, risk, execution, and backtest paths. None supplied one reproducible market-event-to-fill trace with a fail-closed authorization boundary. [[trade-project-full-integration-build-plan]] therefore made a deterministic paper/replay vertical slice the first build gate.

## Decision

Use `packages/` and `workers/` as the reference semantics for normalized market events, quality verdicts, candidate signals, deterministic risk decisions, paper order intents, fills, reconciliation, and replay. [[canonical-local-paper-runtime-v1]] now makes this reference core the default Compose runtime.

This decision does **not** promote live trading. Runtime modes are limited to `replay` and `paper_live`; `AsyncPaperExecutionService` accepts only `PaperBroker` and has no live broker port.

## Safety invariants

- A market event must match its checksum, deterministic provenance-bound ID, source mode, time, and quality contract.
- Strategies or models emit candidates, never approvals or orders.
- Candidate envelopes require local Ollama provenance: exact role/model mapping and a 64-character digest preserved into the verdict; the future producer must verify it against the inference-time local inventory.
- A candidate cannot understate authoritative market age or bid/ask spread.
- Risk loads one active policy, one fresh reconciled portfolio snapshot, and one effective instrument-constraint version from PostgreSQL; missing, stale, ambiguous, or mismatched inputs fail closed.
- Each signal receives one immutable verdict. The verdict binds account, signal, exact candidate hash/event, market event, policy, and portfolio snapshot.
- Redelivery may reuse only that exact verdict; changed candidate or event identity is rejected.
- Execution resolves the exact durable approval before creating an intent. An arbitrary `approved=True` object is insufficient.
- Intent and client order IDs are deterministic, and the intent is persisted before the paper fill is calculated.
- Decision/market TTL and price-deviation checks run immediately before intent creation.
- Ambiguous or partially filled durable intents require reconciliation and are never blindly resubmitted.
- Money arithmetic uses explicit high-precision local `Decimal` contexts.
- The database kill switch defaults active; execution enablement requires an explicit durable inactive state.

## Durable runtime extension

The initial in-memory reference adapters remain useful for deterministic replay. The active workers now add:

- `JetStreamEventBus` with manual ACK, bounded NAK/redelivery, terminal settlement, and producer message IDs;
- `PostgresRiskInputRepository` for policy, portfolio, and instrument-constraint authority;
- `PostgresDecisionStore.record_candidate_and_decision` for atomic candidate/verdict persistence;
- `PostgresExecutionLedger` for durable paper intents, orders, and fills;
- restart-safe paper execution that reuses terminal results and blocks ambiguous retry;
- canonical worker entrypoints for market data, decision, and execution;
- read-only dependency/kill-switch readiness through the control plane.

## Database migrations

- `001_core_schema.sql` establishes the migration and base schema.
- `002_execution_ledger.sql` adds risk policy/decision, intent/order/fill, reconciliation, and kill-switch authority.
- `003_runtime_durability.sql` adds event-processing scaffolding, candidate-event binding, immutable portfolio snapshots, versioned instrument constraints, one-verdict-per-signal enforcement, and position/tax-lot projection schemas.

The migration image applies checksummed migrations before runtime services start.

## Evidence

- Canonical market, risk, execution, reconciliation, and replay behavior has unit, contract, integration, migration, and failure-injection coverage.
- Golden replay produces deterministic event/order/fill identities and a digest invariant to ambient `Decimal` precision.
- Lost-ack/ambiguous-submit tests prohibit a second broker submit.
- PostgreSQL adapter tests cover exact authorization, atomic candidate/verdict binding, stale/missing risk state, durable ledger behavior, and migration structure.
- JetStream adapter/runtime tests cover explicit settlement, deduplication IDs, exact stream authority, retry bounds, and worker subject routing.

## Remaining blockers

1. Implement and separately own the missing `market.validated.v1 -> signals.candidate.v1` producer.
2. Add an audited bootstrap process for active policy, reconciled portfolio, effective constraints, and kill-switch initialization.
3. Wire migration 003's inbox/outbox/offset tables into the worker transaction and ACK lifecycle.
4. Add durable dead-letter capture and operator replay.
5. Implement authoritative cash, position, tax-lot, PnL, and loss-streak projectors from the fill ledger.
6. Complete startup reconciliation and protective stop/OCO lifecycle.
7. Add immutable raw market capture, venue sequence recovery, and promotion-grade backtesting.

## Revisit trigger

Revisit this decision only after transactional event processing, authoritative projections, restart reconciliation, and protective exits reproduce the same state under process-death and redelivery fault injection. Any live path requires a separate ADR and explicit human approval.

## Related

- [[canonical-local-paper-runtime-v1]]
- [[trade-project-full-integration-build-plan]]
- [[nats-event-system]]
- [[broker-abstraction-layer]]
- [[infrastructure-overview]]
