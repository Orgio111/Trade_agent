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
updated: 2026-07-15
sources:
  - "[[trade-project-full-integration-build-plan]]"
  - "[[nats-event-system]]"
  - "[[broker-abstraction-layer]]"
status: stable
---

# Deterministic Paper Core v1

## Context

The repository previously had several disconnected event, risk, execution, and backtest paths. None supplied one reproducible market-event-to-fill trace with a fail-closed authorization boundary. [[trade-project-full-integration-build-plan]] therefore made a deterministic paper/replay vertical slice the first build gate.

## Decision

Adopt the new `packages/` and `workers/` core as the reference semantics for normalized candles, data-quality verdicts, candidate signals, risk decisions, paper order intents, fills, reconciliation, and replay.

This decision does **not** promote the system to live trading. `ExecutionService` accepts only `replay` and `paper_live` source modes. The only implemented broker for this core is deterministic `PaperBroker`.

## Safety invariants

- A parsed market event must match both its payload checksum and its deterministic provenance-bound event ID.
- Strategies emit candidates, never orders. Data age, spread, and expected slippage are supplied at the decision boundary rather than fabricated by the strategy.
- Risk decisions are immutable and bound to account, venue, market type, signal, exact candidate hash, policy version, and portfolio snapshot.
- Execution resolves the exact decision from an append-only decision authority before creating an intent; an arbitrary `approved=True` object is insufficient.
- Intent and client order IDs are deterministic. The intent is reserved before broker submission.
- A timeout after possible broker acceptance becomes `AMBIGUOUS`; it is reconciled by deterministic client ID and is never automatically resubmitted.
- Decision and market-snapshot TTLs are checked immediately before intent creation.
- Money arithmetic uses explicit high-precision local `Decimal` contexts. Replay output is invariant to ambient precision.
- The database kill switch defaults active, and SQL order intents accept only `replay` and `paper_live`.

## Implemented artifacts

- `packages/domain`, `packages/event_contracts`, and `packages/data_quality`
- `packages/risk`, `packages/execution`, `packages/brokers`, and `packages/replay`
- `workers/market_data`, `workers/decision`, and `workers/execution`
- `scripts/replay_market.py` and `scripts/apply_migrations.py`
- `db/migrations/002_execution_ledger.sql`
- `infra/compose/core.yml` and the dedicated migration image
- Contract, unit, integration, replay, failure-injection, and migration tests

## Evidence

- Full Python suite: 294 passed on 2026-07-15.
- Golden replay: six accepted candles, one candidate, one approved decision, one order, one fill, clean reconciliation.
- The replay digest is identical under ambient `Decimal` precision 6, 10, 28, and 50.
- ACK-loss tests prove an accepted broker order enters `AMBIGUOUS`, then reconciles without a second submit.
- A temporary pgvector/PostgreSQL 16 instance applied migrations 001 and 002, reapplied with no changes, retained prefixed string ID columns, and defaulted the kill switch active.

## Deferred hard blockers

1. Replace in-memory bus, decision store, and execution ledger with PostgreSQL/NATS-backed implementations and startup reconciliation.
2. Build authoritative cash, tax-lot, position, realized/unrealized PnL, and loss-streak projections.
3. Persist and reconcile protective stop/OCO or synthetic-stop lifecycles; current stop-based risk is estimated, not enforceable.
4. Add immutable raw market capture, full venue order-book recovery, and promotion-grade backtesting.
5. Expose durable kill-switch and promotion controls through the control plane and dashboard.
6. Complete fault injection for process death, delayed venue visibility, corrupted fill replay, and database failover.

## Revisit trigger

Revisit this decision only after the durable ledger can recover from process restart and broker acknowledgement loss while producing the same cash, position, order, and fill state byte-for-byte.

## Related

- [[trade-project-full-integration-build-plan]]
- [[nats-event-system]]
- [[broker-abstraction-layer]]
- [[infrastructure-overview]]
