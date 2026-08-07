---
title: Production Remediation — 2026-07-24
type: overview
tags:
  - production-readiness
  - remediation
  - sre
  - security
  - paper-trading
created: 2026-07-24
updated: 2026-07-30
sources:
  - "[[production-readiness-audit-2026-07-16]]"
  - "[[canonical-local-paper-runtime-v1]]"
  - "[[deterministic-paper-core-v1]]"
  - "[[infrastructure-overview]]"
  - "[[nats-event-system]]"
  - "[[alpha-certification-pipeline]]"
status: active
---

# Production Remediation — 2026-07-24

## Current state

The insecure 23-container legacy deployment is stopped and preserved for
rollback. A separate 12-service canonical runtime now operates on new PostgreSQL,
Redis, and NATS volumes in `paper_live` mode. It has no live broker credential
scope or live execution implementation.

This materially supersedes the runtime baseline in
[[production-readiness-audit-2026-07-16]], but it does not approve live trading
or claim long-duration autonomy.

## Verified architecture

- The default graph contains infrastructure, a read-only control plane, a real
  Binance public closed-candle producer, validation, incremental features,
  candidate generation, deterministic risk, paper execution, and reconciliation.
- PostgreSQL migrations 001-007 are checksum-bound. The runtime role has no
  superuser, database creation, role creation, replication, bypass-RLS, or
  migration-history write authority.
- Canonical processes use loopback host bindings, non-root users, read-only root
  filesystems, dropped Linux capabilities, no-new-privileges, PID bounds, and
  bounded logs.
- Chroma 1.5.9 is excluded from the default graph because
  GHSA-f4j7-r4q5-qw2c has no verified stable patched release. It remains an
  explicit isolated knowledge profile only.
- vLLM is an explicit disabled-by-default GPU profile with no restart loop.

## Real data and paper execution proof

A supervised acceptance window used real Binance public BTC/ETH 1m closed candles
and an explicitly selected deterministic baseline candidate provider. The
provider is plumbing validation, not approved alpha. It cannot place orders and
does not bypass [[deterministic-paper-core-v1]] risk authority.

| Stage | Count |
|---|---:|
| Raw / validated public market events | 74 / 74 |
| Feature snapshots | 46 |
| Candidates / risk decisions | 6 / 6 |
| Approved / rejected decisions | 2 / 4 |
| Paper intents / simulated fills | 2 / 2 |
| Reconciliation mismatches | 0 |
| Duplicate inbox / fills | 0 / 0 |

The first four candidates were rejected for tick misalignment. This exposed and
fixed a deterministic price-rounding defect before two later candidates passed
risk and produced simulated fills. The candidate worker was then restored to the
default local Ollama provider.

## Reliability discoveries

The live flow exposed a checkpoint bug not visible in isolated unit tests:
PostgreSQL `jsonb` normalized serialized checkpoint bytes, while the stored hash
used the pre-normalized string. A second defect evaluated accepted exchange clock
skew against the earlier local receive timestamp. Together they caused feature
messages after the first candle to terminate without state progress.

Checkpoint hashes now use semantic canonical JSON, and runtime features use the
actual processing clock after the market quality gate admits bounded skew. The
control plane also requires fresh BTC and ETH feature state, so recurrence makes
the runtime NotReady and disables paper execution.

A later public book-ticker timeout exposed a restart-loop boundary in portfolio
reconciliation. Failed projection already persisted a sanitized failure and
activated the kill switch, but the periodic runner still terminated the worker.
It now reports an unhealthy lease and retries without losing process continuity;
a successful projection clears only the reconciliation-owned switch. A separate
transient PostgreSQL lease-write timeout could permanently poison producer
health in memory. Renewable lease health now recovers on the next successful
write. Final verification returned HTTP 200 readiness, seven fresh worker
leases, no stale lease errors, and reconciliation restart count zero.

## Recovery and validation

- NATS fault injection held the dependency down for 31.031 seconds; readiness and
  paper execution disabled, then recovered without duplicate inbox/fill or
  reconciliation drift.
- Legacy PostgreSQL and NATS volumes have encrypted AES-256-GCM backups and were
  restored only into new isolated volumes.
- The Python suite passes 679 tests. Deployable Ruff and MyPy scopes pass.
- Go test/vet, frontend lint/typecheck/build/audit, Prometheus validation, and
  serial Playwright microstructure tests pass.
- Parallel Playwright still has resource-sensitive microstructure timeouts; CI
  uses one worker.

## Remaining release gates

The system remains not approved for production/live execution. Required work
includes CA certificate revocation/reissue, off-host immutable backup plus KMS
custody, canonical Kubernetes/Terraform source reconstruction, complete model and
artifact registries, Go behavioral and Rust gates, edge CSP hardening, alert
delivery proof, and actual 24-hour then seven-day paper soaks.

The elapsed soak is now executable through
[[24x7-paper-certification]]. Its resumable controller binds Git, the project
manifest lock, and canonical container image identities; samples runtime and
database invariants; injects bounded NATS, candidate-worker, and PostgreSQL
faults; and authenticates local status with HMAC-SHA256. It advances from the
24-hour phase to the seven-day phase only through hard gates and cannot promote
without acknowledged alert delivery plus fresh, off-host RPO/RTO evidence. The
controller does not convert missing elapsed time or external evidence into a
pass.

## System upgrade

Turn the current acceptance evidence into a continuous promotion gate: require a
fresh public-feed trace, at least one deterministic rejection, one reconciled
paper fill, zero duplicate/DLQ drift, and a signed runtime evidence bundle before
any candidate release can advance to a long-duration paper soak. The first
implementation is [[24x7-paper-certification]]; its next upgrade is asymmetric
KMS-backed signing and an approved off-host alert/backup receiver.

## Alpha certification boundary

[[alpha-certification-pipeline]] now implements content-addressed Binance
historical snapshots, exact gap policies, purged/embargoed multi-regime
walk-forward evaluation, canonical candidate/risk replay with next-bar fills and
explicit costs, temporary-only training, signed model cards, immutable staging,
and a separate exact-artifact seven-day paper-shadow controller. Final
promotion copies the already staged digest without retraining and requires
valid reliability, alpha, and shadow signatures with zero duplicate or
reconciliation drift.

This closes the missing local model-registry control but not its empirical
release gate. No trainable model is approved. The active reliability soak was
started before the alpha implementation, so the new candidate-worker image and
source digest require a fresh reliability certification and subsequent shadow
run.
