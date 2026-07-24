# Database and Data Integrity Audit

## Observed stores

| Store | Observed state | Production interpretation |
|---|---|---|
| PostgreSQL | Reachable; two databases; only legacy schema | Canonical schema not deployed |
| Redis | PONG, empty, AOF enabled | Available but no active pipeline state |
| Qdrant | Green, `trade_memory`, zero points | Real service, no runtime evidence |
| Chroma | 1,066 project-knowledge records; zero runtime-memory records | Knowledge use exists; trading memory empty |
| NATS JetStream | One legacy stream, zero messages/consumers | Persistence service up, application flow absent |

The canonical migrations define versioned trading state, inbox/outbox, leases,
policy, broker constraints, reconciliation, and runtime roles. The observed
PostgreSQL instance had none of these tables.

## [AUD-008] Runtime database role and schema do not match canonical migrations

- Severity: High
- Category: Database integrity / least privilege
- Component: PostgreSQL
- File: `migrations/001_*.sql` through `migrations/007_*.sql`, `scripts/apply_migrations.py`
- Line: Multiple
- Runtime service: `quantex-postgres`
- Status: Confirmed
- Evidence: `schema_migrations` was absent; only six legacy tables existed with zero estimated rows; the sole login role had superuser, createdb, createrole, and replication privileges.
- Impact: Canonical workers cannot safely run, schema/version claims are false, and a compromised service has cluster-wide database power.
- Root cause: The observed volume was initialized by an older deployment and no controlled migration/role convergence occurred.
- Reproduction: Query `pg_roles`, `information_schema.tables`, and `to_regclass('schema_migrations')` using the configured runtime identity without printing credentials.
- Recommended fix: Back up first; provision a fresh canonical database; run checksummed migrations with separate admin and runtime identities; grant only the minimum runtime privileges; refuse startup on version drift.
- Validation after fix: Seven expected migration checksums exist; runtime role is non-superuser/no-createdb/no-createrole/no-replication; workers pass transaction/restart tests.
- Estimated effort: M
- Priority: P0

## [AUD-009] Backup, restore, RPO, and RTO are unverified

- Severity: High
- Category: Disaster recovery
- Component: PostgreSQL, NATS, Redis, Chroma/Qdrant, model artifacts
- File: `docker-compose.yml`, `deployment/`, `scripts/`
- Line: N/A
- Runtime service: Stateful services
- Status: Confirmed
- Evidence: No operational backup job, immutable destination, retention policy, restore drill, RPO, or RTO implementation was found; PostgreSQL archive mode was off and checksums were off.
- Impact: Host, volume, corruption, or operator failure can cause unrecoverable state and unknown outage duration.
- Root cause: Persistence was provisioned, but lifecycle and recovery operations were not designed as product features.
- Reproduction: Search deployable configuration and scripts for scheduled backup/restore; inspect PostgreSQL archive/checksum settings.
- Recommended fix: Define data classes and RPO/RTO; implement encrypted, off-host, tested backups; include NATS stream state and configuration; run restore drills into isolated infrastructure.
- Validation after fix: A documented drill restores a known snapshot and verifies ledger/order/event consistency inside the stated RPO/RTO.
- Estimated effort: L
- Priority: P0

## Integrity notes

- The canonical inbox/outbox and reconciliation design is appropriate for
  at-least-once delivery, but it is not active in the observed database.
- All legacy trading tables were empty, so no production performance or PnL claim
  can be verified.
- Redis and Qdrant were operational but effectively unused.
- Chroma project knowledge was populated, while runtime agent memory was empty.

