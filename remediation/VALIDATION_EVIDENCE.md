# Sanitized Validation Evidence

Updated: 2026-07-24 (Asia/Ulaanbaatar)

No secret values, resolved secret-bearing Compose output, credentials, or private
keys are recorded here.

## Input and containment

- `audit.zip` SHA-256:
  `fed84561a8ea2336551d5273c665146407f45ca9be982aa2818eaf25dfac7a07`.
- ZIP validation: 25 files, 0 unsafe paths, 0 missing files, 0 hash mismatches.
- All audit reports and evidence were read.
- Baseline: branch `fix/production-ready-paper-runtime`, commit `0139367ceb05`.
- Preserved user changes: `frontend/next-env.d.ts`,
  `frontend/package-lock.json`; `audit/` remains untracked input.
- Exactly 23 legacy `trade_agent` containers were stopped after identity, image,
  mount, health, port, and rollback capture. Containers, networks, and volumes
  were not removed.
- Legacy host ports were confirmed closed. Legacy persistent state remains intact.

## Canonical runtime

- Compose project: `trade_agent_canonical`.
- Default graph: 12 services:
  `postgres`, `redis`, `migrate`, `nats`, `control-plane`,
  `market-producer`, `market-data-worker`, `feature-worker`,
  `candidate-worker`, `reconciliation-worker`, `decision-worker`,
  `execution-worker`.
- Runtime mode: `paper_live`; live trading enabled: `false`.
- Current candidate provider after supervised acceptance: `ollama`.
- Loopback host bindings only: PostgreSQL, Redis, NATS client/monitor, control
  plane.
- Canonical processes: read-only root filesystem, `cap_drop=ALL`,
  `no-new-privileges`, PID limit 128, `json-file` `10m`/5 rotation.
- Container environment-name inspection found no broker/cloud/provider secret
  names. Workers receive only the runtime DB password file.
- Readiness after deployment:
  `migrations=7`, one active policy, fresh portfolio, initialized constraints,
  fresh BTC/ETH feature state, kill switch false, seven worker leases ready,
  NATS reachable, approved local Ollama inventory present.
- A public book-ticker timeout exposed a reconciliation crash-loop path. The
  failure already activated the PostgreSQL kill switch, but the periodic runner
  propagated the exception. It now retries without terminating, reports a
  sanitized unhealthy lease, and clears only after a successful projection.
- A separate one-shot PostgreSQL lease-write timeout exposed permanent
  in-memory producer degradation. The next successful heartbeat now renews the
  lease. Final runtime verification returned HTTP 200 readiness with all seven
  leases fresh, no stale worker errors, and reconciliation restart count 0.
- PostgreSQL retained five sanitized transient reconciliation failures as
  `RuntimeError:<fingerprint>` evidence; the next twelve sampled cycles were
  matched, the kill switch cleared through reconciliation authority, and the
  worker remained at restart count 0.
- Under post-build host contention, one of six consecutive readiness samples
  timed out on the bounded NATS probe and correctly disabled execution. The next
  five samples were HTTP 200 Ready; direct NATS health and PostgreSQL
  `pg_isready` checks also passed. This is not a substitute for the pending
  24-hour soak.

## PostgreSQL authority

- Exact immutable migrations applied: 001 through 007.
- Runtime role flags:
  `superuser=false`, `createdb=false`, `createrole=false`,
  `inherit=false`, `replication=false`, `bypassrls=false`.
- `schema_migrations`: runtime SELECT granted; INSERT/UPDATE/DELETE denied.
- Canonical tables include inbox/outbox, leases, feature checkpoints, policies,
  constraints, decisions, intents, broker orders, fills, portfolio projection,
  reconciliation, DLQ, kill switch, and replay audit.
- Feature incident reproduced and fixed:
  PostgreSQL JSONB formatting changed checkpoint bytes and the worker used
  `received_ts` despite an accepted 10-20 ms exchange clock lead. Semantic
  canonicalization plus processing-time evaluation advanced BTC/ETH feature state
  from version 1 to version 10+.
- Readiness now becomes false when both BTC/ETH feature states are not fresh
  within two minutes.

## Real public-feed supervised paper acceptance

Artifact: `artifacts/supervised-paper-flow-latest.json`.

| Stage | Count |
|---|---:|
| Real Binance public raw closed candles | 74 |
| Validated market events | 74 |
| Published feature snapshots | 46 |
| Explicit deterministic-baseline candidates | 6 |
| Deterministic risk decisions | 6 |
| Approved / rejected | 2 / 4 |
| Paper order intents | 2 |
| Simulated paper fills | 2 |
| Reconciliation runs / mismatches | 429 / 0 |
| Duplicate inbox / duplicate fills | 0 / 0 |
| Pending outbox / DLQ | 0 / 0 |

The first four candidates were correctly rejected as
`PRICE_NOT_TICK_ALIGNED`; tick rounding was then fixed and two later candidates
were approved and filled. This proves the risk gate rejected malformed candidates
before execution. Both successful lineages retain trace ID, risk decision, client
order ID, and paper fill ID. The supervised provider was disabled immediately
after capture and the candidate worker was recreated with `ollama`.

## Fault and restore evidence

- Isolated acceptance project:
  `tradeagent_acceptance_20260724d`.
- Baseline ready: true.
- Deliberate NATS stop: 31.031 seconds; readiness false and paper execution
  disabled.
- NATS restart: readiness and execution recovered.
- Post-recovery invariants: 0 duplicate inbox/fills, 0 pending outbox,
  0 expired leases, 0 failed reconciliation.
- Legacy NATS encrypted backup:
  AES-256-GCM, ciphertext SHA-256
  `2e0d77208be347d5a0b84649a76057717744b507c26c1d7684d49621c028458d`,
  key ID `b02fb358cbecd9ae`.
- Legacy PostgreSQL encrypted backup:
  AES-256-GCM, ciphertext SHA-256
  `44f1037007c60dd27d58bf656d2b0d25c1945c537cf46a4321734d8feb457b07`,
  key ID `c1e56f7b7b832df2`.
- Isolated restored PostgreSQL: expected `quantex` DB, six legacy tables, one
  legacy login, and no falsely invented canonical migration/kill-switch state.
- Isolated restored NATS: one stream, zero consumers/messages/bytes, matching the
  legacy source.
- Off-host immutable copy and KMS custody are not completed.

## Quality and release gates

| Gate | Sanitized result |
|---|---|
| Full Python suite | PASS: 650 tests, 7 deprecation warnings |
| Ruff deployable scope | PASS: `packages workers` |
| MyPy deployable scope | PASS: 117 source files with explicit package bases |
| Broader legacy test Ruff | FAIL: 25 pre-existing findings; not release scope |
| Go test / vet | PASS; module reports no test files |
| Frontend lint | PASS: 0 errors, 7 allowed warnings |
| Frontend typecheck | PASS |
| Frontend Next.js production build | PASS |
| npm audit | PASS: 0 vulnerabilities |
| Playwright parallel | 14 passed, 6 microstructure timeouts |
| Playwright microstructure serial retry | PASS: 7/7 |
| Compose redacting preflight | PASS: 0 findings |
| Compose render/services | PASS: 12 expected services |
| Prometheus | PASS: config, 13 alerts, 20 recording rules |
| Knowledge manifest/lock | PASS: 575 governed files, 24 components, 0 unowned paths |
| Rust test/clippy | NOT RUN: Cargo/rustc unavailable |
| Terraform validate | NOT RUN: Terraform unavailable |
| 24-hour / seven-day soak | NOT COMPLETE |

## Supply chain and advisories

- GitHub Actions use commit SHAs.
- Deployable third-party images and Dockerfile base images use digests.
- CI image tags use `${{ github.sha }}` and request SBOM plus maximum provenance
  attestations.
- Chroma 1.5.9 was confirmed affected by GHSA-f4j7-r4q5-qw2c /
  CVE-2026-45829. No stable patched release was available at verification time;
  it is disabled from the default graph instead of falsely declared fixed.

## Known non-results

- No live or testnet broker order was sent.
- No external credential was validated or rotated.
- No certificate was revoked/reissued.
- No cloud infrastructure was applied.
- No alert receiver delivery was proven.
- No 24-hour or seven-day elapsed soak was claimed.
