# Remediation Backlog

This backlog maps every numbered finding to an implementable task. Commands are
safe validation commands; no remediation was executed during the audit.

## P0 — Immediate / release stop

| ID | Component | Exact files | Proposed change | Acceptance criteria | Validation command | Dependency | Risk | Effort |
|---|---|---|---|---|---|---|---|---|
| AUD-001 | Deployment | `docker-compose.yml`, `project.manifest.toml`, `scripts/acceptance.py` | Deploy a uniquely named canonical paper-only stack and reject topology drift | 13 expected services, canonical stream/consumers, fresh leases, nonzero paper E2E | `docker compose config --services`; read-only status/NATS queries | AUD-011, AUD-012 | High migration risk; preserve old volumes read-only | L |
| AUD-002 | API security | `orchestrator/`, `services/dashboard/`, nginx/Compose | Isolate/disable legacy mutation APIs; add authn/authz and rate limits | Anonymous mutation 401/403; no admin port publicly bound | OpenAPI inspection; anonymous safe OPTIONS/GET checks | Identity/gateway design | Lockout or compatibility break | L |
| AUD-003 | Chroma | `docker-compose.yml`, knowledge/vector adapters | Remove from trading network or replace/isolate until patched | No affected reachable 1.5.9 instance; approved clients only | Image/SBOM check; network-policy inspection | Replacement/upstream fix | Knowledge search downtime | M |
| AUD-004 | Secrets | `.env` policy, `docker-compose.yml` | Rotate uncertain credentials; per-service read-only secrets; remove broker secrets from paper | Paper containers receive no live broker secret; ACL owner-restricted | Redacted inspect/name matrix; ACL check | Secret manager | Rotation outage | M |
| AUD-005 | TLS | TLS issuance/runbook/history | Revoke and reissue historical key/cert pair | New fingerprint/serial; revocation recorded | Certificate metadata check only | DNS/CA access | Brief edge maintenance | S |
| AUD-006 | vLLM | `docker-compose.yml`, inference provider/health | Correct model/GPU admission; bounded retries; dependency readiness | 24h zero restarts; model request succeeds; loss => NotReady | Read-only stats, logs, `/v1/models`, readiness | GPU/model choice | Model latency/quality change | M |
| AUD-007 | Market data | `services/market_data/`, `workers/market_producer/` | Retire legacy feed; validate canonical endpoint, freshness, order, duplicate handling | Recent closed candles traverse all stages with lineage | Public GET/status, NATS/DB read queries | Canonical deployment | Exchange rate limits | M |
| AUD-008 | PostgreSQL | `migrations/`, `scripts/apply_migrations.py`, Compose roles | Fresh canonical schema and least-privilege runtime roles | Seven checksum rows; runtime role has no elevated flags | Read-only SQL role/schema queries | Verified backup | Migration/data loss if rushed | M |
| AUD-009 | DR | `deployment/`, `scripts/`, operations docs | Encrypted off-host backups, retention, restore drill, RPO/RTO | Isolated restore passes ledger/event invariants inside SLO | Backup metadata and restore-test report inspection | Storage/KMS | Backup privacy/cost | L |
| AUD-010 | Observability | `monitoring/`, health handlers | Fix rules mount; split liveness/readiness/business readiness; wire alerts | `promtool` passes; injected failures make NotReady and alert | `promtool check config`; read-only metrics/alerts API | Canonical metrics | Alert noise | M |
| AUD-012 | CI/governance | migration test, manifest, workflow | Update immutable migration contract; own 45 paths; run required checks for all PRs | Pytest/knowledge/CI all green from clean clone | Test/knowledge commands | Ownership decisions | Incorrectly accepting changed checksum | S |

## P1 — Production blockers

| ID | Component | Exact files | Proposed change | Acceptance criteria | Validation command | Dependency | Risk | Effort |
|---|---|---|---|---|---|---|---|---|
| AUD-011 | Environment contract | `.env.example`, `README.md`, `AGENTS.md`, Compose | Document secure `DB_RUNTIME_PASSWORD` creation/injection and add redacted preflight | Clean operator can render config without value disclosure | `docker compose config` | Secret manager | Documentation drift | S |
| AUD-013 | Code quality | Manifest deploy scopes, legacy Python | Gate every deployable path; remove/quarantine legacy errors | Deployable scope Ruff/MyPy clean | Ruff/MyPy commands | AUD-001 | Large refactor regression | L |
| AUD-014 | Acceptance | `scripts/acceptance.py`, acceptance tests/artifact | Add deterministic replay and supervised public-feed paper-to-fill cases | Nonzero decisions/intents/fills and tested rejection lineage | Acceptance artifact inspection | Canonical stack | Flaky external feed | M |
| AUD-015 | Supply chain | Compose, K8s, Dockerfiles, CI | Pin image digests and action SHAs; attach SBOM/provenance | No mutable production refs | Static policy scan | Image registry | Pin maintenance | M |
| AUD-016 | Cloud deploy | `deployment/k8s/`, Terraform | Rebuild canonical topology, private admin network, security contexts/policies | Policy-as-code passes; no public admin/metrics CIDRs | Manifest render/static scan | AUD-001 | Network lockout | L |
| AUD-017 | Logging | Compose and monitoring | Bounded rotation, centralized redacted logs, disk alert | Soak stays in disk budget; alert before pressure | Docker inspect/metrics | Log store | Lost forensic detail if undersized | S |
| AUD-018 | Model artifacts | Model loaders, `models/` | Safe formats, digest/signature, compatibility manifest | Tampered/unapproved artifacts rejected | Loader negative tests | Model retraining/export | Model conversion differences | M |
| AUD-019 | Dependencies | `uv.lock`, worker Dockerfile, policy | Upgrade setuptools; resolve torch VEX; rebuild base as fixes appear | Scans clean or reviewed time-limited VEX | `pip-audit`, image scan | Upstream fixes | Dependency compatibility | S |
| AUD-020 | Telemetry | Event/metrics/dashboards/alerts | End-to-end trace and trading safety SLOs | One correlated trace; fault alerts within budget | Metrics/trace/alert read queries | AUD-010 | Cardinality/cost | L |
| AUD-021 | AI governance | Candidate/provider/agent/model registry | Immutable model registry and explicit fallback mode/reason | Unapproved digest blocked; fallback observable | Replay + registry query | Safe model format | Slower experimentation | L |
| AUD-022 | Test pyramid | CI, Go/Rust/frontend tests | Coverage, Go/Rust behavioral tests, hermetic E2E/Compose gates | All runtime gates pass; critical branches meet thresholds | Commands in report 17 | CI capacity/toolchains | Longer CI | L |

## P2 — Hardening

| ID | Component | Exact files | Proposed change | Acceptance criteria | Validation command | Dependency | Risk | Effort |
|---|---|---|---|---|---|---|---|---|
| AUD-023 | Frontend | generated metadata/lockfile, E2E config | Stabilize generated-file policy and clean-clone build | Build/E2E pass without tracked mutation | `git status`; frontend build/E2E in disposable clone | AUD-022 | Lockfile churn | S |
| AUD-024 | Ownership | `project.manifest.toml`, 122 candidate paths | Assign lifecycle/owner; delete proven dead code | No unowned deployable paths; one canonical graph | Knowledge validation/import inventory | AUD-012, AUD-013 | Accidental removal | L |
| AUD-025 | Repo artifacts | `memory/`, `models/`, `artifacts/` | Move binaries/state to signed artifact storage | Repo policy/size budget passes | `git ls-files`; artifact digest verification | Artifact registry | History remains large | M |
| AUD-026 | Edge/UI | nginx and Grafana config | CSP, banner minimization, supported pinned plugins | Header baseline passes; clean Grafana start | Header/log inspection | UI CSP tuning | Frontend breakage | S |

## P3 — Improvements

No P3 item should be started before the P0/P1 safety path is green. Future
optimization candidates include automatic manifest generation, reproducible local
dev profiles, and cost-aware inference scheduling.

## Critical path

```text
AUD-002/AUD-003/AUD-004/AUD-005 isolation
  -> AUD-011/AUD-012 green configuration and gates
  -> AUD-008 canonical database
  -> AUD-001 canonical deployment
  -> AUD-007/AUD-014 real public-feed paper E2E
  -> AUD-010/AUD-020 truthful operations
  -> AUD-009 restore + AUD-006 model stability
  -> soak gates
```

