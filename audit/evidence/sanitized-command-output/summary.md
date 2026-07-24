# Sanitized Command Evidence

Audit date: 2026-07-24 (Asia/Ulaanbaatar)  
Repository: `Trade_agent`  
Mode: read-only; paper/testnet only

No secret values, tokens, passwords, private keys, or full environment dumps are included.

## Repository

```text
tracked_files=591
untracked_files=0 (before audit reports)
ignored_files=72,386
branch=fix/production-ready-paper-runtime
head=0139367
tracking_delta=0 ahead / 0 behind (local tracking ref; no fetch performed)
dirty_user_owned_files=frontend/next-env.d.ts, frontend/package-lock.json
python_files=357
python_ast_parse_errors=0
test_files=80
```

The Git object database passed `git fsck --connectivity-only`. Dangling objects were
present but no corruption was reported. The packed repository size was approximately
236.78 MB.

## Compose and Docker

```text
docker_client=28.5.2
docker_server=28.5.2
docker_context=desktop-linux
host_cpus=12
host_memory≈8.17GiB
containers_total=29
containers_running=23
containers_stopped=6
images=121
compose_projects=trade_agent running(23)
```

`docker compose config` failed without a process-supplied `DB_RUNTIME_PASSWORD`.
With non-secret, process-only placeholders it rendered 13 default services and 33
services when all profiles were enabled. No repository or container state was
changed to perform this render.

The running 23-container project did not match the current 13-service canonical
default. It was an older legacy/full-stack topology.

```text
NATS HTTP health=200
NATS streams=signals only
NATS messages=0
NATS consumers=0
orchestrator health=200
orchestrator brains registered=12
orchestrator brains active=0
orchestrator nats_connected=false
realtime aggregation status=no_signals_yet
vLLM model endpoint=connection reset
vLLM observed restart_count>=11
```

Docker disk usage at the audit point:

```text
images=62.54GB (20.51GB reclaimable)
containers=1.588GB (1.201GB reclaimable)
volumes=2.219GB (1.25GB reclaimable)
build_cache=12.53GB (12.53GB reclaimable)
```

No cleanup command was executed.

## Data stores

```text
PostgreSQL login roles=1
PostgreSQL runtime login role=superuser
canonical schema_migrations table=absent
legacy application tables=6
estimated application rows=0
archive_mode=off
data_checksums=off
Redis ping=PONG
Redis dbsize=0
Redis AOF=enabled
Qdrant collection=trade_memory
Qdrant points=0
Qdrant status=green
Chroma project-knowledge records=1066
Chroma runtime-memory records=0
```

## Tests and static analysis

```text
pytest=1 failed, 604 passed, 7 warnings
pytest failure=tests/contract/test_migration_runner.py
pytest collected=605
canonical Ruff packages/workers=pass
full-repository Ruff=533 findings
canonical MyPy packages/workers=pass
legacy MyPy=477 errors in 87 of 157 checked files
frontend lint=pass with 7 warnings
frontend typecheck=pass
frontend npm audit=0 vulnerabilities
Go test=pass compilation; no test files
Go vet=pass
Rust toolchain=not available
Playwright listed=20 tests
Playwright execution=timed out; no result claimed
knowledge governance=failed; 45 unowned governed paths
```

The Playwright browser processes started by the timed-out audit command exited; no
browser process matching the audit profile remained.

## Dependency and image evidence

```text
Python packages audited=182
Python advisories=4 across 3 packages
Chroma=1 critical advisory, no patched release at audit date
setuptools=1 advisory, fixed version available
torch=1 advisory with inconsistent scanner/advisory version ranges
worker image SARIF=1 critical + 2 high OS-package findings, no fixes listed
```

## Observability

```text
Prometheus scrape targets up=4
promtool config validation=failed
missing referenced file=/etc/prometheus/recording_rules.yml
container log driver=json-file
log rotation options=not configured
```

## Safety statement

No source code, configuration, running container, database, cache, queue, vector
store, or external service was mutated. Only files under `audit/` were created.

