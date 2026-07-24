# Docker and Runtime Audit

## Inventory summary

Docker 28.5.2 was available through `desktop-linux`. There were 29 containers,
23 running and 6 stopped. The running `trade_agent` project was an older
legacy/full-stack deployment, not the 13-service current default Compose topology.
See [`18-container-inventory.md`](18-container-inventory.md).

Current source improves the canonical worker image with a non-root `quantex` user,
but Compose does not enforce read-only roots, capability drops, PID limits, or
per-service security options. Most observed legacy containers ran as root with a
writable root filesystem.

## [AUD-001] Observed deployment is stale and has no operational signal pipeline

- Severity: Critical
- Category: Runtime correctness / real-data integrity
- Component: Entire deployed stack
- File: `docker-compose.yml`, `project.manifest.toml`
- Line: Multiple service/profile definitions
- Runtime service: `trade_agent` Compose project
- Status: Confirmed
- Evidence: 23 legacy containers ran instead of the 13 canonical defaults; canonical workers were absent; NATS had one legacy stream with zero messages/consumers; orchestrator reported zero active brains and `nats_connected=false`; realtime returned `no_signals_yet`.
- Impact: The system cannot make a trustworthy paper decision, enforce the canonical risk path, or prove data lineage despite numerous green container states.
- Root cause: Running Compose state predates the canonical architecture and no deployment-convergence gate compares source, labels, schema, streams, and workers.
- Reproduction: `docker compose ps --all`; inspect Compose labels; query NATS `/jsz`; query orchestrator brains and realtime status.
- Recommended fix: Keep the legacy deployment isolated; build and deploy a new uniquely named canonical paper stack from the audited commit; assert image digests, migration version, NATS stream, worker leases, and control-plane readiness before traffic.
- Validation after fix: Only manifest-active services run; `QUANTEX_CORE` subjects/consumers exist; every required lease is fresh; an instrumented paper event reaches a fill without bypassing risk.
- Estimated effort: L
- Priority: P0

## [AUD-006] vLLM is crash-looping while inference health remains green

- Severity: High
- Category: Runtime correctness / AI
- Component: Local model runtime
- File: `docker-compose.yml`, `inference/providers/vllm.py`
- Line: vLLM resource and provider configuration
- Runtime service: `quantex-vllm`, `quantex-inference`
- Status: Confirmed
- Evidence: vLLM restart count was at least 11; GPU free memory was below the configured utilization requirement; `/v1/models` reset; inference logs reported vLLM unavailable while its own health endpoint returned OK.
- Impact: AI calls fail or fall back without a truthful dependency-health signal; retry loops consume near-saturated CPU/GPU resources.
- Root cause: Model/GPU admission is not checked before start and aggregate health ignores provider readiness.
- Reproduction: Inspect restart count/log tail; request the local model endpoint and inference health.
- Recommended fix: Pin a compatible model/image, lower memory target or reserve GPU capacity, add startup/readiness probes, cap retries, and make provider state part of aggregate readiness.
- Validation after fix: Zero restarts during a 24-hour soak; model list and bounded inference succeed; dependency loss makes inference NotReady.
- Estimated effort: M
- Priority: P0

## [AUD-011] Canonical Compose cannot render from the documented environment

- Severity: High
- Category: Configuration / operability
- Component: Compose and migrations
- File: `docker-compose.yml`, `.env.example`, `README.md`, `AGENTS.md`
- Line: `docker-compose.yml:74,133,169,193,214,235,260,281,307,351`
- Runtime service: Canonical default stack
- Status: Confirmed
- Evidence: Plain `docker compose config` fails because `DB_RUNTIME_PASSWORD` is required; the variable is not documented in `.env.example` or quick-start instructions.
- Impact: A clean operator cannot render or start the canonical stack; ad hoc secret creation encourages configuration drift.
- Root cause: The database role split was introduced without synchronizing the environment contract and operations docs.
- Reproduction: Run `docker compose config` without exporting the variable.
- Recommended fix: Document secret generation and length rules without providing a value; add a redacted preflight that reports missing variable names; keep fail-closed interpolation.
- Validation after fix: A clean, documented process passes `docker compose config` and migration preflight without exposing values.
- Estimated effort: S
- Priority: P1

## [AUD-015] Images and CI actions are not immutably pinned

- Severity: Medium
- Category: Supply chain
- Component: Docker, Kubernetes, GitHub Actions
- File: `docker-compose.yml`, `deployment/k8s/`, `.github/workflows/ci.yml`
- Line: Multiple
- Runtime service: All deployment surfaces
- Status: Confirmed
- Evidence: Five all-profile Compose images use `:latest`; Kubernetes uses multiple latest tags; base images are tag-pinned but not digest-pinned; GitHub Actions use mutable major tags.
- Impact: Rebuilds can change without a source commit, undermining rollback and SBOM-to-runtime correlation.
- Root cause: No organization-wide immutable dependency policy.
- Reproduction: `rg -n ":latest|uses: .*@v[0-9]" docker-compose.yml deployment .github`.
- Recommended fix: Pin images and actions by digest/commit SHA; attach SBOM and provenance to each built image.
- Validation after fix: CI rejects mutable references and runtime labels expose the exact source/image digest pair.
- Estimated effort: M
- Priority: P1

## [AUD-017] Container logs are unbounded

- Severity: Medium
- Category: Reliability / storage
- Component: Docker logging
- File: `docker-compose.yml`
- Line: Service definitions
- Runtime service: All observed containers
- Status: Confirmed
- Evidence: All inspected containers used `json-file` without `max-size` or `max-file`; market and model services generated continuous retry logs.
- Impact: Disk exhaustion can stop databases, message storage, and all containers.
- Root cause: No Compose logging policy and no disk-pressure alert.
- Reproduction: Inspect `.HostConfig.LogConfig` for running containers.
- Recommended fix: Apply bounded log rotation, external log shipping, and disk usage alerts; preserve audit fields and trace IDs.
- Validation after fix: Soak/retry tests keep disk growth within budget and alert before critical capacity.
- Estimated effort: S
- Priority: P1

## Runtime resource observations

vLLM used approximately 98% CPU at one sample; the MoE service used approximately
41% CPU and 546 MB RAM. These are point-in-time samples, not capacity benchmarks.
Docker reported over 33 GB reclaimable across images, containers, volumes, and
build cache. No cleanup was performed.

