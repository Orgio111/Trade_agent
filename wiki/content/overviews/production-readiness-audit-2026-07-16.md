---
title: Production Readiness Audit — 2026-07-16
type: overview
tags:
  - production-readiness
  - sre
  - security
  - audit
  - local-first
created: 2026-07-16
updated: 2026-07-16
sources:
  - "[[canonical-local-paper-runtime-v1]]"
  - "[[deterministic-paper-core-v1]]"
  - "[[infrastructure-overview]]"
  - "[[local-trading-ai-architecture]]"
  - "[[local-knowledge-pipeline-v1]]"
status: active
---

# Production Readiness Audit — 2026-07-16

## Verdict

**NO. The repository and the currently running deployment cannot operate autonomously for weeks or months. Production readiness is 15/100.**

The canonical paper core has useful deterministic contracts and passes its isolated tests, but it is intentionally non-productive: there is no canonical market producer, canonical Ollama candidate producer, authoritative bootstrap, position/PnL projection, startup reconciliation, or protective-order lifecycle. The host instead runs a stale 23-container legacy profile with a broken data path, false-positive health, and no execution input path.

This is a software/SRE assessment only. It does not evaluate, modify, or approve trading strategy.

## Evidence boundary

- Mechanically reviewed 512 governed/unignored source, configuration, test, deployment, and documentation files: 6,316,225 bytes and 161,776 lines. All decoded as text; every included Python file parsed without syntax errors. Generated dependencies/build artifacts were excluded under repository policy.
- Exercised the current 23-container deployment and a separate disposable canonical deployment.
- Built all five canonical application images and applied migrations 001–003 to a fresh PostgreSQL database.
- Ran tests, static analysis, dependency audits, model probes, latency checks, fault injection, and a bounded live soak.
- Did not place a real order, enable live execution, modify `.env`, modify `wiki/raw/`, or invoke cloud inference.
- No multi-week/month soak was performed. Long-duration reliability remains unverified even after blockers are fixed.
- Rust compilation is unverified because Cargo is unavailable on the audited host.

## Scorecard

| Area | Score | Assessment |
|---|---:|---|
| Architecture | 32 | Canonical boundaries exist; deployed topology conflicts with them and critical components are absent. |
| Trading runtime completeness | 8 | No working market-to-paper-fill path; this does not assess strategy quality. |
| Execution | 22 | Isolated paper primitives exist; deployed execution has no input path or durable ledger. |
| Memory | 18 | Knowledge contracts exist; runtime journals/vector stores are empty or stale. |
| Local AI | 30 | Approved Ollama models work; canonical inference is not wired and E2E latency is unproved. |
| Agents | 10 | Legacy agents are inactive/simulated; canonical candidate coordination is absent. |
| Scheduler/autonomy | 4 | No canonical feed, candidate, reconciliation, outbox, retention, or backup scheduler. |
| Database | 25 | Fresh canonical migrations work; deployed database is legacy, empty, and credential-drifted. |
| Networking | 18 | Canonical loopback intent is good; actual legacy services are broadly exposed and feeds are broken. |
| Fault tolerance | 21 | Idempotency primitives exist; NATS failure leaves silently dead workers. |
| Observability | 10 | Health remains green through broken dependencies; no full-path readiness or lag/staleness SLO. |
| Production readiness | **15** | Multiple P0 blockers prevent trustworthy autonomous paper operation. |
| Security | 20 | Critical Chroma issue, unauthenticated surfaces, excess credentials, unsafe artifact loading. |
| Maintainability | 45 | Canonical scope is structured; full repository has 533 Ruff and 465 MyPy errors. |
| Performance | 12 | In-memory replay is fast; no live E2E loop exists and only a tiny warm Phi-3 probe met 300 ms. |
| Scalability | 27 | Durable foundations exist; single replicas, missing projectors/backpressure/DLQ block scale. |

## Actual deployment versus canonical source

| Property | Canonical default | Actually running |
|---|---|---|
| Service graph | Nine local paper/replay services | 23 legacy containers |
| Canonical apps | Migrate, control plane, three workers | All absent from the running profile |
| Execution | Fail-closed paper only | Two inconsistent in-memory paper authorities plus legacy Binance attempts |
| Event bus | `QUANTEX_CORE` typed v1 stream | Legacy `signals` stream with zero messages/consumers |
| Inference | Approved host-local Ollama registry | vLLM crash loop and inactive legacy runners |
| Network | Loopback-oriented | Many unauthenticated ports bound to all interfaces |
| Database | Canonical migrations 001–003 | Legacy tables only; canonical migration ledger absent |
| Readiness | Intended state-aware gate | Static/partial health reports success during failures |

The source topology and its intentional gaps are documented in [[canonical-local-paper-runtime-v1]] and [[infrastructure-overview]]. The running deployment is not that topology.

## P0 blockers

### Missing end-to-end path

- `market-data-worker` consumes `market.raw.v1`; the default runtime has no producer.
- `decision-worker` consumes `signals.candidate.v1`; no canonical producer invokes approved Ollama models.
- `execution-worker` consequently receives no authorized intent.
- The legacy market service hard-codes a testnet feed, receives repeated WebSocket HTTP 404 responses, and does not publish to the canonical bus.
- The legacy execution API exposes reads only and has neither a signal consumer nor order-submission endpoint.

### False operational readiness

A fresh disposable canonical deployment returned HTTP 200 and `ready=true` while also reporting `execution_enabled=false`, zero active policies, uninitialized portfolio/constraints, unknown kill-switch state, and zero input messages. Process readiness is being presented as operational readiness.

### Silently dead NATS workers

In a disposable real-container fault test, all three workers initially connected. After NATS was stopped for 28 seconds, the worker processes stayed alive with restart count zero. After NATS restarted, connection count remained zero. `workers/nats_runtime.py` exhausts reconnect behavior while leaving the process alive, so Docker cannot recover it.

### No authoritative bootstrap or portfolio ledger

There is no production bootstrap for active risk policy, reconciled cash/positions, venue constraints, kill-switch state, or loss/drawdown snapshots. The legacy deployment has two different in-memory balances and restart resets the execution account.

### Non-transactional delivery

Inbox/outbox/offset tables exist only as scaffolding. A crash after database commit and before publish can lose output permanently. Poison events have no DLQ or replay path. Manual ACK alone does not provide end-to-end exactly-once processing. See [[nats-event-system]].

### Incomplete execution lifecycle

The runtime lacks durable cancellation, balances, positions, partial-fill reconciliation, realized/unrealized PnL, executable stops, take-profit/OCO, trailing stops, startup reconciliation, and emergency shutdown actuation. [[deterministic-paper-core-v1]] correctly keeps live execution blocked.

### Critical dependency boundary

ChromaDB 1.5.9 is affected by CVE-2026-45829/GHSA-f4j7-r4q5-qw2c, a pre-authentication code-injection advisory with no patched release at audit time. It must not be exposed or treated as a trusted service until isolated or remediated.

## Startup and runtime evidence

- The host runs 23 legacy containers, while the canonical nine-service graph renders correctly.
- vLLM restarted from 26 to 28 during a three-minute soak and later reached at least 30 restarts.
- vLLM requests roughly 85% of 6 GB VRAM while only about 4.95 GB was free. It uses an unpinned `latest` image and a non-approved external model path.
- NATS had one connection, nearly no traffic, and a legacy stream with zero messages/consumers.
- Orchestrator health claimed `status=ok`, although all 12 runner tasks were inactive and `nats_connected=false`.
- Inference health remained green when vLLM was unavailable. Execution health remained green during Binance user-stream failures.
- Four legacy services create duplicate Binance testnet feeds and reconnect storms.
- Roughly 45 minutes of logs contained thousands of repeated connection/HTTP errors across MoE, market, order-book, feature, and vLLM services.
- Prometheus references an unmounted `recording_rules.yml`; its config fails validation for the next restart.
- Dashboard health is configured as `/health`, but the implemented path is `/api/health`.
- Most legacy containers run as root with writable filesystems and unbounded Docker logs.

The isolated canonical build was successful: five images built, migrations 001–003 applied, control plane and three workers started, and `QUANTEX_CORE` plus three consumers were created. The stream stayed empty because both upstream producers are missing; NATS recovery then failed as described above.

## Database and persistence

- `.env` has duplicate definitions for three operational keys; effective configuration is order-dependent.
- The configured database URL is a placeholder.
- The password in the running PostgreSQL container differs from current local configuration; no secret value or hash was recorded.
- A host client using current configuration failed authentication.
- The deployed database contains legacy tables only; canonical `schema_migrations` and event/risk tables are absent.
- Legacy trade/memory tables are empty except one strategy-performance row.
- Redis has zero keys/subscribers and no authoritative canonical role.
- No backup schedule, retention policy, restore drill, RPO, or RTO is implemented.

## Market-data matrix

| Feed | Status | Finding |
|---|---|---|
| Binance REST | Partial | Adapter/reachability exists; not the canonical producer. |
| Binance WebSocket | Fail | Legacy services loop on HTTP 404 and duplicate feeds. |
| Bybit candles | Partial | Candle path exists. |
| Bybit book/trades | Fail | Placeholders/empty results. |
| Funding/open interest | Fail | Legacy calls fail or are disconnected. |
| Liquidations | Fake | Random fallback is reachable. |
| Fear & Greed | Missing | Neutral default is substituted. |
| Yahoo/yfinance | Research only | Not an authoritative live source. |
| TradingView/Polygon/AlphaVantage/Finnhub/OpenBB/FRED/calendar | Missing | No production adapters found. |
| RSS/Reddit/news | Placeholder | No complete ingestion pipeline. |

Production inputs must fail closed. Synthetic inputs belong only to explicit test/research profiles.

## Execution and hard-risk matrix

| Capability | Isolated code | Autonomous runtime |
|---|---|---|
| Paper market order / full fill | Pass | Fail |
| Partial fill | Partial | Fail |
| Cancel / persistent balance | Missing | Fail |
| Positions / realized-unrealized PnL | Schema or missing | Fail |
| Stop-loss / take-profit | Validation only | Fail |
| OCO / trailing stop | Missing | Fail |
| Position sizing | Deterministic function | Authoritative inputs absent |
| Concurrent-position gate | Pass | Snapshot absent |
| Daily/weekly loss gate | Pass | Snapshot writer absent |
| Drawdown gate | Missing | Fail |
| Kill switch | Check exists | Bootstrap/actuator absent |
| Circuit breaker / emergency stop | Missing | Fail |
| Execution to journal/memory | Missing | Fail |

No risk, strategy, or execution behavior was changed by this audit.

## Local AI and agents

All six approved Ollama models are installed and callable. Neutral local probes produced:

| Model | Cold | Warm | Finding |
|---|---:|---:|---|
| Qwen3 8B | 16.97 s | 1.10 s | Above 300 ms. |
| Phi-3 3.8B | 6.26 s | 227.8 ms | Tiny prompt only; not a full loop. |
| DeepSeek-R1 8B | 31.12 s | Not established | Research role only. |
| Moondream | 7.35 s | Not established | No canonical VLM producer. |
| Mistral | 7.55 s | Not established | No canonical formatting producer. |
| nomic-embed-text | 3.48 s | 134.8 ms | 768 dimensions; not trading-loop latency. |

There is no market → feature → RAG → inference → risk → execution E2E measurement. The legacy LangGraph path uses simulated portfolio/execution state and hard-coded fallback behavior and remains quarantined by [[multi-agent-pipeline]].

## Memory and knowledge

- [[local-knowledge-pipeline-v1]] defines the permanent project-knowledge contract.
- Before this note, offline validation passed at 507 governed files, 392 embedding sources, and 1,142 chunks.
- Live project-knowledge Chroma was stale at 362 sources/1,066 chunks; the final read-only verification found 227 missing and 143 stale IDs.
- Runtime Chroma and Qdrant trade memory held zero records.
- No execution-journal growth was observed; legacy memory/trade tables were empty.
- Obsidian event dedupe performs an O(N) vault scan and will degrade with growth.

Vector stores are projections, never trading or project-memory authority.

## Production-reachable mock/random behavior

| Location | Behavior requiring quarantine/fail-closed replacement |
|---|---|
| `orchestrator/main.py` | Synthetic/random book, liquidation, HMM, TimesFM, and mock fallbacks. |
| `orchestrator/multi_tf.py`, `backtest.py` | Synthetic timeframe/OHLCV fallbacks. |
| `orchestrator/train_ppo_portfolio.py`, `retrain_finrl.py` | Synthetic training/regime data. |
| `orchestrator/seg/*` | Synthetic and random strategy/confidence fallback. |
| `services/rl_learning/__init__.py` | Mock portfolio/random values. |
| `orchestrator/broker/paper_broker.py`, `fix_simulator.py` | Random fill/ticker/simulator state. |
| `orchestrator/autogen_*` | Demo data/runners. |
| `orchestrator/local_inference_server.py` | Mock response when vLLM is unavailable. |
| `orchestrator/brains/finrl_brain.py` | Dummy environment/synthetic data. |
| `orchestrator/moss_engine/*` | Random PnL/adjustments and mock reconciliation. |
| `orchestrator/rag_agent.py` and legacy vector helpers | Hash/random embeddings. |

Legacy `services/execution/__init__.py` can double-charge commission and create a zero-size flipped position. It is not an acceptable paper ledger.

## Security and supply chain

- Running NATS, Redis, Chroma, APIs, and dashboards lack adequate authenticated boundaries; many bind to all interfaces.
- Canonical workers share a database superuser credential. Market data receives a database credential it does not need.
- Legacy administrative/training/compute routes are unauthenticated. CORS is not authentication.
- Pickle/joblib loaders can execute code if writable model artifacts are replaced.
- `install.ps1` executes remote/downloaded content without checksum or signature verification.
- Legacy containers receive cloud and broker credentials despite the canonical local paper-only boundary.
- Python dependency audit found ChromaDB CVE-2026-45829 and setuptools CVE-2026-59890; setuptools should move to 83.0.0 or later.
- Frontend audit found two moderate issues through PostCSS 8.4.31; update the chain to PostCSS 8.5.10 or later. Do not apply the tool's destructive forced downgrade suggestion.
- CI lacks SBOM, vulnerability thresholds, image/secret scans, immutable action pins, Compose smoke, chaos, restore, and soak gates.

## Verification results

| Check | Result |
|---|---|
| Python tests | 524 passed, 7 warnings; many integrations use fake buses/stores/transports. |
| Canonical Ruff / MyPy | Pass; MyPy covered 95 files. |
| Full Ruff | Fail: 533 errors, including 19 undefined names. |
| Full MyPy | Fail: 465 errors in 83 of 151 checked files. |
| Frontend lint | Pass with 7 warnings. |
| Frontend typecheck/build | Pass. |
| Go test/race | Pass compilation, but no Go test files exist. |
| Rust | Unverified; Cargo unavailable. |
| Compose rendering | Default and legacy syntax pass. |
| Canonical images/migrations | Pass. |
| Offline knowledge check | Pass after lock refresh. |
| Live knowledge projection | Fail before resync; stale. |

Nine optional/legacy dependency roots are absent: OpenCV, hmmlearn, matplotlib, NautilusTrader, Optuna, psycopg2, qdrant-client, sentence-transformers, and vLLM. Some are quarantined, but the entire repository is not import-complete.

## Performance and soak

- Canonical in-memory replay, 500 runs: p50 2.007 ms, p95 3.653 ms, p99 5.007 ms, max 6.337 ms. This is not service E2E proof.
- Legacy local health endpoints were generally 3–9 ms. Frontend p50 was about 120 ms, p95 299 ms, max 1.26 s.
- Redis ping p50 was 1.45 ms. PostgreSQL latency was blocked by credential drift.
- The actual market/inference/risk/execution loop had no traffic, so no E2E distribution exists.
- A three-minute soak proved current failure; it cannot prove month-scale reliability after repair.

## Priority remediation

### P0 — before autonomous paper operation

1. Remove the legacy profile from autostart; keep live-broker capabilities disabled. Reconcile config/credentials before reusing volumes.
2. Deploy only canonical services on a clean migration-verified database with execution disabled.
3. Build one authoritative public market producer: exchange input → canonical envelope → quality/deadman gate → `market.raw.v1`; never synthetic fallback.
4. Build the canonical candidate producer with the exact approved Ollama registry, structured validation, role/model/digest provenance, and no cloud fallback.
5. Add an audited bootstrap for active risk policy, reconciled portfolio/cash, venue constraints, and kill switch.
6. Build a durable paper ledger/projectors for orders, fills, positions, partial fills, cancellation, fees, PnL, protective orders, and startup reconciliation.
7. Fix NATS lifecycle to reconnect indefinitely with bounded jitter or exit non-zero; add worker heartbeats and a real recovery test.
8. Wire transactional inbox/outbox, durable dispatcher, DLQ, replay authorization, and idempotency tests.
9. Create per-service database roles and NATS ACLs from manifest authorities; remove excess credentials.
10. Isolate Chroma immediately until an upstream patch or equivalent safe boundary exists.

### P1 — before production reassessment

1. Separate liveness, dependency readiness, and operational readiness. Require producer/worker leases, state freshness, lag, reconciliation, risk initialization, and model residency.
2. Add bounded logs, retention, disk alerts, encrypted backups, restore drills, RPO/RTO.
3. Remove legacy vLLM and add a 6 GB-aware Ollama residency scheduler plus full-loop benchmarks.
4. Remove mock/random fallbacks from production imports and enforce explicit research/simulation profiles.
5. Add SBOM, Python/npm/image vulnerability, secret, immutable-action, Compose smoke, fault, restore, and acceptance gates.
6. Eliminate undefined-name/type failures or enforce a hard legacy quarantine from default images.
7. Rewrite Kubernetes/Terraform for canonical runtime only after local acceptance.

### P2 — maintainability and scale

1. Replace Obsidian O(N) dedupe with an indexed event ID store while preserving append-only Markdown authority.
2. Add bounded pagination, retention, and retrieval limits.
3. Resolve frontend warnings and pin build/runtime artifacts by digest.
4. Establish performance regression baselines only after a real E2E path exists.

## Required patch contracts

- **Worker lifecycle:** terminal NATS close outside normal shutdown sets fatal state and exits non-zero; a worker lease expires when consumption stops.
- **Event transaction:** inbox insert, deterministic state change, and outbox insert commit together; ACK follows commit; a durable dispatcher publishes idempotently; poison events enter DLQ with trace and replay authorization.
- **Operational readiness:** false unless migrations, NATS, producer/worker leases, lag, outbox/DLQ, risk state, reconciled portfolio, constraints, kill switch, approved model residency, and explicit execution mode all pass.
- **Least privilege:** per-service database/NATS identities, loopback operator APIs, bounded logs, pinned images, non-root, read-only filesystems where possible, dropped capabilities, `no-new-privileges`, and memory/PID limits.

## Promotion gate

Reassess autonomous **paper** operation only after every P0 item has executable evidence:

- fresh canonical start from empty volumes
- public data → local Ollama candidate → hard risk → durable paper fill → portfolio/PnL → Obsidian event → vector projection
- restart/replay without duplicate fills
- NATS/PostgreSQL/Ollama/feed fault recovery
- kill-switch and emergency-stop proof
- backup and restore proof
- no production-reachable mock/random fallback
- no exploitable critical/high dependency finding
- staged 24-hour, 7-day, and 30-day soak with SLOs and zero unexplained state drift

Live broker operation is a separate explicitly authorized future gate and is outside this audit.

## References

- [[canonical-local-paper-runtime-v1]]
- [[deterministic-paper-core-v1]]
- [[infrastructure-overview]]
- [[local-trading-ai-architecture]]
- [[local-knowledge-pipeline-v1]]
- [[nats-event-system]]
- [[multi-agent-pipeline]]
- [[trade-project-full-integration-build-plan]]
