# Log

> Append-only timeline of every operation on the wiki. Newest at top. Reverse chronological so the current state is the first thing you read.

## [2026-07-30] note | Alpha certification and signed shadow promotion

Implemented [[alpha-certification-pipeline]] and synchronized [[deterministic-paper-core-v1]] plus [[production-remediation-2026-07-24]]. The new offline path seals content-addressed historical datasets with exact gap policy, runs purged/embargoed multi-regime OOS folds through canonical feature/candidate/risk replay with leakage-safe feature warm-up and next-bar realistic-cost fills, trains only temporary candidates, and writes signed model cards plus immutable staging. Real Binance BTCUSDT and ETHUSDT 1h data for 2022-2025 sealed as 35,062 rows each after exact allowlisting of one truncated and one absent maintenance interval. A separate exact-artifact `alpha_shadow` paper provider and resumable seven-day controller bind the candidate file inside the canonical container to candidate-event digests, image continuity, readiness, fills, drawdown, duplicates, and reconciliation; Git, source-lock, code, report, and artifact digests must agree before final promotion copies the staged file without retraining. All 679 Python tests pass. No trainable model is approved, and this source/image version requires a fresh reliability certification after the current pre-alpha soak.

## [2026-07-28] note | 24/7 paper certification pipeline

Implemented [[24x7-paper-certification]] and synchronized [[production-remediation-2026-07-24]]. The resumable local controller enforces an actual 24-hour NATS/PostgreSQL/worker chaos phase followed by an actual seven-day public-feed paper phase, with minute samples, zero duplicate/reconciliation/DLQ gates, bounded restart and recovery gates, immutable source/image binding, acknowledged alert delivery, fresh off-host RPO/RTO evidence, and HMAC-SHA256 status authentication. A Windows scheduled-task supervisor provides retry/resume across controller or host interruptions, WakeToRun/battery continuity, and an automatic-sleep inhibitor. All 659 Python tests plus scoped Ruff, formatting, and MyPy gates pass. `artifacts/24x7-certification-latest.json` remains explicitly `not_certified` until every elapsed and external gate passes; no live/testnet broker authority was added.

## [2026-07-24] fix | Canonical runtime health recovery

Extended [[production-remediation-2026-07-24]] after supervised runtime observation exposed two transient-failure defects. Reconciliation now persists a sanitized failure, activates the kill switch, reports unhealthy state, and retries a failed public quote cycle without container restart. Shared worker leases now recover after a one-shot PostgreSQL heartbeat write timeout instead of leaving producer health permanently degraded. Regression coverage raised the full Python result to 650 passing tests; the rebuilt canonical runtime returned HTTP 200 readiness with seven fresh leases, no stale worker errors, and reconciliation restart count zero. No live order, external credential action, volume mutation, or cloud apply occurred.

## [2026-07-24] remediation | Canonical paper runtime and real-feed proof

Recorded [[production-remediation-2026-07-24]] after quarantining the insecure 23-container legacy deployment and deploying the separate 12-service canonical paper runtime on new volumes. Verified migrations 001-007, least-privilege runtime roles, scoped secret files, bounded containers, truthful feature freshness, NATS fault recovery, encrypted isolated restore, and a real Binance public closed-candle lineage that produced six candidates, six deterministic risk decisions, two paper intents, two simulated fills, and zero reconciliation mismatches. Fixed JSONB checkpoint identity, accepted clock-skew evaluation, and deterministic tick rounding discovered by the live flow. The default candidate provider was restored to Ollama; no live order, broker credential, cloud apply, certificate rotation, or long-duration soak was claimed.

## [2026-07-16] audit | Production readiness and autonomy verification

Recorded [[production-readiness-audit-2026-07-16]] after an adversarial repository and runtime inspection. The current verdict is NO with production readiness 15/100: the machine runs a broken 23-container legacy profile, while the clean canonical runtime starts fail-closed but lacks both required producers, authoritative bootstrap, durable portfolio/PnL projections, transactional inbox/outbox/DLQ, protective-order lifecycle, and reliable NATS reconnection. Verification included 524 passing Python tests, canonical image builds and migrations, full-repository static analysis, dependency audits, local Ollama residency probes, live service inspection, a bounded soak, and isolated NATS fault injection. No trading logic, real order, cloud inference, `.env` file, or immutable raw source changed.

## [2026-07-16] verification | Canonical runtime and knowledge baseline

Validated [[canonical-local-paper-runtime-v1]] Compose rendering with only the documented `POSTGRES_PASSWORD`; the profile-free graph contains exactly nine canonical services, while the quarantined Grafana path independently fails closed without its own secret. All 524 Python tests passed across isolated canonical and quarantined processes, and migrations 001-003 completed a fresh PostgreSQL apply followed by an idempotent no-op reapply in a disposable database. The offline [[local-knowledge-pipeline-operations]] benchmark planned 392 governed sources and 1,142 chunks in 23.861 seconds at 47.860 chunks/second with an 11.807 MB traced peak; the fake embed/store path processed 192 chunks at 514.361 chunks/second. ChromaDB 1.5.9 was healthy, but the local Ollama API timed out, so no new live embedding receipt was claimed. No trading logic, real order, cloud inference, `.env` file, or immutable raw source changed.

## [2026-07-16] fix | Read-only knowledge verification

Fixed a local knowledge verification boundary where `verify` reused the synchronization namespace initializer and could create a missing Chroma collection. Verification and query now require an existing namespace, validate its immutable model/schema metadata, and fail closed without creating or modifying the collection; only `sync` retains collection-initialization authority. Updated [[local-knowledge-pipeline-operations]] and added focused adapter and integration regression tests. No trading logic, live service, `.env` file, or immutable raw source changed.

## [2026-07-16] note | Canonical worker image boundary

Recorded the frozen minimal `workers` dependency group used by [[canonical-local-paper-runtime-v1]] and [[infrastructure-overview]]. The shared non-root image now installs only the canonical FastAPI, NATS, PostgreSQL, HTTP, and validation runtime surface; its final local build and import smoke test passed at 101,562,709 bytes. Legacy ML, forecasting, broker-research, and cloud dependencies remain outside the default image. No trading logic, live execution path, cloud inference, immutable raw source, or `.env` file changed.

## [2026-07-16] note | Canonical local paper runtime v1

Recorded [[canonical-local-paper-runtime-v1]] and synchronized [[infrastructure-overview]], [[deterministic-paper-core-v1]], [[nats-event-system]], [[local-trading-ai-architecture]], [[multi-agent-pipeline]], [[local-ai-deployment-guide]], and [[trade-project-full-integration-build-plan]]. The default Compose graph is now documented as a local paper/replay core with exact Ollama provenance, read-only control plane, deterministic PostgreSQL risk authority, paper-only execution, and explicit legacy-profile quarantine. Durable inbox/outbox/DLQ wiring, authoritative bootstrap inputs, position projections, startup reconciliation, and the missing candidate producer remain fail-closed gaps. No trading logic, live broker, cloud inference, or immutable raw source changed.

## [2026-07-16] ingest | Local agent references and Hermes integration v1

Analyzed [[tradingagents]], [[journalit]], [[obsidian-ai]], and [[obsidian-memory-for-ai]] at pinned upstream revisions and recorded their exact license boundaries. Implemented [[hermes-local-integration-v1]] as a clean-room, local-only control plane: bounded Ollama workflows, append-only secret-scanned Obsidian events, persist-first projection into the isolated `trade-agent-runtime-memory-v1` Chroma collection, and loopback FastAPI operations in [[hermes-memory-operations]]. [[open-source-agent-integration-analysis]] contains the requested per-repository architecture, reusable patterns, integration mapping, and code-level contracts. No trading, strategy, signal, risk, broker, or execution logic changed. Validation completed with 412 repository tests, strict scoped Ruff/format/mypy gates, and a 250-event Windows journal benchmark at 21.786 ms create p95 and 6.640 ms idempotent-replay p95.

## [2026-07-16] note | Local knowledge pipeline built and live-verified

Completed [[local-knowledge-pipeline-v1]] and [[local-knowledge-pipeline-operations]] as an executable, fully local contract across code, durable documentation, architecture ownership, and embeddings. The offline manifest/lock check passes, all 361 repository tests pass from the frozen local dependency environment, the 19 legacy vector paths remain quarantined with zero overlap against embedding sources, and no trading, strategy, risk, sizing, broker, or execution logic changed. Local Ollama `nomic-embed-text` and ChromaDB 1.5.9 completed full document/vector checksum verification, an idempotent second synchronization embedded zero unchanged chunks, and source-attributed owner-filtered retrieval passed. The offline benchmark planned 340 sources and 1,019 chunks in 14.116 seconds under allocation tracing (72.187 chunks/second, 10.30 MB peak); the deterministic synthetic planner reached 8,480.828 chunks/second and the batched fake embed/store path reached 670.284 chunks/second.

## [2026-07-16] note | Deployment credential injection

Updated [[infrastructure-overview]] and synchronized deployment examples so PostgreSQL DSNs and Grafana passwords come from local runtime environment variables or explicit Kubernetes Secret keys. Removed plaintext credential fallbacks without changing trading, strategy, risk, or execution logic.

## [2026-07-16] note | Local knowledge infrastructure hardening

Aligned [[local-knowledge-pipeline-v1]] and [[local-knowledge-pipeline-operations]] with the executable path/type admission rules and the machine-global, collection-scoped synchronization lease. Pinned the Chroma client consistently, extended drift CI to Codex branches, and replaced literal Compose credentials with explicit local environment inputs without changing trading behavior.

## [2026-07-15] note | Local project-knowledge pipeline v1

Recorded [[local-knowledge-pipeline-v1]] and [[local-knowledge-pipeline-operations]]: `project.manifest.toml` binds component ownership, primary responsibility, code, documentation, and embedding inputs; deterministic lock drift now has an offline CI contract, while a local receipt plus live ChromaDB verification remain explicit local operations. Selected ChromaDB 1.5.9 with Ollama `nomic-embed-text` only for project knowledge, added a scoped supersession note to [[trade-project-full-integration-build-plan]], and reclassified [[rag-agent]] as a quarantined legacy trading-pattern path with no trading behavior change. Updated the index to 62 pages.

## [2026-07-15] note | Deterministic Paper Core v1 built and verified

Implemented [[deterministic-paper-core-v1]] from [[trade-project-full-integration-build-plan]]: canonical provenance-bound events, Binance/Bybit candle adapters, fail-closed quality gates, account/candidate-bound deterministic risk, issued-decision authorization, paper execution with ambiguous-submit recovery, replay, PostgreSQL ledger schema, and a checksum-locked migration runner. Full Python validation is 294 passing tests; golden replay and PostgreSQL fresh apply/no-op reapply pass. Live execution remains hard-blocked because durable repositories, cash/position projections, and protective stop/OCO lifecycle are not yet implemented. Updated the plan checkpoint and index to 60 pages.

## [2026-07-14] lint | Architecture-plan wiki consistency

Verified [[trade-project-full-integration-build-plan]] has no broken wikilinks and all 59 content pages remain indexed by category. Removed two index entries whose files are absent, converted seven current-content broken links to explicit missing-component text, flagged five missing pages, and downgraded unverified pipeline/latency entities from active to draft. Historical log links remain unchanged because the log is append-only.

## [2026-07-14] query | Trade Project Full Integration and Build Plan

Created [[trade-project-full-integration-build-plan]] from a repository audit and current primary documentation. The decision replaces the default 21-service topology with a three-worker event-driven core, keeps AI outside deterministic risk/execution, consolidates PostgreSQL/Redis/pgvector memory, defines validation and promotion gates, and provides a Phase 0–8 roadmap plus a seven-day vertical slice. Added contradiction notices to [[infrastructure-overview]], [[multi-agent-pipeline]], [[nats-event-system]], and [[local-trading-ai-architecture]]; fixed a duplicate index entry.

## [2026-07-01] ingest | Ultra-Low Latency Monitoring — Grafana Dashboard + Prometheus Metrics

Created [[ultra-low-latency-monitoring]] (entity) — Dedicated monitoring stack for GPU, vLLM inference, and scalping engine:
- Grafana dashboard with 16 panels: GPU utilization/VRAM/temp/power, vLLM latency/tokens/requests, scalping latency/signals, pipeline latency, fast-path ratio
- Template variables for GPU name and vLLM model filtering
- Prometheus metrics added to 3 source files: gpu_optimizer.py (6 metrics), local_inference_server.py (4 metrics), scalping_engine.py (2 metrics)
- Recording rules: gpu_metrics (VRAM ratio, power efficiency, temp avg), inference_metrics (vLLM latency, throughput, request rate), scalping_metrics (avg latency, signal rate, vs vLLM ratio)
- Graceful fallback: all metric imports use try/except — system works without prometheus_client
- Key fixes applied: consolidated GPU_PROMETHEUS_AVAILABLE checks, moved VLLM_REQUESTS_TOTAL inside try block, changed VRAM panel from gauge to bargauge

Updated [[index.md]]: 59 pages. Updated [[log.md]].

## [2026-07-01] ingest | Chart Segmentation Pipeline — OpenCV chart feature extraction

Created [[chart-segmentation]] (entity) — Real-time chart image → structured market features using OpenCV:
- Candle detection via HSV color segmentation (green/red masks → morphological close → contour detection)
- Trend detection via polyfit slope on candle positions (image coordinate inversion)
- Support/resistance via percentile clustering of candle extremes (20th/80th percentile)
- Volume detection via bottom-region brightness analysis
- Market structure (higher highs/lower lows) via half-comparison
- Volatility classification via candle height coefficient of variation
- Wick analysis for rejection detection (pin bar patterns)
- Graceful OpenCV degradation (returns empty features if not installed)
- Latency: 10-25ms CPU-only, deterministic
- Design principle: "trading charts are structured time-series visuals, not natural images" — replaces full VLM for speed-critical scenarios

Updated [[index.md]]: 2 sources, 58 pages.

## [2026-07-01] ingest | Ultra-Low Latency Trading Stack — Scalping Engine, Chart Segmentation, GPU Inference, Dual-Path Pipeline

Ingested 5 new entity/concept pages + updated 1 existing page into wiki:

- [[scalping-engine]] (entity) — Ultra-fast CPU-only scalping engine: <5ms logic, 5-candle window, momentum breakout + rejection detection, strict JSON output ({action, confidence, size_pct, reason}), volatility-adaptive sizing.
- [[chart-segmentation]] (entity) — OpenCV chart feature extraction: candle detection via color segmentation, trend lines (polyfit), S/R levels (percentile), volume spike detection, market structure (higher highs/lower lows).
- [[local-inference-server]] (entity) — vLLM + FastAPI GPU inference server: quantized model shortcuts (Qwen/Mistral 7B Q4), SSE streaming, `/generate/fast` endpoint for scalping, async client class, Docker-ready.
- [[gpu-optimizer]] (entity) — RTX 4050 optimization: GPU status monitoring via nvidia-smi, VRAM-based config scaling, CUDA env tuning, model recommendations by VRAM tier, WSL2 detection.
- [[multi-agent-pipeline]] (concept, updated) — Added dual-path architecture: scalping fast-path (<5ms, confidence > threshold) skips VLM/RAG/swarm; heavy path (~500ms) runs full analysis. Configurable `SCALPING_CONFIDENCE_THRESHOLD` env var + per-pipeline override. New TradingState fields: scalping_confidence, scalping_action, scalping_decision, use_scalping_fast_path, scalping_confidence_threshold. Pipeline context now includes threshold metadata.
- [[nats-langgraph-bridge]] (entity, updated) — Added ScalpDecisionEvent publishing when scalping fast-path triggers.

Updated [[index.md]]: 2 sources, 57 pages. Cross-references added to [[multi-agent-pipeline]], [[nats-event-system]], [[scalping-engine]], [[chart-segmentation]], [[local-inference-server]], [[gpu-optimizer]].

## [2026-07-01] ingest | Multi-Agent Orchestration — VLM, RAG, NATS Bridge, Pipeline Architecture

Ingested 4 new entity/concept pages + updated 1 existing page into wiki:

- [[vlm-agent]] (entity) — Vision Language Model chart analysis: matplotlib candlestick rendering → Ollama moondream → structured JSON (trend/pattern/levels/confidence). Parallel execution in LangGraph pipeline.
- [[rag-agent]] (entity) — RAG pattern retrieval: NVIDIA NIM embeddings → Qdrant vector search → trade pattern statistics (win rate, avg PnL, confidence adjustment). Hash-based local fallback.
- [[nats-langgraph-bridge]] (entity) — NATS↔LangGraph bridge: subscribes to `market.candle.>` events, triggers MultiAgentPipeline, publishes TradeSignalEvent/RiskApprovedEvent/RiskRejectedEvent/WSEvent back to NATS.
- [[multi-agent-pipeline]] (concept) — Production LangGraph state machine: `data_ingest → feature_engine → [vlm ‖ rag] → swarm_strategy → risk_gate → execution → log_and_publish`. Parallel fan-out for VLM+RAG. Trace propagation via UUID. 10-gate risk check.
- Updated [[nats-event-system]] (concept) — Added 7 new ACP v2 event types (ChartSnapshot, VLMAnalysis, RAGQuery, RAGResult, RiskCheck, RiskApproved, RiskRejected) + base event fields (priority, trace_id, context) + `agent` NATS stream.

Updated [[index.md]]: 2 sources, 56 pages. Cross-references added to [[brain-ecosystem]], [[nats-event-system]], [[multi-agent-pipeline]], [[candle-buffer]], [[continual-learning-pipeline]], [[ensemble-meta-model]], [[local-trading-ai-architecture]].

## [2026-07-01] ingest | Multi-Agent Orchestration — VLM, RAG, NATS Bridge, Pipeline Architecture

Ingested 4 new entity/concept pages + updated 1 existing page into wiki:

- [[vlm-agent]] (entity) — Vision Language Model chart analysis: matplotlib candlestick rendering → Ollama moondream → structured JSON (trend/pattern/levels/confidence). Parallel execution in LangGraph pipeline.
- [[rag-agent]] (entity) — RAG pattern retrieval: NVIDIA NIM embeddings → Qdrant vector search → trade pattern statistics (win rate, avg PnL, confidence adjustment). Hash-based local fallback.
- [[nats-langgraph-bridge]] (entity) — NATS↔LangGraph bridge: subscribes to `market.candle.>` events, triggers MultiAgentPipeline, publishes TradeSignalEvent/RiskApprovedEvent/RiskRejectedEvent/WSEvent back to NATS.
- [[multi-agent-pipeline]] (concept) — Production LangGraph state machine: `data_ingest → feature_engine → [vlm ‖ rag] → swarm_strategy → risk_gate → execution → log_and_publish`. Parallel fan-out for VLM+RAG. Trace propagation via UUID. 10-gate risk check.
- Updated [[nats-event-system]] (concept) — Added 7 new ACP v2 event types (ChartSnapshot, VLMAnalysis, RAGQuery, RAGResult, RiskCheck, RiskApproved, RiskRejected) + base event fields (priority, trace_id, context) + `agent` NATS stream.

Updated [[index.md]]: 2 sources, 56 pages. Cross-references added to [[brain-ecosystem]], [[nats-event-system]], [[multi-agent-pipeline]], [[candle-buffer]], [[continual-learning-pipeline]], [[ensemble-meta-model]], [[local-trading-ai-architecture]].

## [2026-06-30] ingest | Infrastructure Overview — Docker Compose + K8s + Terraform deployment architecture

Created [[infrastructure-overview]] (overview) — single source of truth for the full QUANTEX deployment architecture across 3 environments:

- **Docker Compose:** 13 services (postgres, redis, qdrant, nats, influxdb, orchestrator, realtime, frontend, nginx, certbot, prometheus, nats-exporter, grafana), 10 named volumes, ~8.75 CPU / ~11.5G memory limits.
- **Kubernetes:** 25 resources (8 Deployments, 1 StatefulSet, 9 Services, 2 ConfigMaps, 2 PVCs, 1 RayCluster), GPU inference (vLLM), RL training (Ray), brain weights ConfigMap.
- **Terraform:** Hetzner Cloud provisioning — tier0 (cx21, 2 vCPU, 4GB, ~$4-5/mo) and tier1 (cx41, 4 vCPU, 16GB, ~$20-40/mo) with firewall rules.
- **Nginx:** Reverse proxy with SSL termination (Let's Encrypt + fallback self-signed), rate limiting (30 r/s API, 10 r/s WS), WebSocket upgrade, security headers.
- **Monitoring:** Prometheus (30-day retention) + Grafana (auto-provisioned dashboards) + NATS exporter.
- **Data flow:** Trinity Architecture (Layer A Python brains → NATS → Layer B Go aggregator → Layer C Next.js frontend).

Updated [[index.md]]: 2 sources, 51 pages. Cross-references added to [[nats-event-system]], [[vllm-inference-provider]], [[brain-ecosystem]], [[real-time-trading-dashboard]], [[ppo-portfolio-manager]], [[broker-abstraction-layer]].
>
> Entry format: `## [YYYY-MM-DD] <op> | <subject>` where `<op>` is `ingest` · `query` · `lint` · `note`.
> Quick last-5: `grep "^## \[" log.md | tail -5`

---

## [2026-06-30] ingest | Frontend Dashboard Components — 6 pages created

Ingested 5 frontend dashboard component source files + 1 planned component into wiki:

- [[agent-swarm-visor]] (entity) — Three.js 3D brain swarm visualization: glowing spheres, inter-agent connections, particle background, HTML overlay labels
- [[price-chart]] (entity) — TradingView lightweight-charts candlestick with volume histogram, EMA 9/21 overlays, NaN-safe data filtering
- [[microstructure-panel]] (entity) — 2×2 grid: orderbook imbalance, delta/CVD divergence, spoofing detection, liquidation cascade risk
- [[inference-routing-panel]] (entity) — Multi-provider inference observability: provider health cards, adaptive chain visualization, latency heatmap, cost tracking
- [[orderbook-heatmap]] (entity) — L2 depth visualization: bid/ask bars, cumulative depth, mid-price marker, whale liquidity cluster detection
- [[odoo-erp-panel]] (concept) — planned Odoo ERP business intelligence panel (not yet implemented)

Updated `index.md`: 2 sources, 50 pages. Cross-references added to [[brain-ecosystem]], [[real-time-trading-dashboard]], [[microstructure]], [[inference-router]], [[incremental-candle-state]], [[odoo-erp-trading-integration]], [[orderflow-nautilus-brain]], [[cost-aware-llm-pipeline]].

## [2026-06-30] note | PolymarketBrain entity page — completes 12-brain wiki coverage

Created [[polymarket-brain]] (entity) — Brain #11: 5-formula mathematical pipeline (Bayesian evidence tracking, longshot bias correction, EV/ROI filtering, Quarter-Kelly sizing, Nash order routing). All 12 brains now have wiki entity pages. Updated `index.md`: 44 pages.

## [2026-06-30] ingest | Fix 9 broken wikilinks — created missing concept pages

Resolved 17 broken wikilinks by creating 9 new concept pages:

- [[hmm-regime]] — HMM regime detection (4-state Gaussian HMM, Viterbi decoding)
- [[ensemble-meta-model]] — ML/RL/rule-based signal fusion with attention entropy scaling
- [[continual-learning-pipeline]] — automatic retraining pipeline (Monitor→Trigger→Train→Validate→Promote)
- [[brain-backtest-infrastructure]] — lightweight backtest engine with 12-brain pandas proxies
- [[backtesting-pipeline]] — alias page pointing to brain-backtest-infrastructure
- [[incremental-candle-state]] — 4-level O(1) per-tick candle state management
- [[model-sequential-loading]] — VRAM-aware sequential model loading for RTX 4050
- [[inference-router]] — multi-provider inference orchestration with semantic cache
- [[microstructure]] — market microstructure concepts (OFI, tick-rule, CVD)

Also fixed `[[backtesting-pipeline]]` references in signal-aggregation-logic.md. Updated `index.md`: 2 sources, 43 pages. Wikilink count: 0 broken (down from 17).

## [2026-06-30] ingest | Event System + Broker Abstraction + vLLM + PPO Portfolio — AGENTS.md sections ingested

Ingested 4 AGENTS.md sections into wiki as concept/entity pages:

- [[nats-event-system]] (concept) — NATS JetStream event backbone: 9 event types, 6 subject categories, stream configuration, deserialization factory. Data flow diagram (Layer A → B → C).
- [[broker-abstraction-layer]] (concept) — Unified broker interface: BaseBroker ABC, 3 implementations (Binance/FIX/Paper), factory pattern, 4-step extension guide.
- [[vllm-inference-provider]] (entity) — Self-hosted GPU inference: vLLM server, OpenAI-compatible API, performance table (RTX 4090/A100/H100), model shortcuts.
- [[ppo-portfolio-manager]] (entity) — PPO RL capital allocation: PortfolioAllocEnv state space, training commands, fallback chain (PPO→Markowitz→Risk Parity→Equal Weight).

Updated [[index.md]]: 2 sources, 36 pages. Cross-references added to [[brain-ecosystem]], [[real-time-trading-dashboard]], [[odoo-erp-trading-integration]].

## [2026-06-30] note | Brain Ecosystem Overview — DRY consolidation + wiki lint

Created [[brain-ecosystem]] (overview) — single source of truth for all 12 brain weights, tiers, registration files, and aggregation flow. Replaced duplicated brain ecosystem tables in [[neural-network-brain]] and [[odoo-erp-brain]] with links to overview. Added [[brain-ecosystem]] link to [[onchain-brain]]. Updated `index.md`: 2 sources, 32 pages. No broken wikilinks.

## [2026-06-30] note | Wiki lint pass — 14 broken links, 3 orphans, 8 missing pages

Full lint pass across 29 wiki pages. Findings: 14 broken wikilinks (8 missing target pages), 3 orphan pages (codecrafters, dynamic-brain-weight-adjustment, nn-brain-development-guide), 1 index drift (dynamic-brain-weight-adjustment not in index), 8 mentioned-but-not-created pages (brain-backtest-infrastructure, ensemble-meta-model, continual-learning-pipeline, hmm-regime, incremental-candle-state, model-sequential-loading, microstructure, backtesting-pipeline).

## [2026-06-30] ingest | Odoo ERP + Real-Time Dashboard — ERP-to-trading integration

Ingested Odoo ERP integration and real-time dashboard architecture into wiki. Created 3 pages:

- [[odoo-erp-trading-integration]] (concept) — full architecture: Odoo data → feature fusion → trading signals. Signal mapping table (inventory/sales/revenue/CRM → bullish/bearish). Decision engine flow. Odoo-aware risk rules.
- [[real-time-trading-dashboard]] (concept) — Next.js + WebSocket + NATS dashboard. Full architecture from market feed → inference → brains → NATS → Go aggregator → WebSocket → UI. NATS event type → panel routing table. 7 panels: Swarm Viz, Price Chart, Microstructure, Market Structure, Inference Routing, Risk, Odoo ERP.
- [[odoo-erp-brain]] (entity) — Brain #12: XML-RPC polling, 5 sub-signals (inventory 25%, sales 30%, purchase 15%, revenue 20%, CRM 10%), weight 0.05, registration status across 7 files.

Updated [[index.md]]: 2 sources, 31 pages. No contradictions with existing content. Key integration point: connects to [[signal-aggregation-logic]] and [[local-trading-ai-architecture]].

## [2026-06-30] note | Local Trading AI Architecture — RTX 4050 + Ollama optimized design

Designed fully autonomous local trading AI system for consumer hardware (RTX 4050 6GB VRAM):
- Created [[local-trading-ai-architecture]] (concept) — sequential model loading, incremental candle state, VRAM-aware routing
- Created [[rtx4050-trading-system]] (entity) — hardware specs, model capacity table, latency profiles, Ollama config
- Created [[local-ai-deployment-guide]] (playbook) — complete step-by-step deployment for Windows/WSL2
- Key innovation: phi3:3.8b always-warm for <100ms scalping; qwen3:8b on-demand for reasoning; deepseek-r1:8b async for macro
- VRAM management: only one 8B model loaded at a time; sequential unload/load via Ollama
- Candle linking: 4-level incremental state (active candle → rolling buffer → key levels → session context → strategy stats)
- Updated `index.md`: 2 sources, 28 pages, added 1 concept + 1 entity + 1 playbook
- Integration point: connects to existing [[signal-aggregation-logic]] and brain registry

## [2026-06-26] note | Attention-based regime detection — research documented

Researched using Transformer attention weights for automatic market regime classification:
- Created [[attention-based-regime-detection]] (concept page) — full analysis of attention entropy, pattern shape, clustering approaches
- Key finding: attention entropy correlates with market regime (low=trending, high=volatile)
- Proposed 3-phase implementation: entropy filter → pattern clustering → dynamic brain weights
- Integration point: `ensemble_meta.py` as 5th regime signal source
- Validation plan: backtest entropy vs volatility (1 week), paper trade (1 month)

## [2026-06-26] note | Custom NN Brain — Transformer architecture added

Added Transformer architecture alongside LSTM in `orchestrator/brains/custom_nn_brain.py`:
- Created `TransformerPredictor` class — multi-head self-attention (4 heads, d_model=64, 2 layers), sinusoidal positional encoding
- Added `forward_with_attention()` — manually runs last encoder layer with `need_weights=True` for reliable attention capture
- Architecture selection via `CUSTOM_NN_ARCHITECTURE` env var (default: `lstm`, options: `lstm` / `transformer`)
- Attention weight averaging across heads → `attention_top_positions` and `attention_entropy` in metadata
- Model checkpoint format with config dict for Transformer, raw state_dict for LSTM
- Removed dead code, unused imports

Test results:
- Transformer: Score=-0.095, Confidence=25%, Attention weights captured successfully
- LSTM backward compat: Score=-0.1447, Confidence=25%, Auto-train working
- Both architectures pass all assertions

Updated `brain_registry.json` — Transformer model entry added.
Updated `wiki/content/entities/neural-network-brain.md` — status: stable, Transformer specs added.

## [2026-06-26] note | Custom NN Brain — implemented, tested, benchmarked

Brain #10 (custom_nn) implemented and validated:
- Created `orchestrator/brains/custom_nn_brain.py` — LSTM brain with rule-based fallback
- Updated `brain_registry.json` — weight 0.05, total now 0.95
- Created `test_custom_nn_brain.py` — standalone test script
- Created `bench_custom_nn_brain.py` — performance benchmark
- Updated `wiki/content/entities/neural-network-brain.md` — status: stable
- Fixed tensor dimension bug in auto_train (sequence building)
- Fixed UnicodeEncodeError in test scripts (ASCII fallback)

Benchmark results:
- Cold call: 6,132ms (Binance data fetch)
- Warm call: 14.5ms avg (min 14.2ms, max 14.9ms)
- Cadence headroom: ~14,985ms per 15s tick
- Auto-train: working (LSTM trained on 200 bars)
- Score: -0.1675, Confidence: 25%

## [2026-06-26] note | Custom NN Brain — full research documented in wiki

Completed comprehensive research for proposed brain #10 (custom neural network brain). All findings documented in wiki:
- Created [[custom-trading-brain-architecture]] (concept) — full architecture design, observation space, training pipeline, evaluation metrics
- Created [[neural-network-brain]] (entity) — brain specifications, ecosystem fit, learning path
- Created [[nn-brain-development-guide]] (playbook) — step-by-step development guide with code examples
- Updated [[codecrafters-build-your-own-x]] (source) — added Karpathy course deep dive, LSTM vs Transformer analysis
- Updated `index.md`: 2 sources, 24 pages

Key decisions:
- Start with LSTM for prototyping (1-2 days), migrate to Transformer for production
- Weight: 0.05 (new total: 0.95)
- Observation space: 128-dim (compatible with existing TradingEnvironment)
- Training: 180d BTCUSDT 1h from Binance, PPO algorithm
- Expert recommendation: LSTM overfits on market noise; Transformer attention is better for crypto

**Status: Research complete, implementation pending.**

## [2026-06-26] ingest | CodeCrafters Build your own X — meta-learning resource

Second ingest. Source: `raw/build-your-own-x.md` — CodeCrafters' curated index of 200+ step-by-step tutorials for building core CS technologies from scratch, spanning 30+ categories.

- Created [[build-your-own-x]] (source page) with category-to-component mapping for Trade_agent relevance.
- Created [[build-your-own-x]] (concept page) documenting the Feynman learning-by-building philosophy and how it applies to our system architecture.
- Updated `index.md`: 2 sources, 19 pages.
- **Relevance**: neural network tutorials → inform [[finrl-brain]]/[[freqai-brain]] tuning; database internals → data pipeline design; distributed systems → Go orchestrator patterns; blockchain → [[onchain-brain]] depth.
- **No contradictions** with existing wiki content. This is a meta-resource, not a technical claim.

## [2026-06-26] note | AGENTS.md + ZCODE.md — Response skeleton & FINAL RULE hardening

Added mandatory response skeleton (§0.10) and FINAL RULE to both AGENTS.md and wiki/ZCODE.md. Now every agent session follows the same structure: Current state → Analysis with [[citations]] → New Ideas / Missing / Upgrades → Prediction Engine → ≥1 system upgrade → 10× question.

## [2026-06-26] note | Brain registry + wiki entity pages — 9/9 brains documented

Created `orchestrator/brains/brain_registry.json` (structured JSON with all 7 brain specs).
Created 7 new entity pages in wiki/content/entities/:
- [[timesfm-brain]] — primary forecaster, weight 0.25
- [[freqai-brain]] — XGBoost technical indicators, weight 0.15
- [[llm-regime-brain]] — dual-tier LLM regime, weight 0.15
- [[finbert-brain]] — NLP sentiment, weight 0.10
- [[finrl-brain]] — PPO + Kelly position sizing, weight 0.10
- [[statarb-brain]] — mean reversion + funding + basis, weight 0.10
- [[onchain-brain]] — exchange flow + whale + mempool, weight 0.05

Also created 3 concept pages (orphaned → resolved):
- [[tick-rule-classification]] — trade direction proxy
- [[cumulative-volume-delta]] — signed volume delta
- [[nautilustrader]] — Rust+Python trading platform entity

Total weight across 9 brains = 0.90 (0.10 gap — safety margin or reserved).
Index updated: 1 source, 16 pages. 0 orphaned flags remaining.

## [2026-06-25] ingest | Cont, Kukanov & Stoikov (2014) — Order Flow Imbalance

First real ingest. Source: `raw/papers/cont-order-flow-imbalance.md` — the OFI foundational reference (event-based OFI, tick-rule proxy, ~40–65% concurrent variance, decay-in-seconds horizon, absorption/extremes failure modes).

Tied the theory directly to the system's existing code rather than leaving it abstract — this is the point of the wiki being inside `Trade_agent`:
- Created [[order-flow-imbalance]] (concept) with the math, the horizon/failure-mode list, and our exact implementation formula.
- Created [[microstructure-brain]] and [[orderflow-nautilus-brain]] (entities) documenting brains #4 (tick OFI, w=0.10) and #4b (depth OFI via NautilusTrader, w=0.12), mapping each design choice back to the source.
- Created the source summary [[cont-order-flow-imbalance-2014]].

Cross-refs in place; every new page has inbound links. Updated index (1 source, 4 pages). **Lint flag:** `[[tick-rule-classification]]` is referenced by 3 pages but not yet created — deferred to the next pass to keep this ingest focused. Open questions logged on the source page: crypto-venue-specific OFI horizon, our tick-rule's misclassification rate vs Lee–Ready, absorption detector false-positive rate.

Scaffolded the second brain inside `Trade_agent/wiki/` per the LLM Wiki pattern. Created `ZCODE.md` (schema), `index.md` (catalog), this `log.md`, `README.md`, and the full `raw/` + `content/` directory tree. No sources ingested yet. Awaiting first ingest.
