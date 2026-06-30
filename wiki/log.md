# Log

> Append-only timeline of every operation on the wiki. Newest at top. Reverse chronological so the current state is the first thing you read.

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
