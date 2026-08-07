# Index

> Content catalog for the trading & quant second brain. Organized by category. ZCode updates this on every ingest. **Read this first** when answering a query — then drill into pages.
>
> **State:** 6 sources · 74 pages · last updated 2026-07-30

---

## Overviews
_Synthesis pages — the big picture of a topic._

- [[production-remediation-2026-07-24]] — canonical paper-runtime containment, real public-feed paper-fill evidence, restore/fault proof, and remaining production gates. (active, updated 2026-07-24)
- [[brain-ecosystem]] — single source of truth for all 12 brain weights, tiers, registration files, and aggregation flow. (stable, updated 2026-06-30)
- [[infrastructure-overview]] — canonical nine-service local paper deployment, loopback boundaries, and explicit legacy-profile quarantine. (stable, updated 2026-07-16)
- [[multi-agent-pipeline]] — quarantined LangGraph/VLM/RAG/swarm design and the contract required for any future candidate-producer integration. (draft, updated 2026-07-16)
- [[production-readiness-audit-2026-07-16]] — adversarial code, deployment, fault, security, latency, persistence, and autonomy audit; current verdict NO at 15/100. (active, updated 2026-07-16)

## Concepts
_Methods and ideas, synthesized across all sources._

- [[odoo-erp-panel]] — planned Odoo ERP business intelligence panel for the real-time dashboard. (planned, updated 2026-06-30)
- [[odoo-erp-trading-integration]] — Odoo ERP business data as trading signals: inventory, sales, revenue, CRM → directional scores. (1 source, updated 2026-06-30)
- [[backtesting-pipeline]] — alias for [[brain-backtest-infrastructure]]; validates brain weights before deployment. (stable, updated 2026-06-30)
- [[brain-backtest-infrastructure]] — lightweight backtest engine simulating 12-brain weighted aggregation with pandas proxies. (stable, updated 2026-06-30)
- [[continual-learning-pipeline]] — automatic retraining pipeline: Monitor → Trigger → Train → Validate → Promote. (stable, updated 2026-06-30)
- [[ensemble-meta-model]] — fuses ML/RL/rule-based signals with adaptive weights and attention entropy regime scaling. (stable, updated 2026-06-30)
- [[hmm-regime]] — Hidden Markov Model for probabilistic market regime classification (4 states). (stable, updated 2026-06-30)
- [[inference-router]] — multi-provider inference orchestration with semantic cache and circuit breaking. (stable, updated 2026-06-30)
- [[incremental-candle-state]] — 4-level incremental state management for O(1) per-tick candle processing. (stable, updated 2026-06-30)
- [[microstructure]] — market microstructure concepts: OFI, tick-rule, CVD. (stable, updated 2026-06-30)
- [[model-sequential-loading]] — VRAM-aware sequential model loading for RTX 4050 6GB. (stable, updated 2026-06-30)
- [[nats-event-system]] — exact `QUANTEX_CORE` v1 subjects, manual-ACK/redelivery semantics, producer dedupe, and current inbox/outbox/DLQ gaps. (stable, updated 2026-07-16)
- [[broker-abstraction-layer]] — Unified broker interface: Binance/FIX/Paper behind BaseBroker ABC. (stable, updated 2026-06-30)
- [[real-time-trading-dashboard]] — Next.js + WebSocket + NATS dashboard architecture for 12-brain trading cockpit. (1 source, updated 2026-06-30)
- [[attention-based-regime-detection]] — using Transformer attention weights (entropy, pattern shape) for automatic market regime classification. (research, updated 2026-06-26)
- [[build-your-own-x]] — meta-learning philosophy: build from scratch to understand. 200+ tutorials across 30+ CS categories. (1 source, updated 2026-06-26)
- [[dynamic-brain-weight-adjustment]] — attention-entropy-based regime scaling for ensemble brain weights. (stable, updated 2026-06-26)
- [[custom-trading-brain-architecture]] — proposed LSTM/Transformer brain architecture for temporal pattern recognition. (1 source, updated 2026-06-26)
- [[local-trading-ai-architecture]] — exact six-role local Ollama registry, candidate provenance, endpoint admission, and deterministic authority boundary. (stable, updated 2026-07-16)
- [[order-flow-imbalance]] — signed order-book pressure; the cleanest mechanical price-formation signal. Now-pressure, not a forecast. (1 source, updated 2026-06-25)
- [[tick-rule-classification]] — trade direction proxy comparing price to previous trade; standard when bid/ask unavailable. (1 source, updated 2026-06-26)
- [[cumulative-volume-delta]] — running sum of signed trade volume; net buying/selling pressure. Complement to OFI. (1 source, updated 2026-06-26)

## Strategies

_(none yet)_

## Indicators
_Signals, features, and metrics._

_(none yet)_

## Entities
_Instruments, venues, tools, people, firms._

- [[agent-swarm-visor]] — Three.js 3D brain swarm visualization: glowing spheres, inter-agent connections, particle background. (active, updated 2026-06-30)
- [[price-chart]] — TradingView lightweight-charts candlestick with volume histogram, EMA 9/21 overlays, NaN-safe data filtering. (active, updated 2026-06-30)
- [[microstructure-panel]] — 2×2 grid: orderbook imbalance, delta/CVD, spoofing detection, liquidation cascade risk. (active, updated 2026-06-30)
- [[inference-routing-panel]] — Multi-provider inference observability: health cards, adaptive chains, latency heatmap, cost tracking. (active, updated 2026-06-30)
- [[orderbook-heatmap]] — L2 depth visualization: bid/ask bars, cumulative depth, mid-price marker, whale cluster detection. (active, updated 2026-06-30)
- [[rtx4050-trading-system]] — hardware-optimized local trading system for RTX 4050 6GB VRAM + Ollama. (1 source, updated 2026-06-30)
- [[microstructure-brain]] — orchestrator brain #4; trade-tick OFI, weight 0.10. (1 source, updated 2026-06-25)
- [[orderflow-nautilus-brain]] — orchestrator brain #4b; depth-based OFI via NautilusTrader (tick fallback), weight 0.12. (1 source, updated 2026-06-25)
- [[nautilustrader]] — open-source high-performance trading platform (Rust core + Python bindings); provides depth-based OFI for brain #4b. (1 source, updated 2026-06-26)
- [[timesfm-brain]] — brain #1; primary forecaster, zero-shot TimesFM 2.5, weight 0.25. (updated 2026-06-26)
- [[freqai-brain]] — brain #2; XGBoost technical indicators (RSI/MACD/BB), weight 0.15. (updated 2026-06-26)
- [[llm-regime-brain]] — brain #3; dual-tier LLM regime classifier (Ollama→NIM→OpenRouter), weight 0.15. (updated 2026-06-26)
- [[finbert-brain]] — brain #5; NLP sentiment via 2× Ollama + cloud tiers, weight 0.10. (updated 2026-06-26)
- [[finrl-brain]] — brain #6; PPO + Kelly position sizing, weight 0.10. (updated 2026-06-26)
- [[statarb-brain]] — brain #8; z-score mean reversion + funding + basis (contrarian), weight 0.10. (updated 2026-06-26)
- [[onchain-brain]] — brain #7; exchange flow + whale + mempool, weight 0.05. (updated 2026-06-26)
- [[vllm-inference-provider]] — self-hosted GPU inference via vLLM, zero API cost, OpenAI-compatible API. (active, updated 2026-06-30)
- [[ppo-portfolio-manager]] — PPO RL-based capital allocation with Markowitz fallback chain. (active, updated 2026-06-30)
- [[polymarket-brain]] — brain #11; Polymarket 5-formula alpha pipeline (Bayesian/Longshot/EV/Kelly/Nash), weight 0.05. (active, updated 2026-06-30)
- [[odoo-erp-brain]] — brain #12; Odoo ERP business intelligence via XML-RPC, weight 0.05. (active, updated 2026-06-30)
- [[neural-network-brain]] — brain #10; custom LSTM for temporal patterns, weight 0.05. (stable, updated 2026-06-26)
- [[vlm-agent]] — experimental VLM chart analysis; missing candle-buffer documentation and excluded from the deterministic hot path. (draft, updated 2026-07-14)
- [[rag-agent]] — legacy experimental trading-pattern RAG using NIM/Qdrant; quarantined from project knowledge and unchanged by the local knowledge pipeline. (draft, updated 2026-07-15)
- [[nats-langgraph-bridge]] — experimental NATS↔LangGraph bridge; requires canonical contract, durable consumer, and replay verification. (draft, updated 2026-07-14)
- [[scalping-engine]] — experimental CPU-only signal engine; latency and risk integration remain unverified. (draft, updated 2026-07-14)
- [[chart-segmentation]] — OpenCV chart feature extraction: candle detection, trend lines, S/R levels, volume analysis. (active, updated 2026-07-01)
- [[ultra-low-latency-monitoring]] — experimental GPU/vLLM/scalping dashboard; two referenced component pages are missing. (draft, updated 2026-07-14)

## Comparisons
_Analyses and side-by-side comparisons (often filed from queries)._

- [[open-source-agent-integration-analysis]] — license-aware extraction and clean local integration plan for TradingAgents, Journalit, Obsidian AI, and Obsidian Memory for AI. (stable, updated 2026-07-16)
- [[trade-project-full-integration-build-plan]] — evidence-backed roadmap plus the canonical local paper runtime and remaining transactional/bootstrap gates. (stable, updated 2026-07-16)

## Playbooks
_How-tos, processes, checklists._

- [[alpha-certification-pipeline]] — immutable historical data, purged multi-regime OOS replay, temporary training, exact-artifact paper shadow, and signed promotion. (active, updated 2026-07-30)
- [[24x7-paper-certification]] — resumable 24-hour chaos and seven-day paper soak with signed evidence and hard promotion gates. (active, updated 2026-07-28)
- [[local-ai-deployment-guide]] — safe default Compose startup, exact Ollama checks, read-only readiness, and fail-closed bootstrap status. (stable, updated 2026-07-16)
- [[signal-aggregation-logic]] — Go orchestrator weighted aggregation formula, weight mismatch fixes, proposed rebalancing. (updated 2026-06-26)
- [[nn-brain-development-guide]] — step-by-step playbook for building, training, and deploying custom NN brain. (stable, updated 2026-06-26)
- [[local-knowledge-pipeline-operations]] — local lock/check/sync/verify/query operations, security exclusions, failure recovery, and release checklist. (stable, updated 2026-07-16)
- [[hermes-memory-operations]] — loopback Hermes event, workflow, Obsidian, Chroma, recovery, and verification operations. (stable, updated 2026-07-16)

## Decisions
_Decision logs: what was decided, when, why._

- [[deterministic-paper-core-v1]] — deterministic domain semantics extended by durable JetStream/PostgreSQL adapters and paper-only worker entrypoints. (stable, updated 2026-07-16)
- [[canonical-local-paper-runtime-v1]] — default local Compose authority graph, Ollama provenance, read-only control plane, durable boundaries, and known gaps. (stable, updated 2026-07-16)
- [[local-knowledge-pipeline-v1]] — manifest-owned, CI-enforced project knowledge using ChromaDB 1.5.9 and local Ollama `nomic-embed-text`. (stable, updated 2026-07-16)
- [[hermes-local-integration-v1]] — bounded local Ollama workflows plus append-only Obsidian memory and an isolated runtime Chroma projection. (stable, updated 2026-07-16)

## Sources
_Per-source summary pages — one per ingested source._

- [[cont-order-flow-imbalance-2014]] — Cont, Kukanov & Stoikov (2014), OFI foundational source. (updated 2026-06-25)
- [[codecrafters-build-your-own-x]] — CodeCrafters' Build your own X: 200+ CS tutorials for building systems from scratch. (updated 2026-06-26)
- [[codecrafters]] — CodeCrafters: developer education platform, source of Build your own X and trading bot tutorials. (updated 2026-06-26)
- [[tradingagents]] — Apache-2.0 LangGraph reference; typed orchestration patterns adopted without its trading authority. (updated 2026-07-16)
- [[journalit]] — proprietary Obsidian journal reference; only clean-room observable patterns are used. (updated 2026-07-16)
- [[obsidian-ai]] — MIT agent-adapter, session, streaming, and vault-context reference. (updated 2026-07-16)
- [[obsidian-memory-for-ai]] — unlicensed memory protocol reference; append/event/provenance concepts only. (updated 2026-07-16)

---

## Orphaned / missing (from lint — flagged for creation)

- **candle-buffer** — referenced by [[nats-langgraph-bridge]] and [[vlm-agent]], but no page exists.
- **feature-engine** and **risk-engine** — referenced by [[scalping-engine]], but no pages exist.
- **gpu-optimizer** and **local-inference-server** — historical log entries claim creation, but the pages are absent from the current tree.
