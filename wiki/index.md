# Index

> Content catalog for the trading & quant second brain. Organized by category. ZCode updates this on every ingest. **Read this first** when answering a query — then drill into pages.
>
> **State:** 2 sources · 59 pages · last updated 2026-07-01

---

## Overviews
_Synthesis pages — the big picture of a topic._

- [[brain-ecosystem]] — single source of truth for all 12 brain weights, tiers, registration files, and aggregation flow. (stable, updated 2026-06-30)
- [[infrastructure-overview]] — full deployment architecture: Docker Compose (13 services), Kubernetes (25 resources), Terraform (Hetzner), Nginx reverse proxy, monitoring stack. (stable, updated 2026-06-30)
- [[infrastructure-overview]] — full deployment architecture: Docker Compose (13 services), Kubernetes (25 resources), Terraform (Hetzner), Nginx reverse proxy, monitoring stack. (stable, updated 2026-06-30)
- [[multi-agent-pipeline]] — dual-path LangGraph state machine: scalping fast-path (<5ms) + heavy path (VLM→RAG→swarm), configurable threshold, conditional routing. (stable, updated 2026-07-01)

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
- [[nats-event-system]] — NATS JetStream event backbone: 16 event types (incl. ACP v2), subject-based routing, stream configuration, base event fields (priority, trace_id, context). (stable, updated 2026-07-01)
- [[broker-abstraction-layer]] — Unified broker interface: Binance/FIX/Paper behind BaseBroker ABC. (stable, updated 2026-06-30)
- [[real-time-trading-dashboard]] — Next.js + WebSocket + NATS dashboard architecture for 12-brain trading cockpit. (1 source, updated 2026-06-30)
- [[attention-based-regime-detection]] — using Transformer attention weights (entropy, pattern shape) for automatic market regime classification. (research, updated 2026-06-26)
- [[build-your-own-x]] — meta-learning philosophy: build from scratch to understand. 200+ tutorials across 30+ CS categories. (1 source, updated 2026-06-26)
- [[dynamic-brain-weight-adjustment]] — attention-entropy-based regime scaling for ensemble brain weights. (stable, updated 2026-06-26)
- [[custom-trading-brain-architecture]] — proposed LSTM/Transformer brain architecture for temporal pattern recognition. (1 source, updated 2026-06-26)
- [[local-trading-ai-architecture]] — fully autonomous trading AI optimized for RTX 4050 + Ollama with incremental candle state. (1 source, updated 2026-06-30)
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
- [[vlm-agent]] — VLM chart analysis agent: matplotlib rendering → Ollama moondream → structured JSON output. (active, updated 2026-07-01)
- [[rag-agent]] — RAG pattern retrieval agent: NIM embeddings → Qdrant vector search → trade pattern statistics. (active, updated 2026-07-01)
- [[nats-langgraph-bridge]] — NATS↔LangGraph bridge: subscribes candle events, triggers pipeline, publishes results including ScalpDecisionEvent. (active, updated 2026-07-01)
- [[scalping-engine]] — ultra-fast CPU-only scalping engine: <5ms logic, 5-candle window, momentum/rejection detection, strict JSON output. (active, updated 2026-07-01)
- [[chart-segmentation]] — OpenCV chart feature extraction: candle detection, trend lines, S/R levels, volume analysis. (active, updated 2026-07-01)
- [[local-inference-server]] — vLLM + FastAPI GPU inference server for RTX 4050: quantized models, SSE streaming, /generate/fast endpoint. (active, updated 2026-07-01)
- [[gpu-optimizer]] — RTX 4050 optimization: GPU status monitoring, VRAM-based config, CUDA env tuning, model recommendations. (active, updated 2026-07-01)
- [[ultra-low-latency-monitoring]] — Grafana dashboard + Prometheus metrics for GPU utilization, VRAM, vLLM inference latency, scalping engine timing. (active, updated 2026-07-01)

## Comparisons
_Analyses and side-by-side comparisons (often filed from queries)._

_(none yet)_

## Playbooks
_How-tos, processes, checklists._

- [[local-ai-deployment-guide]] — step-by-step deployment of local trading AI on RTX 4050 + Ollama. (stable, updated 2026-06-30)
- [[signal-aggregation-logic]] — Go orchestrator weighted aggregation formula, weight mismatch fixes, proposed rebalancing. (updated 2026-06-26)
- [[nn-brain-development-guide]] — step-by-step playbook for building, training, and deploying custom NN brain. (stable, updated 2026-06-26)

## Decisions
_Decision logs: what was decided, when, why._

_(none yet)_

## Sources
_Per-source summary pages — one per ingested source._

- [[cont-order-flow-imbalance-2014]] — Cont, Kukanov & Stoikov (2014), OFI foundational source. (updated 2026-06-25)
- [[codecrafters-build-your-own-x]] — CodeCrafters' Build your own X: 200+ CS tutorials for building systems from scratch. (updated 2026-06-26)
- [[codecrafters]] — CodeCrafters: developer education platform, source of Build your own X and trading bot tutorials. (updated 2026-06-26)

---

## Orphaned / missing (from lint — flagged for creation)

_(none currently)_
