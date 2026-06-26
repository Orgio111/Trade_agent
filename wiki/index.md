# Index

> Content catalog for the trading & quant second brain. Organized by category. ZCode updates this on every ingest. **Read this first** when answering a query — then drill into pages.
>
> **State:** 2 sources · 24 pages · last updated 2026-06-26

---

## Overviews
_Synthesis pages — the big picture of a topic._

_(none yet)_

## Concepts
_Methods and ideas, synthesized across all sources._

- [[order-flow-imbalance]] — signed order-book pressure; the cleanest mechanical price-formation signal. Now-pressure, not a forecast. (1 source, updated 2026-06-25)
- [[tick-rule-classification]] — trade direction proxy comparing price to previous trade; standard when bid/ask unavailable. (1 source, updated 2026-06-26)
- [[cumulative-volume-delta]] — running sum of signed trade volume; net buying/selling pressure. Complement to OFI. (1 source, updated 2026-06-26)
- [[build-your-own-x]] — meta-learning philosophy: build from scratch to understand. 200+ tutorials across 30+ CS categories. (1 source, updated 2026-06-26)
- [[custom-trading-brain-architecture]] — proposed LSTM/Transformer brain architecture for temporal pattern recognition. (1 source, updated 2026-06-26)

## Strategies

_(none yet)_

## Indicators

_(none yet)_
_Specific trading strategies._

_(none yet)_

## Indicators
_Signals, features, and metrics._

_(none yet)_

## Entities
_Instruments, venues, tools, people, firms._

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
- [[neural-network-brain]] — brain #10; custom LSTM for temporal patterns, weight 0.05. (stable, updated 2026-06-26)

## Comparisons
_Analyses and side-by-side comparisons (often filed from queries)._

_(none yet)_

## Playbooks
_How-tos, processes, checklists._

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
