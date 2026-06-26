# Log

> Append-only timeline of every operation on the wiki. Newest at top. Reverse chronological so the current state is the first thing you read.
>
> Entry format: `## [YYYY-MM-DD] <op> | <subject>` where `<op>` is `ingest` · `query` · `lint` · `note`.
> Quick last-5: `grep "^## \[" log.md | tail -5`

---

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
