# Log

> Append-only timeline of every operation on the wiki. Newest at top. Reverse chronological so the current state is the first thing you read.
>
> Entry format: `## [YYYY-MM-DD] <op> | <subject>` where `<op>` is `ingest` · `query` · `lint` · `note`.
> Quick last-5: `grep "^## \[" log.md | tail -5`

---

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
