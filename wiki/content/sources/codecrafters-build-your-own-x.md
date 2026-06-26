---
title: "Build Your Own X — CodeCrafters Guides"
type: source
tags: [meta-learning, systems-engineering, tutorials, reference]
created: 2026-06-26
updated: 2026-06-26
source_path: raw/build-your-own-x.md
source_type: article
authors: [CodeCrafters, Daniel Stefanovic, community contributors]
date: 2024-01-01
status: stable
---

# Build Your Own X — CodeCrafters Guides

> **TL;DR** — A curated index of 200+ step-by-step tutorials for building core CS technologies from scratch, spanning 30+ categories. The core philosophy: *What I cannot create, I do not understand* (Feynman). A meta-resource for deepening implementation-level understanding of every layer of the stack we use.

## Key takeaways

- **Feynman learning principle applied at scale** — building from scratch forces understanding of internals that API-level usage never reveals. Directly applicable to our system: we build our own execution engine (Rust), orchestrator (Go), and brains (Python) rather than using off-the-shelf solutions.
- **30 technology categories** from distributed systems to neural networks to operating systems. The most relevant to Trade_agent: neural networks, databases, web servers, blockchain, command-line tools, and search engines.
- **Multi-language coverage** — every topic has implementations in 3–5+ languages. Heavy presence of C, C++, Python, Go, Rust, and JavaScript. Matches our polyglot stack (Python/Rust/Go/TypeScript).
- **Deep learning from scratch** tutorials (Neural Networks section) — 17 entries covering perceptrons, CNNs, LSTM, PyTorch internals, and OCR. Directly relevant to [[finrl-brain]], [[freqai-brain]], and future custom model work.
- **Database internals** — 13 entries covering Redis, B+Tree, KV stores, graph databases. Relevant to our data pipeline and caching strategy.
- **Blockchain/cryptocurrency** — 20+ entries across 10 languages. Directly informs [[onchain-brain]] depth.
- **Web server & browser engineering** — foundational for our Next.js frontend and potential websocket infrastructure.
- **No trading-specific tutorials** in the collection — this is a general-purpose CS learning resource, not a quant library.

## Summary

CodeCrafters' "Build your own X" is the definitive open-source index of implementation-level tutorials for core computer-science technologies. Organized by category (3D rendering, AI, databases, Docker, compilers, neural networks, OS, etc.), each entry links to a detailed tutorial where the reader re-implements a known system from scratch in a specific language.

The collection's value for Trade_agent is as a **reference curriculum** — a structured path for any team member (human or AI) who needs to deepen understanding of a subsystem before modifying it. When we need to tune our Rust execution engine, understanding how a database or network stack works from scratch accelerates debugging. When we extend our ML brains, building neural networks from scratch clarifies what PyTorch abstracts away.

The collection also serves as a **benchmark for our system's sophistication** — we already implement several of these patterns (custom execution engine in Rust, orchestrator in Go, custom ML pipelines). The question is: which of these tutorials could improve our architecture?

## Concepts introduced / updated

- [[build-your-own-x]] — the meta-learning philosophy and curated resource index

## Relevance to our trading

| Category | Trade_agent component | Actionable link |
|---|---|---|
| Neural Networks | [[finrl-brain]], [[freqai-brain]], [[finbert-brain]] | From-scratch NN tutorials → understand PyTorch internals before tuning RL |
| Database | data pipeline, [[microstructure]] tick storage | B+Tree / KV store internals → inform schema design |
| Web Server | frontend (Next.js), realtime (Go) | HTTP internals → optimize API layer |
| Blockchain | [[onchain-brain]] exchange flow data | Build-your-own-blockchain → understand on-chain mechanics deeply |
| Search Engine | wiki search, signal ranking | TF-IDF / vector space → improve signal search |
| Distributed Systems | Go orchestrator, NATS messaging | Kafka-like system → design message bus patterns |
| Command-Line Tool | CLI interface | Rust CLI apps → improve our tooling |

## Contradictions / updates

- None. This is a meta-resource, not a technical claim. No contradiction with existing wiki content.

## Karpathy Course Deep Dive

The most valuable resource from this collection for our system is Andrej Karpathy's "Neural Networks: Zero to Hero" (YouTube, 8 videos, ~15 hours):

| Video | Topic | Trading Relevance |
|---|---|---|
| 1. Micrograd | Autograd + backprop | Custom loss functions (penalize drawdown) |
| 2-5. Makemore | MLP, activations, batch norm | Model diagnostics, overfitting detection |
| 6. WaveNet | Convolutional sequence model | Time-series prediction architecture |
| 7. Let's build GPT | Transformer from scratch | **Attention mechanism for regime detection** |
| 8. Tokenization | BPE tokenizer | NLP brain (finbert) improvements |

**Key insight**: Karpathy does NOT cover LSTM/GRU — he builds toward Transformers. Expert analysis confirms this is the right direction: LSTMs overfit on market noise; Transformers capture long-range dependencies better.

## Open questions

- Should we start with LSTM for prototyping or go directly to Transformer?
- What sequence length (50, 100, 200 candles) works best for BTCUSDT?
- How to build a custom loss function that penalizes drawdown more than MSE?
- Can attention weights be used to dynamically adjust brain weights based on regime?
