---
title: TradingAgents
type: source
tags: [multi-agent, langgraph, workflow, memory, open-source]
created: 2026-07-16
updated: 2026-07-16
sources: []
source_url: https://github.com/TauricResearch/TradingAgents
source_type: repository
authors: [Tauric Research]
date: 2026-07-05
revision: 01477f9afb7a47b849ed4c9259d3a9a4738d9fda
license: Apache-2.0
status: stable
---

# TradingAgents

> **TL;DR** — TradingAgents is a synchronous LangGraph research scaffold. Its reusable value is typed graph orchestration, exhaustive routing, bounded rounds, structured artifacts, checkpoint identity, and append-only memory—not its trading roles or provider stack.

## Key takeaways

- The composition root builds analyst, debate, synthesis, risk-perspective, and portfolio nodes around a typed shared state.
- Conditional loops are bounded, and route maps enumerate every possible target.
- Operational checkpoints are separated by instrument and graph shape.
- Markdown memory uses idempotent pending/resolved records and bounded retrieval context.
- The repository has no FastAPI service, asynchronous runtime, broker adapter, or real order-submission implementation.
- Apache-2.0 permits reuse with its attribution obligations; this project adopts patterns through a clean local implementation rather than vendoring the upstream graph.

## Summary

The upstream graph runs analysts and tool loops, a bounded bull/bear discussion, a synthesis node, a proposal node, three risk perspectives, and a final manager. The architecture demonstrates useful workflow mechanics, but the role semantics let LLMs participate in risk and portfolio judgments. That authority boundary conflicts with Trade_agent, where deterministic code alone owns risk and execution.

The adopted patterns are declarative node specifications, typed state, complete conditional maps, bounded execution, structured Pydantic artifacts, graph-revision-aware checkpoints, and append-only memory receipts. Synchronous invocation, cloud providers, free-text fallbacks, and direct hot-path Markdown writes are rejected.

## Concepts introduced / updated

- [[open-source-agent-integration-analysis]] — compares the graph patterns against all four reference systems.
- [[hermes-local-integration-v1]] — records the clean local implementation boundary.
- [[multi-agent-pipeline]] — retains deterministic risk and execution outside model-controlled orchestration.

## Relevance to our system

TradingAgents informs the bounded DAG and provenance contracts in `packages/hermes`. It does not replace the existing trading pipeline, risk engine, broker abstraction, or execution ledger. Local Ollama is the only inference boundary, and Obsidian projection occurs asynchronously outside the decision loop.

## Contradictions / updates

- Upstream documentation describes a trading framework, but the inspected revision produces research outputs rather than implementing a broker execution path.
- Upstream LLM risk and portfolio roles are explicitly incompatible with Trade_agent's deterministic authority boundary.

## Open questions

- Whether a later checkpoint store is necessary after real workload traces show resumability value.
- Whether generic DAG definitions should eventually compile to LangGraph or remain dependency-light Python.

## Primary evidence

- [Repository and README](https://github.com/TauricResearch/TradingAgents/tree/01477f9afb7a47b849ed4c9259d3a9a4738d9fda)
- [Apache-2.0 license](https://github.com/TauricResearch/TradingAgents/blob/01477f9afb7a47b849ed4c9259d3a9a4738d9fda/LICENSE)
- [Graph setup](https://github.com/TauricResearch/TradingAgents/blob/01477f9afb7a47b849ed4c9259d3a9a4738d9fda/tradingagents/graph/setup.py)
- [Typed state](https://github.com/TauricResearch/TradingAgents/blob/01477f9afb7a47b849ed4c9259d3a9a4738d9fda/tradingagents/agents/utils/agent_states.py)
- [Conditional routing](https://github.com/TauricResearch/TradingAgents/blob/01477f9afb7a47b849ed4c9259d3a9a4738d9fda/tradingagents/graph/conditional_logic.py)
- [Checkpoint identity](https://github.com/TauricResearch/TradingAgents/blob/01477f9afb7a47b849ed4c9259d3a9a4738d9fda/tradingagents/graph/checkpointer.py)
- [Markdown memory](https://github.com/TauricResearch/TradingAgents/blob/01477f9afb7a47b849ed4c9259d3a9a4738d9fda/tradingagents/agents/utils/memory.py)
