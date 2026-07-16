---
title: Multi-Agent Trading Pipeline
type: concept
tags: [pipeline, langgraph, multi-agent, experimental, legacy, ollama]
created: 2026-07-01
updated: 2026-07-16
sources:
  - "[[canonical-local-paper-runtime-v1]]"
  - "[[open-source-agent-integration-analysis]]"
status: draft
---

# Multi-Agent Trading Pipeline

## Current classification

The repository's LangGraph/VLM/RAG/swarm pipeline is an experimental pre-canonical design. It is not the default runtime, not a risk authority, and not an execution path. Its Compose services are quarantined behind the explicit `legacy` profile. [[canonical-local-paper-runtime-v1]] is the active paper/replay boundary.

## Historical design

The original graph intended to coordinate:

```text
candle ingest
  -> feature extraction
  -> CPU scalping branch or VLM/RAG/swarm branch
  -> risk node
  -> simulated execution node
  -> logging/publication
```

The design explored specialist-agent composition, conditional routing, trace propagation, and bounded fallback. Those are reusable orchestration patterns, but the historical implementation cannot be promoted as a trading hot path because it used synthetic risk state, simulated execution, inconsistent event contracts, and unverified latency claims.

## Reusable engineering patterns

Only the following patterns are candidates for clean integration:

- typed immutable graph state;
- explicit node input/output schemas;
- bounded timeouts and deterministic fallbacks;
- parallel advisory analysis where GPU capacity permits;
- trace and provenance propagation;
- human-reviewed promotion of model output into a candidate envelope;
- replay fixtures for every graph transition.

These patterns may produce `CandidateForRiskEvent` only through a separately owned candidate-producer service. They must never emit approvals, order intents, or broker calls.

## Canonical integration boundary

```text
optional local agent workflow
  -> provider=ollama + exact model role/name/digest
  -> CandidateForRiskEvent
  -> signals.candidate.v1
  -> deterministic decision-worker
  -> risk.approved.v1 | risk.rejected.v1
```

The canonical runtime currently supplies no candidate producer. Therefore the legacy LangGraph graph is not silently connected to `signals.candidate.v1`. Connecting it requires a new component owner, manifest authority, contract tests, model-locality tests, replay evidence, and an ADR.

## Hermes separation

[[hermes-local-integration-v1]] applies upstream agent patterns to software-engineering and research workflows only. Hermes can call the approved local Ollama registry and persist engineering knowledge to Obsidian/Chroma, but it cannot publish trading candidates, approve risk, access brokers, or execute orders. This prevents a generic agent framework from becoming an alternate trading authority.

## Known gaps before reconsideration

- No canonical candidate-producer interface or service lifecycle.
- No deterministic graph checkpoint/replay contract bound to the v1 event schemas.
- No measured RTX 4050 concurrency, VRAM, or end-to-end latency budget.
- No prompt/model-digest promotion registry for candidate-producing workflows.
- No evidence that VLM/RAG/swarm output adds out-of-sample value over a model-free baseline.
- Historical `orchestrator/` event, risk, and execution modules remain incompatible with the canonical worker authority boundary.

## Revisit criteria

Revisit this page only after the deterministic core has a bootstrapped risk state, transactional event processing, restart reconciliation, and a tested candidate-producer port. Any promoted graph must remain upstream of deterministic risk and function with execution disabled.

## Related

- [[canonical-local-paper-runtime-v1]]
- [[deterministic-paper-core-v1]]
- [[nats-event-system]]
- [[local-trading-ai-architecture]]
- [[open-source-agent-integration-analysis]]
- [[hermes-local-integration-v1]]
- [[nats-langgraph-bridge]]
- [[vlm-agent]]
- [[rag-agent]]
- [[scalping-engine]]
