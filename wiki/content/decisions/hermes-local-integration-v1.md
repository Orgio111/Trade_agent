---
title: Hermes Local Integration v1
type: decision
tags: [architecture, hermes, ollama, obsidian, chromadb, multi-agent, local-first]
created: 2026-07-16
updated: 2026-07-16
sources:
  - "[[open-source-agent-integration-analysis]]"
  - "[[tradingagents]]"
  - "[[journalit]]"
  - "[[obsidian-ai]]"
  - "[[obsidian-memory-for-ai]]"
status: stable
---

# Hermes Local Integration v1

## Context

Trade_agent needs a safe engineering integration between Hermes, approved local Ollama models, permanent Obsidian memory, and Chroma retrieval. Existing trading risk, execution, and legacy vector paths must remain unchanged. The reference repositories contain valuable patterns but also incompatible trading authority, cloud adapters, direct filesystem access, proprietary code, and unlicensed code.

The new component must also preserve the ownership and drift guarantees in [[local-knowledge-pipeline-v1]] while keeping runtime journal records out of the project-knowledge collection.

## Options considered

| Option | Strength | Rejected because |
|---|---|---|
| Vendor TradingAgents and Journalit | Fast visible feature growth | Brings incompatible trading semantics, sync loops, provider dependencies, and prohibited proprietary code |
| Let Hermes write directly into the vault | Minimal backend code | No path, schema, secret, idempotency, or concurrency boundary |
| Put all memory in one Chroma collection | Simple query setup | Blurs canonical project knowledge and runtime records; makes drift and deletion unsafe |
| Clean-room local integration component | Explicit authority, provenance, and testing | Requires new contracts and operational tooling |

## Decision

Create one `hermes-integration` component under `packages/hermes` with exactly one primary responsibility: accept non-authoritative local-agent artifacts and project validated events into permanent Obsidian memory plus a disposable runtime retrieval index.

The component provides:

- strict immutable Pydantic contracts;
- exact role-to-model mappings for `qwen3:8b`, `phi3:3.8b`, `deepseek-r1:8b`, `moondream`, `mistral`, and `nomic-embed-text`;
- a loopback-only Ollama adapter with no provider fallback;
- a bounded asynchronous dependency graph with task, concurrency, timeout, and deadline limits;
- an append-only Obsidian repository with resolved-root containment, safe paths, create-only idempotency, checksum conflict detection, and secret scanning;
- a persist-first application service: canonical Markdown succeeds before Chroma indexing starts;
- an isolated Chroma namespace named `trade-agent-runtime-memory-v1` using the existing local `OllamaEmbedder` and `ChromaVectorStore` primitives;
- a loopback FastAPI surface for event ingest, memory query, workflow execution, and health;
- pinned upstream repository, revision, license, and reuse policy in `project.manifest.toml`.

## Authority boundaries

The integration is non-authoritative:

- it never creates signals, orders, trades, position sizes, or risk overrides;
- it does not import or call broker and execution packages;
- it does not replace local models;
- it does not alter strategies, risk policies, execution state machines, or the existing trading LangGraph;
- model output remains an artifact until another existing deterministic subsystem validates any domain-specific use;
- no raw vault path, shell, cloud provider, or automatic mutable-memory apply tool is exposed.

## Canonical and derived state

```text
JournalEvent
  -> Obsidian event Markdown        canonical, append-only
  -> content checksum receipt      durable identity evidence
  -> nomic-embed-text vector       derived
  -> runtime Chroma collection     disposable and rebuildable
```

The project wiki remains canonical project knowledge and is indexed in `trade-agent-project-knowledge-v1`. Runtime Hermes journal events belong to the configured external Obsidian vault and are indexed in `trade-agent-runtime-memory-v1`. Neither collection is queried as a fallback for the other.

## Failure semantics

- Invalid schema, remote URL, unexpected model, unsafe path, secret-like content, zero vector, dimension drift, or checksum conflict fails closed.
- If Obsidian append fails, no embedding begins.
- If Chroma/Ollama projection fails after the append, the event remains canonical and the API reports a retryable projection failure with its journal path.
- Replaying the same event ID and checksum is idempotent; the same ID with different content is a conflict.
- Workflow dependency failure blocks downstream tasks. Deadline expiry yields explicit terminal states.

## Consequences

Positive:

- Local-only authority and model boundaries become testable.
- Runtime memory is auditable from Markdown provenance and can be rebuilt.
- Multi-agent engineering work gains bounded concurrency without entering the trading hot path.
- CI detects ownership, documentation, source, license provenance, and embedding drift.

Trade-offs:

- v1 supports immutable events, not arbitrary fact mutation or collaborative edit merging.
- FastAPI workflow calls return completed runs rather than token streaming.
- Full restart reconciliation and durable outbox processing remain future work.
- On OneDrive/NTFS, a high event rate would create small-file overhead; callers must journal durable summaries, not candles or token deltas.

## Revisit triggers

- A real need for human-edited mutable facts requires proposal, review, compare-and-swap, and applied-receipt semantics.
- Measured projection loss after process failure justifies a durable outbox and replay cursor.
- Workflow traces justify LangGraph checkpointing or server-sent progress events.
- Runtime collection growth requires retention tiers or deterministic compaction.

## Related

- [[open-source-agent-integration-analysis]]
- [[hermes-memory-operations]]
- [[local-knowledge-pipeline-v1]]
- [[multi-agent-pipeline]]
- [[local-trading-ai-architecture]]
