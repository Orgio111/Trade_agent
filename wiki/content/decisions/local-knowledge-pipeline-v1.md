---
title: Local Knowledge Pipeline v1
type: decision
tags:
  - architecture
  - knowledge-management
  - manifest
  - chromadb
  - ollama
  - embeddings
  - ci
created: 2026-07-15
updated: 2026-07-16
sources:
  - "[[trade-project-full-integration-build-plan]]"
  - "[[rag-agent]]"
  - "[[rtx4050-trading-system]]"
status: stable
---

# Local Knowledge Pipeline v1

## Context

Project knowledge was split between code, Obsidian pages, architecture decisions, and several unrelated vector-memory experiments. A code change could therefore leave its documentation, ownership declaration, or embeddings stale without a deterministic CI failure. The vector-store choices in [[trade-project-full-integration-build-plan]] and [[rag-agent]] also describe runtime trading memory, not the repository's permanent project knowledge.

The project needs one local, reviewable contract that answers four questions for every indexed component:

1. Who owns it?
2. What is its one primary responsibility?
3. Which code and durable documentation belong to it?
4. Which exact content chunks should exist in the project-knowledge index?

## Options considered

| Option | Strength | Rejected because |
|---|---|---|
| Documentation conventions only | No runtime dependency | Cannot deterministically detect code/documentation/embedding drift |
| PostgreSQL + pgvector | Reuses the future authoritative trading ledger database | Couples repository knowledge to runtime trading infrastructure and prevents a lightweight offline-first workflow |
| Qdrant | Strong filtered vector search | Adds another service and would blur the boundary with the legacy trading-pattern path in [[rag-agent]] |
| ChromaDB + local Ollama | Local HTTP service, simple metadata filtering, no cloud inference | Requires an explicit lock and verification protocol to make index state auditable |

## Decision

Adopt a dedicated **project-knowledge** pipeline with:

- one human-authored `project.manifest.toml` that maps component ID, owner, one primary responsibility, code paths, documentation paths, and embedding inputs;
- repository-wide tracked-file discovery for production code, configuration, and durable documentation, with narrow declared exclusions for generated, raw, machine-local, and legacy artifacts;
- a generated deterministic `project.manifest.lock.json` that hashes every governed owned text file and separately records normalized embedding-source chunks, chunk IDs, and embedding-model identity;
- **ChromaDB 1.5.9** in local Docker server mode (`chroma`, `127.0.0.1:8100` → container port 8000) as the project-knowledge vector store;
- local Ollama **`nomic-embed-text`** as the only embedding model;
- a local, ignored `.local/knowledge/sync-receipt.json` written only after a successful live synchronization;
- a fail-closed CI check that needs neither Ollama nor ChromaDB and rejects manifest, ownership, documentation, exclusion, or lock drift;
- explicit local `sync` and `verify` operations for live vector-store state.

The TOML manifest is the ownership and indexing contract. Obsidian pages under `wiki/` remain the durable source of truth; ChromaDB and the local receipt are rebuildable retrieval projections, never authoritative copies.

Runtime Hermes journal events are governed separately by [[hermes-local-integration-v1]]. They remain canonical Markdown in the explicitly configured external Obsidian vault and project only into `trade-agent-runtime-memory-v1`; they never enter the project collection `trade-agent-project-knowledge-v1`. The manifest now also pins upstream repository revision, license, and reuse policy for components derived from external reference systems.

## Scope boundary and supersession

This decision supersedes the pgvector/Qdrant recommendations in [[trade-project-full-integration-build-plan]] and [[rag-agent]] **only for project knowledge**: source code, architecture, documentation, research synthesis, and lessons learned.

It does not migrate or redesign:

- the PostgreSQL execution ledger;
- trading journal or event-time strategy memory;
- `orchestrator/rag_agent.py` or `orchestrator/memory/vector_memory.py`;
- Qdrant/NIM experiments used by the legacy trading-pattern RAG path;
- any strategy, signal, risk, sizing, broker, or execution behavior.

Those trading-memory paths are baseline-frozen. They remain ownership-hashed governed files in the lock, but are excluded from its embedding `sources` and from ChromaDB. Any addition or removal fails manifest validation until a separate, explicitly approved architecture decision updates the baseline.

## Manifest, lock, and receipt contract

`project.manifest.toml` is the single human-authored declaration of:

- schema and collection version;
- local Ollama and Chroma server settings;
- component ownership and exactly one primary responsibility;
- code, documentation, architecture, research, and lesson-learned inputs;
- deterministic chunking rules;
- executable text-type, repository-path, and security-exclusion rules.

`project.manifest.lock.json` records content-derived identities, not mutable runtime state. Its `governed_files` inventory binds each normalized path and canonical UTF-8 hash to its component and owner, including owned tests, configuration, and deployment files that are intentionally not embedded. Stable chunk IDs are derived from the project, embedding contract, chunker version, normalized path, chunk index, and chunk content hash. The generated lock file does not hash itself because `check` already compares its complete canonical bytes with regenerated evidence. A rename, content change, or contract change therefore creates explicit drift instead of silently overwriting unrelated vectors.

`.local/knowledge/sync-receipt.json` records the local synchronization evidence required by `verify`. It is machine-local and ignored by Git because a receipt from one Chroma instance cannot prove another instance is synchronized.

`sync` also acquires a machine-global, collection-scoped writer lease keyed by the local Chroma endpoint and collection identity. The lease lives in the host temporary directory, so concurrent clones or worktrees on the same machine cannot mutate the same collection simultaneously. It coordinates writers only; the manifest, lock, receipt, and live verification remain the evidence of correctness.

CI must fail when:

- a declared path is missing, outside the repository, duplicated across owners, or matched by no component;
- a relevant tracked production/configuration/documentation file appears under an undeclared root without exactly one owner;
- a component has zero or multiple primary responsibilities;
- required code lacks a durable documentation owner;
- an embedding input is a secret or literal YAML/TOML/env credential, raw source, generated artifact, cache, binary, symlink escape, or temporary file;
- normalized governed-file or embedding source/chunk hashes differ from the checked-in lock;
- the manifest, chunking policy, collection version, or embedding-model identity changes without a refreshed lock.

## Offline CI versus live verification

The two guarantees are intentionally separate:

| Check | Requires Ollama/ChromaDB | Proves |
|---|:---:|---|
| `lock` | No | The expected deterministic source/chunk inventory can be reproduced |
| `check` | No | The TOML manifest and checked-in JSON lock match the repository and ownership rules |
| `sync` | Yes | Expected chunks were embedded locally, reconciled into ChromaDB, and acknowledged by a local receipt |
| `verify` | Local Ollama + ChromaDB | The model identity, collection, and local receipt match the current manifest lock |
| `query` | Local Ollama + ChromaDB | Local semantic retrieval works against verified project knowledge |

An offline CI pass does **not** prove that a developer's Chroma server is running or synchronized. A regenerated lock alone is also not evidence of successful embedding. Live readiness requires `sync` followed by `verify`; operator workflows that depend on retrieval must run both.

## Failure semantics

- Embedding failures never produce zero vectors or hash-based pseudo-embeddings.
- A partial Ollama/ChromaDB failure does not write a successful synchronization receipt.
- Concurrent `sync` writers for the same local collection fail closed under the machine-global lease.
- Model, dimension, collection, or chunk-policy mismatch fails closed and requires a deliberate rebuild/version change.
- Stale Chroma records are removed only through manifest-scoped reconciliation; unrelated collections are never deleted.
- Query refuses a missing, unverified, or drifting receipt/collection instead of returning potentially stale context.
- Secret and temporary-file exclusions apply before text is admitted as an embedding input.

## Rationale

The boundary keeps permanent project memory local and reproducible without making PostgreSQL, NATS, or a trading worker prerequisites for documentation work. ChromaDB is acceptable here because it is a disposable projection with a deterministic rebuild contract. Pinning ChromaDB 1.5.9 and `nomic-embed-text` makes service and vector compatibility explicit instead of accidental.

The design also prevents knowledge tooling from acquiring trading authority. Retrieval can inform engineering work, but it cannot approve risk or place orders, consistent with [[deterministic-paper-core-v1]].

## Revisit trigger

Revisit this decision only if a reproducible local benchmark shows that ChromaDB cannot meet corpus size, filtered-retrieval, backup, or verification requirements. Any replacement must preserve the manifest, deterministic lock, local receipt, local Ollama embedding boundary, fail-closed CI, and trading-memory quarantine.

## Related

- [[local-knowledge-pipeline-operations]]
- [[trade-project-full-integration-build-plan]]
- [[rag-agent]]
- [[rtx4050-trading-system]]
- [[deterministic-paper-core-v1]]
