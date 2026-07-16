# Changelog

All notable project changes are recorded here. This file follows Keep a Changelog structure; releases remain versioned by the project release process.

## [Unreleased]

### Added

- A canonical local paper/replay runtime with three async Python workers, a read-only FastAPI control plane, exact local Ollama role/provenance contracts, and a bounded `QUANTEX_CORE` JetStream subject set.
- PostgreSQL runtime durability migration 003 with event inbox/outbox/offset scaffolding, immutable portfolio snapshots and instrument constraints, single-signal verdict/candidate-event binding, and position/tax-lot projection schemas.
- Durable JetStream and PostgreSQL adapters for explicit ACK/redelivery, producer deduplication IDs, authoritative risk inputs, atomic candidate/verdict persistence, exact approval lookup, and paper-only execution ledger recovery.
- Architecture and deployment documentation in `canonical-local-paper-runtime-v1`, including explicit bootstrap, candidate-producer, transactional inbox/outbox, DLQ, projection, and reconciliation gaps.
- A clean-room, local-only Hermes integration layer with strict agent/event contracts, bounded asynchronous DAG workflows, exact Ollama model-role routing, and serialized GPU inference by default.
- A create-only Obsidian event journal with UTC paths, content-derived identity, secret scanning, symlink/junction containment, machine-local per-event leases, idempotent receipts, and conflict detection.
- A separate `trade-agent-runtime-memory-v1` Chroma projection using local `nomic-embed-text`, canonical Markdown provenance, persist-first failure semantics, and bounded retrieval.
- A loopback-only FastAPI control plane for journal ingest, runtime-memory query, workflow execution, and health, runnable with `python -m packages.hermes` (and declared as `quantex-hermes` for packaged installs).
- License-aware source pages, comparison, ADR, and operations documentation for TradingAgents, Journalit, Obsidian AI, and Obsidian Memory for AI.
- Manifest-pinned upstream repository revisions, licenses, and reuse policies, together with strict Hermes lint, formatting, typing, unit, integration, and offline latency benchmark coverage.
- A fully local project-knowledge pipeline contract binding architecture ownership, one primary responsibility, code, durable documentation, and embedding inputs in `project.manifest.toml`.
- Deterministic `project.manifest.lock.json` generation that hashes every governed owned text file, records separate embedding-source chunks, and drives an offline fail-closed CI drift check.
- Local ChromaDB 1.5.9 server synchronization and verification with Ollama `nomic-embed-text`; no cloud inference is used.
- A machine-local synchronization receipt at `.local/knowledge/sync-receipt.json` and lock/check/sync/verify/query operations guidance.
- Architecture decision and operations documentation in `wiki/content/decisions/local-knowledge-pipeline-v1.md` and `wiki/content/playbooks/local-knowledge-pipeline-operations.md`.

### Changed

- Split Chroma namespace initialization from live verification so `verify` and `query` open an existing collection read-only and fail closed when it is absent.
- Made the canonical local paper runtime the default Docker Compose graph; pre-canonical orchestration, cloud/vLLM inference, Go/Rust runtime, legacy risk/execution, frontend, proxy, and observability services now require the explicit `legacy` profile.
- Added a frozen minimal `workers` dependency group for the shared non-root canonical image, excluding legacy ML and research dependencies from the default runtime build.
- Removed inactive legacy-profile interpolation blockers from the default Compose render; only `POSTGRES_PASSWORD` is required for the canonical graph, while legacy Grafana still fails closed without its own explicit password.
- Bound infrastructure ports to loopback, kept execution limited to `paper_live`/`replay` and `PaperBroker`, and changed legacy mutation/risk automation surfaces to fail closed without changing strategy or trading logic.
- Synchronized the Obsidian architecture, event, model, deployment, and build-plan pages with the active runtime and marked the historical LangGraph multi-agent pipeline as experimental/quarantined.
- Extended the architecture manifest and generated lock with the single-responsibility `hermes-integration` component and external-reference provenance; no strategy, signal, risk, broker, or execution behavior changed.
- Extracted the existing machine-local process lease into a shared knowledge utility so project and runtime projections use one cross-platform locking implementation.
- Scoped project-knowledge storage to ChromaDB while leaving PostgreSQL ledger state and legacy trading-pattern Qdrant/NIM paths unchanged.
- Pinned the Chroma Python client to 1.5.9 across dependency entry points, extended CI drift enforcement to `codex/**` branches, and removed committed Compose credential literals in favor of explicit local environment configuration.
- Documented the machine-global, collection-scoped synchronization lease and aligned the operations contract with executable path/type admission rules.
- Declared and pinned the NATS, CCXT, LangGraph, AutoGen, metrics, and legacy ML test dependencies used by the collected orchestration test surface.
- Added a locked local development dependency group so test, lint, and type-check tools survive reproducible `uv sync` operations.
- Changed the CI test job to build its complete environment from the frozen `uv.lock` instead of an incomplete hand-maintained package subset.
