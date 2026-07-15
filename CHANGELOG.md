# Changelog

All notable project changes are recorded here. This file follows Keep a Changelog structure; releases remain versioned by the project release process.

## [Unreleased]

### Added

- A fully local project-knowledge pipeline contract binding architecture ownership, one primary responsibility, code, durable documentation, and embedding inputs in `project.manifest.toml`.
- Deterministic `project.manifest.lock.json` generation and an offline fail-closed CI drift check.
- Local ChromaDB 1.5.9 server synchronization and verification with Ollama `nomic-embed-text`; no cloud inference is used.
- A machine-local synchronization receipt at `.local/knowledge/sync-receipt.json` and lock/check/sync/verify/query operations guidance.
- Architecture decision and operations documentation in `wiki/content/decisions/local-knowledge-pipeline-v1.md` and `wiki/content/playbooks/local-knowledge-pipeline-operations.md`.

### Changed

- Scoped project-knowledge storage to ChromaDB while leaving PostgreSQL ledger state and legacy trading-pattern Qdrant/NIM paths unchanged.

