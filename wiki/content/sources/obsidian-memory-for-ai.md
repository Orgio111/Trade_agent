---
title: Obsidian Memory for AI
type: source
tags: [obsidian, memory, operations, provenance, clean-room]
created: 2026-07-16
updated: 2026-07-16
sources: []
source_url: https://github.com/jrcruciani/obsidian-memory-for-ai
source_type: repository
authors: [J. R. Cruciani]
date: 2026-07-16
revision: 7af41fe60a068e4440022d3507c4a3aee81892d0
license: none-declared
status: stable
---

# Obsidian Memory for AI

> **TL;DR** — The specification offers strong memory concepts—canonical path identity, atomic facts, append-only events, proposal envelopes, precondition hashes, receipts, and generated views—but has no declared license. Only independently implemented concepts may be used.

## Key takeaways

- Human narrative, machine facts, events, proposals, receipts, claims, and generated views have separate storage responsibilities.
- Paths act as stable record identities; frontmatter defines the type surface.
- Event records are append-only; mutable facts use temporal provenance and supersession.
- Agents propose operations into scoped inboxes; validation and precondition checks precede application.
- Applied receipts and regenerated views make derived state auditable and rebuildable.
- The inspected repository has no license file or SPDX declaration, so its code, schemas, templates, and prose are not copied.

## Summary

The repository is a filesystem protocol and example toolkit rather than an Obsidian plugin, model service, vector database, FastAPI server, or daemon. Its multi-agent coordination is based on operation IDs, scoped inboxes, advisory claims, optimistic hashes, and receipts. This is valuable for memory governance but requires stronger resolved-path containment, symlink defense, identity policy, and explicit mutation allowlists.

Trade_agent v1 adopts only an independently designed append-only event subset. Canonical Markdown is written before a separate Chroma projection, and no model gets a raw apply or filesystem-write capability.

## Concepts introduced / updated

- [[open-source-agent-integration-analysis]] — records the clean-room extraction and rejected risks.
- [[hermes-local-integration-v1]] — selects append-only events and projection receipts for v1.
- [[hermes-memory-operations]] — defines safe operational procedures.
- [[local-knowledge-pipeline-v1]] — preserves Markdown authority over vector projections.

## Relevance to our system

The proposal/validate/apply pattern is the later path for mutable facts. The implemented v1 slice is deliberately smaller: validated immutable journal events, deterministic paths, secret scanning, create-only writes, idempotency checks, and an isolated retrieval index.

## Contradictions / updates

- The request described this repository as open source; no license grants open-source reuse rights.
- Advisory claim files are not locks and cannot establish trusted actor identity.

## Open questions

- When mutable project facts are introduced, define an explicit human approval and compare-and-swap policy before adding an apply endpoint.

## Primary evidence

- [Repository](https://github.com/jrcruciani/obsidian-memory-for-ai/tree/7af41fe60a068e4440022d3507c4a3aee81892d0)
- [v3 specification](https://github.com/jrcruciani/obsidian-memory-for-ai/blob/7af41fe60a068e4440022d3507c4a3aee81892d0/SPEC-v3.md)
- [Automation guide](https://github.com/jrcruciani/obsidian-memory-for-ai/blob/7af41fe60a068e4440022d3507c4a3aee81892d0/automation-guide.md)
- [Example operation tooling](https://github.com/jrcruciani/obsidian-memory-for-ai/blob/7af41fe60a068e4440022d3507c4a3aee81892d0/examples/v3-minimal-vault/tools/ops.py)
- [Example lint tooling](https://github.com/jrcruciani/obsidian-memory-for-ai/blob/7af41fe60a068e4440022d3507c4a3aee81892d0/examples/v3-minimal-vault/tools/lint.py)
