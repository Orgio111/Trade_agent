---
title: Journalit
type: source
tags: [obsidian, journaling, projection, proprietary, clean-room]
created: 2026-07-16
updated: 2026-07-16
sources: []
source_url: https://github.com/Cursivez/journalit
source_type: repository
authors: [Cursivez]
date: 2026-07-16
revision: 00f1f7044b0a6ab5b49bda682b582f87741c5d09
license: proprietary-source-available
status: stable
---

# Journalit

> **TL;DR** — Journalit demonstrates useful observable journaling behavior, but it is not open source. Trade_agent uses only independently reimplemented ideas: canonical Markdown projection, stable identity, staged startup, receipts, disposable indexes, and preserved human-owned note regions.

## Key takeaways

- The public repository is source-available proprietary software; copying, modification, redistribution, and derivative works require permission.
- The plugin separates lifecycle initialization, services, commands, views, processors, typed events, canonical Markdown records, and disposable indexes.
- Observable workflows follow preview/commit/project/acknowledge phases.
- Stable identifiers and schema/template revisions are more durable than filenames.
- Human-authored note bodies and generated structured fields have separate ownership.
- Optional authentication, backend synchronization, import, and exchange-rate network paths violate the fully local boundary and are rejected.

## Summary

Journalit is an Obsidian trading-journal plugin rather than an agent system. It has no LLM agent graph, shared agent memory, or multi-agent coordination. Its engineering relevance lies in local Markdown as a human-readable projection, typed mutation services, staged startup, cache invalidation, stable IDs, review hierarchy, and projection acknowledgements.

The implementation is not reusable under its license. All adopted behavior is expressed as a clean-room design in [[hermes-local-integration-v1]] and deliberately avoids Journalit's classes, schemas, templates, UI, remote services, and trading analytics.

## Concepts introduced / updated

- [[open-source-agent-integration-analysis]] — documents the license boundary and the clean-room mapping.
- [[hermes-memory-operations]] — defines local append, retry, query, and recovery operations.
- [[local-knowledge-pipeline-v1]] — distinguishes canonical Markdown from disposable vector projections.

## Relevance to our system

Stable source IDs, content checksums, append-only receipts, deterministic paths, and post-commit projection improve the Obsidian memory layer. No Journalit trading fields, account analytics, risk logic, remote backend, or import code enter Trade_agent.

## Contradictions / updates

- The request grouped Journalit with open-source repositories; its official license contradicts that premise.
- Version metadata in the inspected public mirror is not fully aligned, so it cannot serve as a reproducible implementation contract.

## Open questions

- Whether later journal projections need compare-and-swap updates for human-owned note regions; v1 intentionally supports immutable events only.

## Primary evidence

- [Repository](https://github.com/Cursivez/journalit/tree/00f1f7044b0a6ab5b49bda682b582f87741c5d09)
- [Proprietary license](https://github.com/Cursivez/journalit/blob/00f1f7044b0a6ab5b49bda682b582f87741c5d09/LICENSE)
- [README license statement](https://github.com/Cursivez/journalit/blob/00f1f7044b0a6ab5b49bda682b582f87741c5d09/README.md#license)
- [Privacy and network disclosure](https://github.com/Cursivez/journalit/blob/00f1f7044b0a6ab5b49bda682b582f87741c5d09/PRIVACY.md)
- [Plugin lifecycle](https://github.com/Cursivez/journalit/blob/00f1f7044b0a6ab5b49bda682b582f87741c5d09/src/core/PluginInitializer.ts)
- [Service lifecycle](https://github.com/Cursivez/journalit/blob/00f1f7044b0a6ab5b49bda682b582f87741c5d09/src/services/ServiceManager.ts)
- [Typed event boundary](https://github.com/Cursivez/journalit/blob/00f1f7044b0a6ab5b49bda682b582f87741c5d09/src/services/events/EventBus.ts)
