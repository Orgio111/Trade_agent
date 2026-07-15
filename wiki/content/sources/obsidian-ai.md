---
title: Obsidian AI Agentic Copilot
type: source
tags: [obsidian, adapters, sessions, streaming, local-first]
created: 2026-07-16
updated: 2026-07-16
sources: []
source_url: https://github.com/spencermarx/obsidian-ai
source_type: repository
authors: [Spencer Marx]
date: 2026-07-16
revision: 14d045b65db2b82ae4ebc946f9db750c3212182f
license: MIT
status: stable
---

# Obsidian AI Agentic Copilot

> **TL;DR** — The reusable architecture is its agent-adapter contract, isolated session lifecycle, normalized streaming events, message buffering, and bounded vault context. Its cloud CLI processes and direct vault-write authority are rejected.

## Key takeaways

- `AgentAdapter` normalizes agent detection, lifecycle, prompts, streaming messages, and session identifiers.
- `SessionManager` owns concurrent process state, cancellation, cleanup, and event delivery.
- `MessageQueue` separates incoming streaming deltas from rendering.
- `VaultContext` explicitly packages vault path, active file, selection, and cursor context.
- The plugin does not provide durable semantic memory, RAG, a supervisor graph, or multi-agent fan-in.
- Claude/OpenCode child processes, broad shell/file tools, raw context injection, and post-write review do not meet Trade_agent's local and pre-write control boundaries.

## Summary

This MIT-licensed Obsidian plugin is a desktop client for command-line agents. Its strongest reusable contribution is interface separation: UI, context collection, session lifecycle, agent transport, stream parsing, and message presentation are independent. Trade_agent maps those concepts to Python protocols, FastAPI contracts, bounded async workflow state, and direct loopback Ollama calls—without spawning cloud-oriented CLIs.

## Concepts introduced / updated

- [[open-source-agent-integration-analysis]] — compares the adapter and session patterns.
- [[hermes-local-integration-v1]] — adopts local-only ports and bounded workflow execution.
- [[hermes-memory-operations]] — constrains vault access to typed append/query operations.

## Relevance to our system

Hermes becomes a loopback client of typed endpoints rather than a process with raw vault access. Ollama adapters are exact-model and local-only. Obsidian receives validated journal events, while Chroma supplies bounded context with canonical Markdown provenance.

## Contradictions / updates

- The repository name is ambiguous and is not an official Obsidian first-party project; this page records the exact selected upstream.
- The upstream's cloud CLI adapters are reference implementations, not acceptable runtime dependencies for this fully local architecture.

## Open questions

- Whether an Obsidian UI client should later consume server-sent workflow events; it is not needed for the first backend slice.

## Primary evidence

- [Repository](https://github.com/spencermarx/obsidian-ai/tree/14d045b65db2b82ae4ebc946f9db750c3212182f)
- [MIT license](https://github.com/spencermarx/obsidian-ai/blob/14d045b65db2b82ae4ebc946f9db750c3212182f/LICENSE)
- [Adapter and context contracts](https://github.com/spencermarx/obsidian-ai/blob/14d045b65db2b82ae4ebc946f9db750c3212182f/src/adapters/types.ts)
- [Session manager](https://github.com/spencermarx/obsidian-ai/blob/14d045b65db2b82ae4ebc946f9db750c3212182f/src/session/session-manager.ts)
- [Message queue](https://github.com/spencermarx/obsidian-ai/blob/14d045b65db2b82ae4ebc946f9db750c3212182f/src/session/message-queue.ts)
- [Vault context collector](https://github.com/spencermarx/obsidian-ai/blob/14d045b65db2b82ae4ebc946f9db750c3212182f/src/utils/vault-context.ts)
