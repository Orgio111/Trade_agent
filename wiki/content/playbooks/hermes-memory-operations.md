---
title: Hermes Memory Operations
type: playbook
tags: [hermes, obsidian, chromadb, ollama, operations, recovery]
created: 2026-07-16
updated: 2026-07-16
sources:
  - "[[hermes-local-integration-v1]]"
  - "[[open-source-agent-integration-analysis]]"
  - "[[local-knowledge-pipeline-operations]]"
status: stable
---

# Hermes Memory Operations

## Purpose

Operate the non-authoritative Hermes control plane defined by [[hermes-local-integration-v1]]. This playbook never authorizes trading, risk overrides, order placement, or broker access.

## Prerequisites

- A local Obsidian vault path supplied through `HERMES_VAULT_PATH`.
- Ollama listening on a loopback HTTP address with `nomic-embed-text` and any workflow models required by the submitted plan.
- ChromaDB 1.5.9 listening on the configured loopback port.
- The runtime collection name remains `trade-agent-runtime-memory-v1`.
- `project.manifest.toml` and `project.manifest.lock.json` pass the offline drift check.

Never point `HERMES_VAULT_PATH` at `wiki/raw`, a repository source tree, a symlink/junction, or a network share. Never put secrets, raw chain-of-thought, candles, token deltas, or high-rate telemetry into journal events.

## Start the local API

Run the Hermes service on `127.0.0.1` only. The application factory validates the vault and local dependency addresses before serving. Do not bind it to `0.0.0.0` unless a later ADR introduces authenticated transport and a trusted network policy.

```powershell
$env:HERMES_VAULT_PATH = "C:\absolute\path\to\obsidian-vault"
.venv\Scripts\python.exe -m packages.hermes
```

The module always binds to `127.0.0.1`; `HERMES_API_PORT` may select a different local port. `quantex-hermes` is also declared for environments that install the repository as a Python package.

Verify health:

```text
GET /api/v1/hermes/health
```

A healthy result requires the vault, exact local embedding model, and Chroma heartbeat. A missing model or unavailable index is a hard dependency failure, not a cloud-fallback trigger.

## Append a durable event

```text
POST /api/v1/hermes/events
```

Required semantic fields are event kind, agent ID, title, body, and occurrence time. Supply an explicit upstream event UUID when retrying across processes. The service:

1. validates and canonicalizes the event;
2. scans rendered Markdown for secret-like material;
3. resolves the only allowed event path under the vault;
4. creates the file without overwrite permission;
5. verifies idempotency or reports a checksum conflict;
6. embeds the committed event using `nomic-embed-text`;
7. upserts one provenance-bearing runtime record;
8. returns both journal and vector receipts.

If the response says the journal write succeeded but projection failed, correct the local Ollama/Chroma problem and resubmit the identical event. Do not alter its ID or content to hide the failure.

## Query runtime memory

```text
POST /api/v1/hermes/memory/query
```

Queries are bounded to 25 results and use caller-supplied filters only on scalar provenance metadata. Treat every result as a derived view: follow `source_path` back to canonical Markdown before using it as durable project evidence.

Project architecture and source-code questions still use the project-knowledge pipeline in [[local-knowledge-pipeline-operations]], not runtime memory.

## Run a bounded local workflow

```text
POST /api/v1/hermes/workflows
```

Every task declares an approved model role, dependencies, instructions, and timeout. The plan declares a global deadline and concurrency bound. Failures are terminal and explicit; dependent tasks become blocked. There is no retry to another provider or model.

Do not submit workflows that request order creation, risk bypass, position sizing, or broker calls. The endpoint has no such capabilities, and generated text is non-authoritative.

## Recovery matrix

| Symptom | Canonical state | Recovery |
|---|---|---|
| Unsafe path or secret rejection | Nothing written | Correct the request; never disable validation |
| Same ID, same checksum | Existing event valid | Treat as successful idempotent replay |
| Same ID, different checksum | Existing event protected | Investigate producer identity; issue a new legitimate event ID only for a genuinely distinct event |
| Ollama unavailable after append | Markdown valid, vector missing/stale | Restore local Ollama and resubmit identical event |
| Chroma unavailable after append | Markdown valid, vector missing/stale | Restore Chroma and resubmit identical event |
| Chroma namespace metadata drift | Markdown valid | Bump collection version through an ADR and rebuild; do not modify metadata in place |
| Workflow timeout | No memory mutation unless caller separately journals a final artifact | Reduce task scope or use the correct local model role |
| Vault moved | Existing Markdown valid at old path | Stop service, change the explicit vault setting, validate, then restart |

## Verification checklist

- [ ] Service is bound to loopback.
- [ ] Health reports the expected local model and collection.
- [ ] Event file exists under `trade-agent/memory/events/YYYY/MM/DD/`.
- [ ] Frontmatter event ID and checksum match the API receipt.
- [ ] Chroma hit metadata points back to the same relative Markdown path.
- [ ] Replaying an identical request returns the same canonical path.
- [ ] A changed payload under the same ID fails conflict testing.
- [ ] Project-manifest lock/check and the full test suite pass.
- [ ] `.obsidian/workspace.json` and `wiki/raw` remain untouched.

## Performance procedure

Benchmark Markdown append separately from Ollama embedding and Chroma upsert. Report warm p50/p95/max and event count. The journal/API path is background control-plane work; it must never be inserted into the trading decision latency budget. If OneDrive file latency dominates, batch only durable summary creation—not canonical integrity checks.

### 2026-07-16 baseline

The offline benchmark wrote and idempotently replayed 250 events through the real secret scan, path validation, machine-local event lease, atomic create, integrity read, and receipt path on Windows. It excluded Ollama and Chroma network time.

| Operation | p50 | p95 | max |
|---|---:|---:|---:|
| Create canonical event | 14.992 ms | 21.786 ms | 41.049 ms |
| Idempotent replay | 5.144 ms | 6.640 ms | 8.174 ms |

The create p95 passes the 25 ms control-plane budget. This is not an end-to-end model or vector latency claim and is not part of the trading decision loop.

## Related

- [[hermes-local-integration-v1]]
- [[open-source-agent-integration-analysis]]
- [[local-knowledge-pipeline-operations]]
