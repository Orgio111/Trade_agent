---
title: Local Trading AI Architecture
type: concept
tags: [local-ai, ollama, rtx4050, provenance, deterministic-risk]
created: 2026-06-30
updated: 2026-07-16
sources:
  - "[[canonical-local-paper-runtime-v1]]"
  - "[[hermes-local-integration-v1]]"
status: stable
---

# Local Trading AI Architecture

## Definition

Trade_agent uses one host-local Ollama runtime for model inference while deterministic Python owns data validation, risk authorization, and paper execution. Model output is untrusted candidate input. It can never approve risk, create an order directly, disable the kill switch, or substitute for the execution ledger.

## Approved model registry

`packages/local_ai/models.py` is the single immutable role registry. Hermes re-exports it instead of maintaining a second mapping.

| Role | Exact Ollama model | Permitted use |
|---|---|---|
| `reasoning` | `qwen3:8b` | Candidate reasoning and bounded local workflows |
| `fast` | `phi3:3.8b` | Latency-sensitive candidate reasoning |
| `research` | `deepseek-r1:8b` | Offline/background research |
| `vision` | `moondream` | Chart/image interpretation outside deterministic authority |
| `tool_formatting` | `mistral` | Structured local output formatting |
| `embedding` | `nomic-embed-text` | Local Obsidian/Chroma projection |

An untagged registry name such as `mistral` or `moondream` may match Ollama's explicit `:latest` representation. No other alias, provider, or model is admitted. Cloud inference and OpenAI-compatible remote endpoints are outside the canonical runtime.

## Candidate provenance

Every `CandidateForRiskEvent` records:

- `provider=ollama`;
- a role restricted to `reasoning` or `fast`;
- the exact model selected by that role;
- a required 64-character model digest supplied by the future candidate producer;
- the exact market event, market snapshot, candidate content, trace, creation time, checksum, and deterministic event ID.

The deterministic risk verdict copies the same provenance and binds it to candidate hash, candidate event ID, market event ID, account, policy, and portfolio snapshot. The envelope currently validates digest shape and continuity; because the default candidate producer does not exist, capture-time comparison against Ollama's inventory digest remains to be implemented. Provenance makes a result auditable; it does not make the result authoritative.

## Local endpoint policy

Ollama URLs are accepted only when they use plain local HTTP and resolve to:

- `127.0.0.1` or another loopback address;
- `localhost`;
- `host.docker.internal` for Docker-to-host access.

Credentials, URL query parameters, fragments, and non-root paths are rejected. Runtime settings do not load `.env` files. This preserves an explicit local-only boundary and prevents a configuration change from silently routing trading intelligence to a cloud service.

## Readiness

The read-only control plane queries Ollama `/api/tags` and `/api/ps` and requires the exact registry to be installed. The check does not generate model output. Missing models make runtime readiness fail closed. See [[canonical-local-paper-runtime-v1]] and [[local-ai-deployment-guide]].

## Hardware policy

The RTX 4050 has a constrained 6 GB VRAM budget. Therefore:

- model loading and inference are not part of deterministic risk or execution latency;
- concurrent large-model inference is not assumed;
- Hermes serializes GPU inference by default;
- model warm-state and latency must be measured locally instead of asserted from design estimates;
- a missing or slow model must cause abstention/failure, never a risk bypass.

## Responsibility boundary

```text
Ollama model
  -> provenance-bound candidate
  -> deterministic risk engine + PostgreSQL authority
  -> exact approval
  -> deterministic PaperBroker execution
```

The `mistral` tool-formatting role does not own execution. Execution is ordinary deterministic code. `nomic-embed-text` creates retrieval embeddings only; retrieval content cannot mutate risk or broker state.

## Strengths and weaknesses

| Strength | Limitation |
|---|---|
| Fully local inference and embeddings | 6 GB VRAM constrains concurrency and warm-model choices |
| One exact role registry prevents routing drift | Capture-time digest verification still belongs in the missing candidate producer |
| Provenance survives candidate-to-verdict transport | The default runtime has no candidate producer yet |
| Deterministic risk/execution works independently of model internals | Model quality and latency are not yet promotion evidence |

## Contradictions / updates

Earlier versions of this page assigned execution to `mistral`, described unverified sub-200/300 ms inference, and treated sequential loading as an implemented production policy. [[canonical-local-paper-runtime-v1]] supersedes those claims: execution is model-free and paper-only, model health is read-only, and latency remains a benchmark question.

## Related

- [[canonical-local-paper-runtime-v1]]
- [[deterministic-paper-core-v1]]
- [[local-ai-deployment-guide]]
- [[hermes-local-integration-v1]]
- [[model-sequential-loading]]
- [[rtx4050-trading-system]]
