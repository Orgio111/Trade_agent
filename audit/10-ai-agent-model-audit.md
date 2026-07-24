# AI, Agent, and Model Audit

## Model/provider map

| Layer | Implementation | Observed state |
|---|---|---|
| Canonical candidate | Local Ollama `qwen3:8b`, schema/digest checks | Not deployed |
| vLLM provider | OpenAI-compatible local GPU service | Crash-looping |
| Cloud providers | NVIDIA/OpenRouter/Groq/OpenAI adapters | Secret names present; calls not validated |
| Legacy brain swarm | 12 registered brains | Zero active; NATS disconnected |
| VLM | Chart rendering + local vision adapter | Code exists; runtime unverified |
| RAG | Qdrant/Chroma retrieval | Qdrant empty; project Chroma populated |
| RL | PPO/evolution/training paths | Restore failure; zero runtime steps |

The canonical architecture correctly makes the deterministic risk engine
authoritative. AI output must never directly enable execution or override risk.

## [AUD-021] AI/agent behavior is fragmented and fallback provenance is not release-gated

- Severity: Medium
- Category: AI safety / model governance
- Component: Candidate, swarm, VLM, RAG, RL, provider router
- File: `workers/candidate/`, `orchestrator/`, `inference/`, `agents/`, `rl/`, `ml/`
- Line: Multiple
- Runtime service: Candidate/inference/orchestrator/RL
- Status: Confirmed
- Evidence: Canonical local-model validation exists, but observed legacy brains are inactive; vLLM is unavailable; RL restore fails; Qdrant trade memory is empty; code contains numerous fallback/mock/synthetic/random paths; no single model registry ties digest, dataset, metrics, latency, and approval to a release.
- Impact: The runtime may shift between model, rule fallback, empty context, or broken agent behavior without a comparable performance/risk record.
- Root cause: Research implementations and production candidate paths share the repository without an enforced model lifecycle contract.
- Reproduction: Map provider/agent entry points, inspect runtime statuses, and search fallback/synthetic/model-load paths.
- Recommended fix: Create a model registry with immutable digest, safe serialization, dataset/provenance, approved task, latency/cost budget, schema, calibration, rollback, and expiration; emit explicit `model_mode`/fallback reason in every candidate event.
- Validation after fix: Replay demonstrates identical deterministic risk outcomes for identical candidate inputs; fallback use is measurable and alertable; unapproved model digests cannot start.
- Estimated effort: L
- Priority: P1

## Agent controls

- Prompt/schema validation: present in the canonical candidate adapter, but not
  verified in runtime.
- Hallucination containment: deterministic risk is the intended containment layer.
- Memory: project knowledge is populated; runtime memory and Qdrant are empty.
- Cost/rate limits: inference budget configuration exists, but no production cost
  SLO or circuit breaker was demonstrated.
- Model warmup: legacy unauthenticated warmup route is an abuse/DoS surface.
- Model monitoring: no verified calibration drift, feature drift, confidence
  distribution, or fallback-rate dashboard.

## Production model decision

No AI model is approved for autonomous 24/7 decision influence on the observed
runtime. A model may participate only in an isolated paper/replay environment after
the deterministic safety gates and provenance registry are validated.

