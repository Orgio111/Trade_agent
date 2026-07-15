---
title: Open-Source Agent Integration Analysis
type: comparison
tags: [architecture, multi-agent, obsidian, ollama, rag, hermes, integration]
created: 2026-07-16
updated: 2026-07-16
sources:
  - "[[tradingagents]]"
  - "[[journalit]]"
  - "[[obsidian-ai]]"
  - "[[obsidian-memory-for-ai]]"
status: stable
---

# Open-Source Agent Integration Analysis

| Reference | Legal reuse boundary | Adopt | Reject |
|---|---|---|---|
| [[tradingagents]] | Apache-2.0; code and patterns with attribution | typed state, bounded DAG, exhaustive routing, artifact/checkpoint identity | trading roles, LLM risk authority, cloud providers, sync hot-path loops |
| [[journalit]] | Proprietary; clean-room patterns only | stable record identity, canonical Markdown projection, receipts, disposable indexes | all source/schema/UI code, remote backend, analytics and trading logic |
| [[obsidian-ai]] | MIT | adapters, session lifecycle, normalized stream, bounded context | cloud CLI spawning, raw shell/vault access, post-write review |
| [[obsidian-memory-for-ai]] | No declared license; clean-room patterns only | append-only events, proposal/receipt concepts, provenance, generated views | all source/schema/template copying, advisory claims as locks, unrestricted paths |

The selected architecture extends Trade_agent with a non-authoritative engineering and memory control plane. It does not replace [[multi-agent-pipeline]], deterministic risk, execution, broker adapters, or local Ollama intelligence.

```text
Hermes / local Obsidian client
  -> loopback FastAPI contracts
  -> bounded async workflow -> exact approved Ollama model
  -> validated JournalEvent
  -> append-only Obsidian Markdown (canonical)
  -> nomic-embed-text
  -> trade-agent-runtime-memory-v1 (derived Chroma)
```

## 1. TradingAgents

### 1. System breakdown

**Architecture summary.** A synchronous LangGraph composition root registers analyst/tool loops, a bounded research debate, synthesis, proposal, risk-perspective discussion, and a final manager. It returns artifacts and reports; it is not a FastAPI service or broker execution engine.

**Agent types.** Market, sentiment, news, and fundamentals analysts; bull and bear researchers; a research manager; a trader role; aggressive, neutral, and conservative risk roles; and a portfolio-manager role.

**Data flow.** Input identity and date -> analyst/tool nodes -> bounded research debate -> synthesis -> proposal -> bounded risk-perspective discussion -> final artifact -> checkpoint/report/memory.

**Memory system.** Typed graph state, per-instrument graph-shaped checkpoints, bounded prior context, and append-only Markdown records with pending/resolved lifecycle. Obsidian frontmatter, RAG, FastAPI, and broker submission are absent.

### 2. Reusable components

- Declarative node registry and typed state envelopes.
- Complete conditional route maps with contract tests.
- Explicit task/round/deadline bounds.
- Graph revision and source provenance on artifacts.
- Structured outputs that fail closed at machine boundaries.
- Idempotent canonical record plus deferred outcome receipt.

### 3. Integration plan

- **Trade_agent fit:** extend `packages/hermes/workflow.py`; do not replace `orchestrator/langgraph_pipeline.py`, `packages/risk`, or `packages/execution`.
- **Hermes:** submits only typed workflow or journal requests over loopback FastAPI; it receives status and artifacts, not broker credentials.
- **Ollama:** roles resolve through an exact local allowlist. There is no provider fallback or cloud SDK.
- **Obsidian:** only validated final artifacts are journaled. Raw reasoning and intermediate token streams remain transient.
- **Latency:** multi-agent work is a background/control-plane path and never part of the `<300 ms` order decision loop.

### 4. Code-level extraction

```python
class WorkflowTask(BaseModel):
    task_id: str
    role: LocalModelRole
    instructions: str
    depends_on: tuple[str, ...]
    timeout_seconds: float

class BoundedWorkflowEngine:
    async def execute(self, plan: WorkflowPlan) -> WorkflowRun: ...
```

Suggested endpoint mapping: `POST /api/v1/hermes/workflows`, `GET /api/v1/hermes/health`. There is deliberately no signal, order, trade, risk-override, or execution endpoint.

```text
validate unique task IDs and acyclic dependencies
while pending tasks exist and deadline remains:
  mark dependents of failed tasks blocked
  run the next dependency-ready wave under a semaphore
  validate each local-model result and record terminal state
return succeeded / failed / timed_out with complete task evidence
```

The shared-state schema carries run ID, task ID, role, dependency IDs, status, exact model, latency, content, and redacted failure—not trading authority.

## 2. Journalit

### 1. System breakdown

**Architecture summary.** A TypeScript/React Obsidian plugin with staged lifecycle initialization, core and lazy services, commands, views, processors, typed events, canonical note services, and disposable indexes.

**Agent types.** None. It has no LLM planner, debate graph, agent memory, or multi-agent coordination.

**Data flow.** Form/command -> validation and mutation planning -> canonical Markdown commit -> receipt/read model -> typed notification -> cache/index/view refresh. Import behavior publicly exposes preview -> commit -> projection -> acknowledgement phases.

**Memory system.** Markdown/frontmatter records are durable; settings, UI state, and indexes have independent lifecycles. Human-owned bodies survive structured-field regeneration. It has no RAG.

### 2. Reusable components

Only clean-room behavior patterns are reusable:

- stable source identity and schema revision separate from the filename;
- deterministic path policy and collision validation;
- canonical Markdown plus preserved human-owned content;
- post-commit event/invalidation flow;
- idempotent projection receipt and reconcile operation;
- staged/lazy startup and disposable derived indexes.

### 3. Integration plan

- **Trade_agent fit:** extend `packages/hermes/obsidian.py` and `service.py`; execution ledger and trading logic are unchanged.
- **Hermes:** can append/query/retry engineering memory, never create orders or mutate canonical execution facts.
- **Ollama:** only `nomic-embed-text` projects already committed Markdown. Other models cannot invent canonical journal fields.
- **Obsidian:** v1 writes immutable event files. Future mutable notes require compare-and-swap and explicit user ownership regions.
- **License:** no Journalit implementation, schemas, templates, CSS, or UI are copied.

### 4. Code-level extraction

```python
class ObsidianJournalStore:
    async def append(self, event: JournalEvent) -> JournalReceipt: ...

class HermesMemoryService:
    async def ingest(self, event: JournalEvent) -> HermesIngestReceipt: ...
```

Endpoints: `POST /api/v1/hermes/events`, `POST /api/v1/hermes/memory/query`, and `GET /api/v1/hermes/health`.

```text
validate event, path, checksum, and secret policy
create canonical Markdown with create-only semantics
if same ID and checksum exists: return its receipt
if same ID and different checksum exists: fail conflict
embed committed document and idempotently upsert its runtime record
return both canonical and projection receipts
```

The clean-room frontmatter schema includes `type`, `schema_version`, `event_id`, `event_kind`, `agent_id`, `occurred_at`, `content_sha256`, tags, source references, and trace ID.

## 3. Obsidian AI

### 1. System breakdown

**Architecture summary.** Obsidian chat UI -> vault-context collector -> session manager -> agent adapter -> child CLI -> normalized streaming parser -> message queue -> renderer.

**Agent types.** Claude Code, OpenCode, and generic CLI adapters. These are transport implementations, not cooperating roles.

**Data flow.** Prompt plus selected vault context -> one-shot process -> JSONL/plain stream -> normalized messages -> buffered UI. Session IDs provide continuity.

**Memory system.** In-process messages and external CLI sessions only. The plugin has no durable semantic memory, Chroma, or audit ledger.

### 2. Reusable components

- Agent/model protocol separate from orchestration.
- Per-session lifecycle, bounded concurrency, cancellation, and cleanup.
- Normalized progress/result/error events.
- Explicit, bounded vault context instead of implicit whole-vault access.
- UI transport separated from runtime execution.

### 3. Integration plan

- **Trade_agent fit:** the adapter concept informs `packages/hermes/ollama.py`; session semantics inform the bounded workflow engine.
- **Hermes:** calls typed local HTTP operations, not a shell command with vault-root access.
- **Ollama:** direct `httpx` communication to a validated loopback URL replaces cloud-oriented child CLIs.
- **Obsidian:** filesystem authority is reduced to an append-only repository; active-file context can later be a read-only, size-bounded request field.
- **Security:** direct Bash, WebSearch, Write, Edit, raw chain-of-thought storage, and post-hoc Keep/Revert are rejected.

### 4. Code-level extraction

```python
class AgentRunner(Protocol):
    async def run(self, task: WorkflowTask, context: Mapping[str, str]) -> AgentOutput: ...

class OllamaChatGateway(AgentRunner):
    async def health(self) -> ComponentHealth: ...
    async def aclose(self) -> None: ...
```

The API and workflow schemas are the same local contracts described above. A later streaming endpoint may expose normalized `started`, `delta`, `tool`, `completed`, `failed`, and `cancelled` events, but streaming is not required for canonical journaling.

## 4. Obsidian Memory for AI

### 1. System breakdown

**Architecture summary.** Human narrative, one-file facts, append-only events, typed schemas, agent-scoped proposals, advisory claims, applied receipts, generated views, and lint/compact/query tools.

**Agent types.** No model roles. Filesystem-scoped agent inboxes and operation identities coordinate writers.

**Data flow.** Proposed operation -> schema/path/precondition validation -> controlled apply -> receipt -> regenerated views -> lint.

**Memory system.** Paths are primary keys; events are immutable; facts are temporal and supersedable; receipts and views are derived. There is no FastAPI, model, embedding service, or database.

### 2. Reusable components

Only clean-room concepts are used:

- canonical Markdown versus disposable read-model separation;
- append-only events, stable IDs, source provenance, and checksums;
- operation proposals instead of raw model writes;
- optimistic precondition hashes and applied receipts;
- controlled predicates, staleness/contradiction views, and pre/post validation.

### 3. Integration plan

- **Trade_agent fit:** v1 uses the append-only subset in `packages/hermes`; future mutable facts require a separate approved operation service.
- **Hermes:** receives append/query/status capabilities only. No raw `apply` or filesystem tool is exposed.
- **Ollama:** structured extraction may propose future operations, but deterministic validators decide whether a proposal is valid; models never apply it.
- **Obsidian:** resolved-root containment, symlink/junction defense, an operation-specific allowlist, create-only event paths, and secret scanning harden the reference concept.
- **Chroma:** indexes only successfully committed canonical events; `_views`, proposals, claims, receipts, raw chats, and transient files are excluded.

### 4. Code-level extraction

```python
class JournalEvent(BaseModel):
    event_id: UUID
    occurred_at: datetime
    kind: JournalEventKind
    agent_id: str
    title: str
    body: str
    content_sha256: str

class RuntimeMemoryIndex:
    async def index_event(self, event: JournalEvent, journal: JournalReceipt) -> RuntimeIndexReceipt: ...
    async def query(self, text: str, *, top_k: int, filters=None) -> RuntimeMemoryQueryResult: ...
```

The future mutable-fact pseudocode is intentionally not an endpoint in v1:

```text
model or agent proposes typed operation
validator resolves target under an allowed vault prefix
validator compares expected content checksum
human/policy gate approves application
repository atomically applies and emits immutable receipt
views and Chroma rebuild only from the committed canonical record
```

## Accepted integration slice

The implemented slice is deliberately narrow:

1. strict journal and workflow contracts;
2. loopback-only Ollama and FastAPI boundaries;
3. bounded generic async DAG execution;
4. append-only, secret-scanned Obsidian events;
5. `nomic-embed-text` projection into `trade-agent-runtime-memory-v1`;
6. manifest-pinned upstream provenance, ownership, docs, tests, and embedding inputs;
7. no mutation to risk, execution, strategies, legacy RAG, or `.obsidian` state.

## Rejected integration surface

- Cloud inference, provider fallbacks, remote research tools, or credentialed URLs.
- Model-controlled risk, portfolio sizing, order execution, or broker APIs.
- Upstream code from proprietary or unlicensed repositories.
- Whole-vault context injection, raw chain-of-thought retention, or direct filesystem tools.
- Debate or embedding work on the trading hot path.
- Mixing project knowledge and runtime journal records in one Chroma collection.

## Sources

- [[tradingagents]]
- [[journalit]]
- [[obsidian-ai]]
- [[obsidian-memory-for-ai]]
- [[hermes-local-integration-v1]]
- [[hermes-memory-operations]]
