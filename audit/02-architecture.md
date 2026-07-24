# Architecture

## Reverse-engineered architecture

The intended system is an event-driven modular platform. The canonical path is a
set of deterministic Python workers connected through versioned NATS subjects and
PostgreSQL inbox/outbox tables. Local Ollama produces candidate analysis; a
deterministic risk worker is authoritative; execution is paper-only; reconciliation
and leases govern readiness. The observed runtime was instead a legacy
microservice/full-stack deployment.

There is no dedicated authentication service. The canonical control plane is
read-only, but the running legacy orchestrator exposes mutation routes without an
OpenAPI security scheme.

## 1. High-level system architecture

```mermaid
flowchart LR
  EX["Binance public market feed"] --> MP["Canonical market producer"]
  MP --> N["NATS JetStream"]
  N --> W["Canonical worker pipeline"]
  W --> PG["PostgreSQL inbox/outbox + trading state"]
  W --> O["Ollama candidate model"]
  W --> PE["Deterministic paper execution"]
  PG --> CP["Read-only control plane"]
  CP --> UI["Next.js frontend"]
  N --> RT["Go realtime / WebSocket"]
  M["Prometheus + Grafana"] -.scrape.-> CP
```

## 2. Service dependency diagram

```mermaid
flowchart TD
  MIG["migrate"] --> PG["postgres"]
  MDP["market-producer"] --> N["nats"]
  MDW["market-data-worker"] --> N
  MDW --> PG
  FW["feature-worker"] --> N
  FW --> PG
  CW["candidate-worker"] --> N
  CW --> PG
  CW --> OL["ollama"]
  DW["decision-worker"] --> N
  DW --> PG
  EW["execution-worker"] --> N
  EW --> PG
  RW["reconciliation-worker"] --> PG
  CP["control-plane"] --> PG
```

## 3. Runtime request flow

```mermaid
sequenceDiagram
  participant Browser
  participant Nginx
  participant Frontend
  participant ControlPlane
  participant PostgreSQL
  Browser->>Nginx: HTTPS request
  Nginx->>Frontend: UI route
  Frontend->>ControlPlane: Read-only status API
  ControlPlane->>PostgreSQL: Query leases/readiness/state
  PostgreSQL-->>ControlPlane: Snapshot
  ControlPlane-->>Frontend: JSON status
  Frontend-->>Browser: Rendered state
```

## 4. Data flow diagram

```mermaid
flowchart LR
  RAW["market.raw.v1"] --> VAL["Validate + normalize"]
  VAL --> V["market.validated.v1"]
  VAL --> REJ["market.rejected.v1"]
  V --> F["features.ready.v1"]
  F --> C["signals.candidate.v1"]
  C --> R{"Deterministic risk"}
  R --> RA["risk.approved.v1"]
  R --> RR["risk.rejected.v1"]
  RA --> OI["orders.intent.v1"]
  OI --> OU["orders.updated.v1"]
  OU --> REC["Reconciliation"]
```

## 5. Docker network diagram

```mermaid
flowchart TB
  HOST["Host"]
  subgraph CAN["Canonical Compose network"]
    N["NATS"]
    PG["PostgreSQL"]
    R["Redis"]
    C["Chroma"]
    WK["Workers"]
    CP["Control plane"]
  end
  HOST -->|"loopback intended"| N
  HOST -->|"loopback intended"| PG
  HOST -->|"loopback intended"| R
  HOST -->|"loopback intended"| C
  HOST -->|"loopback intended"| CP
  WK --> N
  WK --> PG
```

Observed legacy containers published many ports on all interfaces rather than the
loopback-oriented canonical defaults.

## 6. Database interaction diagram

```mermaid
flowchart LR
  EV["NATS event"] --> IN["Inbox deduplication"]
  IN --> ST["Domain state transaction"]
  ST --> OUT["Outbox event"]
  OUT --> PUB["Publisher"]
  PUB --> N["NATS"]
  REC["Reconciliation"] --> ST
  LEASE["Worker leases"] --> READY["Readiness gate"]
  READY --> CP["Control plane"]
```

## 7. Background job flow

```mermaid
flowchart TD
  T["Worker start"] --> L["Acquire/renew lease"]
  L --> S["Subscribe durable consumer"]
  S --> H["Handle event"]
  H --> TX["Inbox + domain + outbox transaction"]
  TX --> ACK["Acknowledge"]
  TX --> DLQ["DLQ after terminal failure"]
  DLQ --> REPLAY["Operator-controlled replay"]
```

## 8. AI/agent workflow

```mermaid
flowchart LR
  FT["Deterministic features"] --> CA["Candidate worker"]
  CA --> OL["Local Ollama qwen3:8b"]
  OL --> DIG{"Model digest + schema valid?"}
  DIG -->|yes| C["Candidate signal"]
  DIG -->|no| FB["Explicit fallback / reject"]
  C --> RG["Deterministic risk gate"]
  RG -->|approved| PE["Paper execution"]
  RG -->|rejected| LOG["Auditable rejection"]
```

The model is advisory. Risk and execution must remain deterministic.

## 9. Deployment topology

```mermaid
flowchart TB
  subgraph SRC["Current source candidate"]
    C13["13 default Compose services"]
  end
  subgraph RUN["Observed local runtime"]
    L23["23 running legacy/full-stack containers"]
  end
  subgraph K8S["Kubernetes manifests"]
    OLD["Older orchestrator/realtime/frontend topology"]
  end
  SRC -.not deployed.-> RUN
  SRC -.not represented.-> K8S
```

## 10. Failure and recovery flow

```mermaid
stateDiagram-v2
  [*] --> Ready
  Ready --> NotReady: lease stale / NATS unavailable / reconciliation failure
  NotReady --> KillSwitch: reconciliation failure
  NotReady --> Recovering: dependency restored
  Recovering --> Ready: migrations + leases + clean reconciliation
  KillSwitch --> ManualReview: operator-owned activation
  ManualReview --> Ready: explicit safe clearance
```

The committed acceptance artifact verifies NATS-driven NotReady and recovery, but
it does not generate a risk decision, order intent, or fill.

## Architecture conclusions

- Style: event-driven modular platform with legacy microservices coexisting.
- Scheduler/cron: no verified production scheduler; workers are long-running loops.
- WebSocket: Go realtime and legacy paths exist; no live signal delivery observed.
- Logging: container stdout/json-file; no bounded rotation.
- Alerting: configuration exists but production delivery was not verified.
- Backup/recovery: no operational backup or restore mechanism verified.
- Primary architecture risk: source/runtime/deployment topology drift.

