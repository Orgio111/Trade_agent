# Trading AI System

A modular AI-assisted algorithmic trading system focused on:
- market data ingestion,
- strategy research,
- backtesting,
- risk management,
- paper/live execution,
- AI research agents,
- monitoring and observability.

---

# Core Goals

- Fast research iteration
- Reliable execution
- Strict risk control
- Open-source-first architecture
- Modular agent-based design
- Production-ready engineering practices

---

# Main Stack

## Trading / Exchange
- CCXT
- Exchange native SDKs (optional)

## Backtesting
- vectorbt
- backtrader

## Data
- DuckDB
- Parquet
- PostgreSQL

## AI / ML
- PyTorch
- LightGBM
- XGBoost
- Transformers
- NVIDIA NIM APIs

## API / Backend
- FastAPI
- Redis
- WebSockets

## Monitoring
- Prometheus
- Grafana
- Loki

## UI
- Next.js
- TailwindCSS
- shadcn/ui

---

# Safety Rules

- Never let AI bypass risk limits.
- Never trade without monitoring.
- Always test on paper trading first.
- Keep execution deterministic.
- Log every decision.
- Assume APIs can fail.

---

# Quick Start

## Docker Compose (Local Dev)

```bash
git clone <repo>
cd Trade_agent

# Copy environment file and fill in API keys
cp .env.example .env

# Start all services (NATS + PostgreSQL + Redis + Qdrant + Orchestrator + Frontend + Monitoring)
docker compose up -d

# With hot-reload for development:
docker compose -f docker-compose.yml -f docker-compose.override.yml up -d
```

Access:
- **Frontend**: http://localhost:3000
- **Orchestrator API**: http://localhost:8001
- **Grafana**: http://localhost:3001 (admin / quantex123)
- **NATS monitoring**: http://localhost:8222

---

# Deployment

## 1. Docker Compose (Single Server)

Best for MVP / single-server deployment. All services run on one host via Docker Compose.

```bash
# Full production stack (no hot-reload)
docker compose -f docker-compose.yml up -d

# Stop everything
docker compose down
```

**Trinity Architecture Data Flow (Docker Compose):**

```
┌─────────────────────────────────────────────────────────────────┐
│ Layer A: Python Orchestrator (11 AI Brains)                     │
│  timesfm  freqai  llm_regime  microstructure  orderflow_nautilus│
│  finbert  finrl   statarb     onchain         custom_nn         │
│  polymarket_alpha                                               │
│         │                                                       │
│         ▼ publish → signals.raw (NATS JetStream)                │
├─────────────────────────────────────────────────────────────────┤
│ Layer B: Go Realtime (Weighted Aggregation)                     │
│  Subscribe ← signals.raw                                        │
│  11 brain weights × confidence → final score                    │
│  Action: BUY / SELL / HOLD                                      │
│  Publish → signals.aggregated (NATS JetStream)                  │
│         │                                                       │
│         ▼ WebSocket ──► Frontend Dashboard                      │
├─────────────────────────────────────────────────────────────────┤
│ Layer C: Rust Execution Engine (Paper / Live)                   │
│  Subscribe ← signals.aggregated                                 │
│  Pre-trade risk → Order placement → Execution confirmation      │
└─────────────────────────────────────────────────────────────────┘
```

### NATS JetStream Subjects

| Subject | Publisher | Consumer | Payload |
|---------|-----------|----------|---------|
| `signals.raw` | Python brains (Layer A) | Go orchestrator (Layer B) | `{brain_id, score, confidence, ...}` |
| `signals.aggregated` | Go orchestrator | Rust execution (Layer C) | `{action, final_score, active_brains, brain_scores, weights_used}` |
| `signals.executed` | Rust execution | Monitoring | `{execution_id, filled_price, ...}` |

### Brain Weights

Each brain has a weight in the final aggregated score (total = 1.00):

| Brain | Weight | Cadence | Role |
|-------|:------:|:-------:|------|
| TimesFM | 0.25 | 60s | Primary forecaster (Google foundation model) |
| FreqAI | 0.15 | 15s | Technical indicator ML (XGBoost) |
| LLM Regime | 0.15 | 15s | LLM regime classification |
| Microstructure | 0.08 | 10s | Tick-level order flow |
| OrderFlow Nautilus | 0.08 | 5s | L2/L3 orderbook depth |
| FinBERT | 0.07 | 30s | NLP news sentiment |
| FinRL | 0.07 | 30s | RL position sizing |
| StatArb | 0.05 | 15s | Z-score mean reversion |
| Custom NN | 0.04 | 30s | LSTM/Transformer temporal patterns |
| OnChain | 0.03 | 60s | Exchange flows + whale transactions |
| Polymarket Alpha | 0.03 | 60s | Prediction market alpha |

---

## 2. Kubernetes (Production Cluster)

Best for multi-node, scalable production deployment with self-healing.

### Architecture (25 K8s Resources)

```
┌──────────────────────────────────────────────────────────────────┐
│                    quantex namespace                              │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────┐  │
│  │ orchestrator │    │   realtime   │    │    frontend       │  │
│  │ (replicas: 2)│───▶│ (replicas: 2)│───▶│ (replicas: 2)    │  │
│  │ 11 brains    │    │ aggregation  │    │ dashboard         │  │
│  │ NATS_URL     │    │ WebSocket    │    │ Next.js           │  │
│  └──────┬───────┘    └──────┬───────┘    └───────────────────┘  │
│         │                   │                                    │
│         ▼                   ▼                                    │
│  ┌──────────┐       ┌──────────────┐                             │
│  │ NATS     │       │ Prometheus   │                             │
│  │ JetStream│       │ + Grafana    │                             │
│  └──────────┘       └──────────────┘                             │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────────┐  │
│  │PostgreSQL│  │  Redis   │  │  Qdrant  │  │  NATS Exporter │  │
│  └──────────┘  └──────────┘  └──────────┘  └────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### Deploy

```bash
# Prerequisites: Kubernetes 1.28+ cluster, kubectl configured

# 1. Create namespace
kubectl apply -f deployment/k8s/quantex-namespace.yaml

# 2. Create secrets (replace with actual API keys)
kubectl -n quantex create secret generic quantex-secrets \
  --from-literal=nvidia-api-key="nvapi-..." \
  --from-literal=openrouter-api-key="sk-or-v1-..." \
  --from-literal=groq-api-key="gsk_..." \
  --from-literal=postgres-password="secret"

# 3. Deploy all services (19 resources)
kubectl apply -f deployment/k8s/services.yaml

# 4. Verify
kubectl -n quantex get pods
kubectl -n quantex get svc
```

### Brain Weights ConfigMap

The `quantex-brain-weights` ConfigMap stores the 11 brain weights that the Go orchestrator reads via `BRAIN_WEIGHT_<ID>` env vars:

```bash
# View current weights
kubectl -n quantex get configmap quantex-brain-weights -o yaml

# Override a weight (e.g., boost Polymarket)
kubectl -n quantex edit configmap quantex-brain-weights
# Change: brain_weight_polymarket_alpha: "0.03" → "0.10"

# Restart realtime to pick up new weights
kubectl -n quantex rollout restart deployment quantex-realtime
```

### Scaling

```bash
kubectl -n quantex scale deployment quantex-orchestrator --replicas=3
kubectl -n quantex scale deployment quantex-realtime --replicas=3
kubectl -n quantex scale deployment quantex-frontend --replicas=3
```

### Monitoring

```bash
kubectl -n quantex port-forward svc/quantex-grafana 3001:3000
# Open http://localhost:3001 (admin / quantex123)

kubectl -n quantex port-forward svc/quantex-nats 8222:8222
# Open http://localhost:8222/healthz
```

---

## 3. Terraform (Hetzner Cloud)

For bare-metal provisioning on Hetzner Cloud with automatic Docker setup:

```bash
cd deployment/terraform
export TF_VAR_hcloud_token="your-token"
export TF_VAR_deployment_tier="tier0"  # $4-5/month - cx21 (2 vCPU, 4GB RAM)
# export TF_VAR_deployment_tier="tier1"  # $20-40/month - cx41 (4 vCPU, 16GB RAM)

export TF_VAR_ssh_key_name="quantex-deploy"
terraform init
terraform apply
```

The firewall automatically allows:
- `:80/:443` — Web traffic
- `:3000` — Frontend
- `:8001` — Orchestrator API
- `:4222` — NATS (internal subnet only)
- `:9090/:3001` — Prometheus/Grafana

---

# API Overview

### REST Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | System health check |
| `GET /metrics` | Prometheus metrics |
| `GET /api/v1/portfolio` | Paper trading portfolio |
| `GET /api/v1/positions` | Open positions |
| `GET /api/v1/trades` | Trade history |
| `GET /api/v1/signal` | Latest trading signal |
| `GET /api/v1/brains` | Brain runner status (11 brains) |
| `POST /api/v1/trade` | Execute trade |
| `POST /api/v1/backtest/run` | Run backtest |
| `POST /api/v1/ml/train` | Train ML model |
| `POST /api/v1/ml/predict` | Generate ML prediction |
| `POST /api/v1/swarm/debate` | Run full swarm debate cycle |
| `POST /api/v1/agent/deepseek-analyze` | DeepSeek analysis |
| `POST /api/v1/agent/risk-check` | Risk evaluation |
| `POST /api/v1/rl/train` | RL training step |
| `POST /api/v1/rl/evolve` | Genetic strategy evolution |
| `GET /api/v2/agents/routing` | Agent→model routing config |
| `GET /api/v2/inference/routing` | Full inference routing overview |
| `GET /api/v2/inference/adaptive-routing` | Per-agent latency tracking |
| `GET /api/v2/market-structure` | SMC/Wyckoff analysis |
| `POST /api/v2/microstructure/spoofing` | Spoofing detection |
| `POST /api/v2/microstructure/delta` | CVD divergence analysis |
| `POST /api/v2/microstructure/liquidation-cascade` | Cascade risk prediction |
| `POST /api/v2/risk/check` | Enhanced risk v2 check |
| `POST /api/v2/risk/kelly` | Kelly position sizing |
| `POST /api/v2/execution/plan` | Execution plan with slicing |
| `POST /api/v2/forecast/timesfm` | Price forecast via Google TimesFM |
| `GET /api/v2/forecast/timesfm/status` | TimesFM model load status |

### WebSocket (`/ws`)

Real-time push events every 5s:
- **`type: "portfolio"`** — Balance, equity, PnL, positions, drawdown
- **`type: "signal"`** — Latest ML signal with confidence
- **`type: "inference_routing"`** — Provider health, adapted chains, cost usage

---

## Agent → Provider Routing

Each agent has an optimal provider chain, automatically adapted based on real latency observations:

| Agent | Primary | Fallback | Task Type |
|-------|---------|----------|-----------|
| Supervisor | NVIDIA NIM | Groq → OpenRouter | reasoning |
| Market Analyst | NVIDIA NIM | Groq → OpenRouter | analysis |
| DeepSeek Analyst | NVIDIA NIM | Groq → OpenRouter | reasoning |
| Swing Agent | NVIDIA NIM | Groq → OpenRouter | analysis |
| Scalping Agent | **Groq** | OpenRouter → NIM | urgent |
| Execution Agent | **Groq** | OpenRouter → NIM | urgent |
| Anomaly Agent | **Groq** | OpenRouter → NIM | fast |
| Risk Guardian | **Groq** | OpenRouter → NIM | reasoning |
| Sentiment Agent | OpenRouter | Groq → NIM | classification |
| Regime Agent | OpenRouter | Groq → NIM | classification |
| Memory | OpenRouter | Groq → NIM | classification |

---

# Tech Stack

| Layer | Language | Framework/Tools |
|-------|----------|----------------|
| Inference Router | Python | OpenAI SDK, aiohttp, Qdrant |
| AI Agents | Python | FastAPI, LangGraph patterns |
| RL/Machine Learning | Python | scikit-learn, Stable-Baselines3, Optuna |
| Execution Engine | Rust | Tokio, CCXT, NATS |
| Realtime Services | Go | NATS, gorilla/websocket |
| Frontend | TypeScript | Next.js 15, lightweight-charts, Three.js |
| Database | SQL | PostgreSQL 16 + pgvector |
| Vector Store | — | Qdrant |
| Cache | — | Redis 7.2 |
| Streaming | — | NATS JetStream + InfluxDB 2.7 |
| Monitoring | — | Prometheus + Grafana |
| Reverse Proxy | — | Nginx + Let's Encrypt SSL |

---

