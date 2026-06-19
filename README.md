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

<<<<<<< Updated upstream
### 🧠 ML & RL Integration
- **TimesFM forecasting**: Google Research's pretrained time-series foundation model for zero-shot price forecasting (point + quantile bands), exposed as a signal source (`/api/v1/signal?source=timesfm`) and a dedicated endpoint. Loaded lazily; optional `timesfm[torch]` dependency.
- FreqAI-style ML signal generation (Random Forest classifier on 64+ features)
- Gymnasium-compatible RL trading environment (6 actions, 64-feature obs)
- Genetic strategy evolution (tournament selection, crossover, mutation)
- Optuna Bayesian hyperparameter optimization
=======
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
>>>>>>> Stashed changes

---

# Recommended Development Flow

## Phase 1
- Data ingestion
- Historical storage
- Backtesting engine
- Baseline strategies

## Phase 2
- Risk engine
- Paper trading
- Dashboard
- Monitoring

## Phase 3
- Live trading
- AI agent integration
- Portfolio allocation
- Strategy optimization

## Phase 4
- Multi-agent orchestration
- Automated research loops
- Adaptive strategy switching

---

# High-Level Architecture

Market Data
    ↓
Data Pipeline
    ↓
Feature Engineering
    ↓
Strategy Engine
    ↓
Risk Engine
    ↓
Execution Engine
    ↓
Exchange

AI Agents observe:
- market state
- strategy health
- portfolio risk
- performance metrics

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

```bash
git clone <repo>
cd trading-ai

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

<<<<<<< Updated upstream
The orchestrator starts on port 8001. Open `http://localhost:3001` for Grafana or access the frontend via nginx on port 80/443.

---

## API Overview

### REST Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | System health check |
| `GET /metrics` | Prometheus metrics |
| `GET /api/v1/portfolio` | Paper trading portfolio |
| `GET /api/v1/positions` | Open positions |
| `GET /api/v1/trades` | Trade history |
| `GET /api/v1/signal` | Latest trading signal |
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

## Tech Stack

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

## Project Structure
=======
Run services:
>>>>>>> Stashed changes

```bash
docker compose up
```

Start backend:

```bash
uvicorn api.main:app --reload
```

---

# Suggested Initial Strategy Types

- Trend following
- Mean reversion
- Breakout
- Volatility expansion
- Regime switching

---

# Long-Term Vision

Build a scalable autonomous trading research and execution platform capable of:
- multi-strategy orchestration,
- portfolio optimization,
- adaptive market regime analysis,
- AI-assisted research,
- low-latency execution,
- continuous monitoring.
