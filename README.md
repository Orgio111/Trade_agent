# QUANTEX — Autonomous AI Trading System

**Multi-language, multi-agent, multi-provider AI trading ecosystem.**  
Built with Rust (execution), Python (AI/ML + inference routing), Go (realtime), and TypeScript (frontend).  
Cloud-first inference with adaptive provider routing (Groq → NVIDIA NIM → OpenRouter).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      QUANTEX TRADING SYSTEM                          │
├─────────────────┬──────────────────┬───────────────────┬────────────┤
│   INFERENCE      │   PYTHON AI/ML   │   RUST EXECUTION   │   GO WS    │
│   ─────────      │   ───────────    │   ─────────────    │  ─────    │
│   • Groq LPU     │   • 7 AI Agents  │   • Order engine   │  Events   │
│   • NVIDIA NIM   │   • Swarm debate │   • Risk core      │  Pub/sub  │
│   • OpenRouter   │   • RL training  │   • CCXT exchange  │  WS bcast │
│   • Semantic     │   • Backtesting  │   • WS engine      │           │
│     cache        │   • Feature eng  │                    │           │
│   • Adaptive     │   • Market str.  │                    │           │
│     routing      │   • Microstruct. │                    │           │
│   • Cost tracker │   • Vector mem   │                    │           │
│   • Circuit      │   • Enhanced     │                    │           │
│     breaker      │     risk v2      │                    │           │
├─────────────────┴──────────────────┴───────────────────┴────────────┤
│                    TYPESCRIPT FRONTEND (Next.js 15)                   │
│                    ────────────────────────────                       │
│                    • Live dashboard with 5 tabs                       │
│                    • Price charts (lightweight-charts)                │
│                    • Agent swarm visor                                │
│                    • Microstructure & market structure panels         │
│                    • Inference routing panel (provider health,        │
│                      adaptive chains, cost usage)                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Features

### 🤖 Multi-Agent Swarm Intelligence
- **7 agents** in parallel debate (Market Analyst, DeepSeek Analyst, Risk Guardian, Sentiment, Regime, Scalping, Swing)
- **4-round debate cycle**: Independent analysis → Cross-examination → Weighted voting → Risk veto
- Credibility-weighted voting with agent performance tracking
- Streaming deep reasoning via DeepSeek R1

### 🔄 Adaptive Inference Routing
- **3 providers**: Groq (ultra-fast LPU, 800+ tok/s), NVIDIA NIM (high-quality reasoning), OpenRouter (200+ model fallback)
- **Per-agent provider chains**: Each agent has an optimal provider order (e.g., scalping uses Groq first, deep reasoning uses NVIDIA NIM first)
- **EMA-based latency tracking**: Automatically reorders provider chains based on observed P50/P99 latency
- **Circuit breaker**: Skips failing providers after N consecutive failures
- **Semantic caching**: 40-60% cost reduction via Qdrant vector similarity cache
- **Budget tiers**: Free ($0), Low ($0.50/day), Medium ($2.00/day)

### 🏦 Institutional-Grade Trading Components
- **Market Structure Engine**: Liquidity sweeps, order blocks, FVG, BOS/CHOCH, Wyckoff phases
- **Microstructure Engine**: Spoofing detection, hidden liquidity, CVD/divergence, liquidation cascade prediction, order book imbalance
- **Enhanced Risk Engine v2**: Kelly sizing, kill switches, anti-martingale, volatility expansion limits, correlation management, time-based decay
- **Execution Engine**: TWAP/VWAP/Iceberg slicing, smart order routing, slippage estimation
- **Position Manager**: Multi-level TP, trailing stops, DCA entry ladders, break-even triggers

### 📊 Comprehensive Backtesting
- Binance REST API and CSV data sources
- Realistic fill simulation with configurable slippage
- Sharpe, Sortino, Calmar, profit factor, expectancy
- Walk-forward validation

### 🧠 ML & RL Integration
- FreqAI-style ML signal generation (Random Forest classifier on 64+ features)
- Gymnasium-compatible RL trading environment (6 actions, 64-feature obs)
- Genetic strategy evolution (tournament selection, crossover, mutation)
- Optuna Bayesian hyperparameter optimization

---

## Quick Start

```bash
# Prerequisites
rustc 1.93+, Python 3.12+, Go 1.26+, Node.js 18+

# Clone and enter
git clone <repo> && cd Trade_agent

# Copy environment variables
cp .env.example .env
# Edit .env with your API keys (NVIDIA_API_KEY, OPENROUTER_API_KEY, GROQ_API_KEY)
# Set INFERENCE_BUDGET_TIER=free (default) | low | medium

# Start infrastructure
docker compose up -d

# Install dependencies
cd execution && cargo build
cd ../orchestrator && pip install -r requirements.txt
cd ../frontend && npm install

# Run
cd .. && python -m orchestrator.main
```

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

```
Trade_agent/
├── inference/                 # Cloud inference layer
│   ├── router.py              #   Multi-provider intelligent router
│   ├── cache.py               #   Semantic caching (Qdrant)
│   ├── cost_tracker.py        #   Token usage & cost monitoring
│   └── providers/             #   Provider implementations
│       ├── base.py            #     Abstract provider interface
│       ├── groq.py            #     Groq LPU inference
│       ├── nim.py             #     NVIDIA NIM API client
│       └── openrouter.py      #     OpenRouter aggregator
│
├── orchestrator/              # Python AI/ML orchestrator
│   ├── main.py                #   FastAPI app with 50+ endpoints
│   ├── agents.py              #   7 AI agent implementations
│   ├── agent_routing.py       #   Agent→model routing + adaptive chains
│   ├── inference_integration.py # Bridge between agents and InferenceRouter
│   ├── strategy.py            #   Rule-based strategy engine
│   ├── ml_signals.py          #   FreqAI-style ML signal generator
│   ├── backtest.py            #   Backtesting engine
│   ├── paper_account.py       #   Paper trading simulator
│   ├── position_manager.py    #   SL/TP management
│   ├── feature_engine.py      #   64+ technical indicators
│   ├── market_structure.py    #   SMC/Wyckoff analysis
│   ├── microstructure.py      #   Order flow analysis
│   ├── execution.py           #   TWAP/VWAP/Iceberg execution
│   ├── metrics.py             #   Prometheus metrics
│   ├── swarm/debate_system.py #   Multi-agent debate system
│   ├── memory/vector_memory.py#   Qdrant vector memory
│   ├── risk/risk_engine.py    #   10-gate risk management
│   ├── risk/risk_engine_v2.py #   Enhanced institutional risk
│   ├── rl/trading_env.py      #   Gymnasium trading environment
│   └── rl/strategy_evolver.py #   Genetic strategy evolution
│
├── execution/                 # Rust execution engine
├── realtime/                  # Go realtime services
├── frontend/                  # Next.js dashboard
├── deployment/                # K8s, Terraform, Nginx configs
├── monitoring/                # Prometheus + Grafana
├── db/                        # SQL migrations
└── docker-compose.yml         # 12+ service stack
```

---

## API Keys

Set these in `.env`:

| Variable | Required | Source |
|----------|----------|--------|
| `NVIDIA_API_KEY` | Recommended | [NVIDIA NGC](https://ngc.nvidia.com/) |
| `OPENROUTER_API_KEY` | Recommended | [OpenRouter](https://openrouter.ai/) |
| `GROQ_API_KEY` | Recommended | [Groq Console](https://console.groq.com/) |
| `INFERENCE_BUDGET_TIER` | No (default: `free`) | `free` / `low` / `medium` |

At least one API key is needed for inference. The router will work with any subset of configured providers.

---

## Inference Budget Tiers

| Tier | Daily Budget | Behavior |
|------|-------------|----------|
| `free` | $0.00 | Uses only free-tier models (Groq free, OpenRouter free) |
| `low` | $0.50 | Mixes free and low-cost paid models |
| `medium` | $2.00 | Full access to all models including premium |

---

## License

MIT
