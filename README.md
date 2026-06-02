# QUANTEX — Autonomous AI Trading System

**Multi-language, multi-agent AI trading ecosystem.**
Built with Rust (execution), Python (AI/ML), Go (realtime), TypeScript (frontend).

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    QUANTEX TRADING SYSTEM                    │
├──────────────────┬──────────────────┬───────────────────────┤
│   RUST CORE      │   PYTHON AI/ML   │   GO REALTIME        │
│   ─────────      │   ───────────    │   ────────────       │
│   • Execution    │   • AI Agents    │   • Event bus        │
│   • Order mgr    │   • RL training  │   • Pub/sub          │
│   • Risk core    │   • Strategy     │   • WS broadcast     │
│   • WS engine    │   • NIM client   │   • Microservices    │
│   • Exchange     │   • Backtesting  │                      │
│                  │   • Vector mem   │                      │
├──────────────────┴──────────────────┴───────────────────────┤
│                    TYPESCRIPT FRONTEND                      │
│                    ──────────────────                       │
│                    • Next.js dashboard                      │
│                    • Live charts & graphs                   │
│                    • AI cockpit                             │
│                    • API gateway                            │
└─────────────────────────────────────────────────────────────┘
```

## Phase 1 — MVP (Weeks 1-4)

### Week 1: Foundation ✅
- [x] Project structure (Rust + Python + Go + TS)
- [x] Docker Compose (PostgreSQL, Redis, NATS)
- [x] PostgreSQL schema (trades, positions, agent_memory)
- [x] CCXT exchange connectivity (Binance testnet)
- [x] Rust execution engine (order types, risk core)
- [x] Python strategy engine (EMA crossover + RSI)
- [x] Basic risk management (SL/TP, max drawdown)

### Week 2: AI Agents
- [ ] OpenRouter DeepSeek R1 analysis agent
- [ ] Backtesting with historical data
- [ ] Position management (full SL/TP tracking)
- [ ] Paper trading mode

### Week 3: Intelligence
- [ ] FreqAI-style ML signal generation
- [ ] Trade memory storage
- [ ] Prometheus + Grafana monitoring

### Week 4: Go Live
- [ ] VPS deployment
- [ ] Telegram alerts
- [ ] Live trading with $10-25

## Quick Start

```bash
# Prerequisites
rustc 1.93+, Python 3.12+, Go 1.26+, Node.js 18+

# Clone and enter
git clone <repo> && cd Trade_agent

# Start infrastructure
docker compose up -d

# Install dependencies
cd execution && cargo build
cd ../orchestrator && pip install -r requirements.txt
cd ../frontend && npm install

# Run
cd .. && python main.py
```

## Tech Stack

| Layer | Language | Framework |
|-------|----------|-----------|
| Execution Engine | Rust | Tokio, async-std |
| AI Agents | Python | LangGraph, FastAPI |
| Realtime Services | Go | NATS, gorilla/websocket |
| Frontend | TypeScript | Next.js 15, Three.js |
| Database | SQL | PostgreSQL 16 + pgvector |
| Cache | — | Redis 7.2 |
| Streaming | — | NATS JetStream |
| Monitoring | — | Prometheus + Grafana |
