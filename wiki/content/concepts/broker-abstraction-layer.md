---
title: "Broker Abstraction Layer"
type: concept
tags: [broker, abstraction, binance, fix, paper, execution, interface]
created: 2026-06-30
updated: 2026-06-30
status: stable
---

# Broker Abstraction Layer

## Definition

A unified Python interface that wraps three distinct broker backends — Binance (real exchange), FIX Simulator (institutional simulation), and Paper Broker (backtesting/paper trading) — behind a single `BaseBroker` ABC. All order management, position tracking, and balance queries go through this layer, making the rest of the system broker-agnostic.

## Intuition

The trading system shouldn't care whether it's talking to Binance's REST API, a FIX 4.4 simulator, or a paper trading engine. The abstraction layer ensures the same `place_order()` call works identically across all three. Switch from paper to live by changing one config string.

## Architecture

```
┌──────────────────────────────┐
│        Trading System        │
└──────────┬───────────────────┘
           │ place_order() / get_positions() / ...
           ▼
┌──────────────────────────────┐
│       BaseBroker (ABC)       │  ← orchestrator/broker/base.py
└──────┬──────────┬──────────┬─┘
       │          │          │
  BinanceBroker  FIXSim   PaperBroker
  (real exchange) (instit.) (backtest)
```

## BaseBroker Interface

Every broker must implement:

```python
class BaseBroker(ABC):
    async def connect(self) -> bool
    async def disconnect(self) -> bool
    async def place_order(self, symbol, side, qty, order_type, price, stop_price) -> BrokerOrder
    async def cancel_order(self, order_id, symbol) -> bool
    async def get_order(self, order_id, symbol) -> BrokerOrder | None
    async def get_open_orders(self, symbol=None) -> list[BrokerOrder]
    async def get_positions(self) -> list[BrokerPosition]
    async def get_balance(self) -> dict[str, float]
    async def get_ticker(self, symbol) -> dict
```

## Broker Implementations

| Broker | File | Purpose | Key Features |
|--------|------|---------|-------------|
| `BinanceBroker` | binance_broker.py | Real Binance exchange | async REST + WebSocket, HMAC-SHA256, rate limiting, retry |
| `FIXSimulator` | fix_simulator.py | Institutional simulation | FIX 4.4 Tag=Value protocol, simulated fills, session sequence numbers |
| `PaperBroker` | paper_broker.py | Backtesting/paper trading | market/limit/stop fills, slippage, fees, PnL tracking, reset |

## Quick Usage

```python
from orchestrator.broker import get_broker

# Paper trading (no API keys needed)
broker = get_broker("paper", initial_balance=10000.0)
await broker.connect()
order = await broker.place_order("BTCUSDT", "buy", 0.01)

# Binance (testnet)
broker = get_broker("binance", api_key="...", api_secret="...", testnet=True)
await broker.connect()
balance = await broker.get_balance()

# FIX Simulator (institutional simulation)
broker = get_broker("fix")
await broker.connect()
order = await broker.place_order("BTCUSDT", "buy", 0.01)
```

## Factory Pattern

`orchestrator/broker/factory.py` provides `get_broker()` — a factory function that instantiates the correct broker class based on a string identifier. Convenience creators: `create_paper_broker()`, `create_binance_testnet()`.

## Adding a New Broker

1. Create `orchestrator/broker/coinbase_broker.py`
2. Implement all `BaseBroker` abstract methods
3. Register in `orchestrator/broker/factory.py`
4. Export in `orchestrator/broker/__init__.py`

## Files

| File | Purpose |
|------|---------|
| `orchestrator/broker/__init__.py` | Package init, all exports |
| `orchestrator/broker/base.py` | `BaseBroker` ABC + `BrokerOrder`, `BrokerPosition`, `BrokerConfig`, enums |
| `orchestrator/broker/binance_broker.py` | `BinanceBroker` — async REST + WebSocket |
| `orchestrator/broker/fix_simulator.py` | `FIXSimulator` — FIX 4.4 protocol |
| `orchestrator/broker/paper_broker.py` | `PaperBroker` — market/limit/stop fills |
| `orchestrator/broker/factory.py` | `get_broker()` factory |

## How we use it

- All brain signals ultimately flow through the broker layer for execution
- Paper broker used for backtesting in `orchestrator/brain_backtest.py`
- Binance broker for live/paper trading via `orchestrator/execution.py`
- Safety rule: always paper-trade first before live execution

## Strengths & weaknesses

**Strengths:**
- Broker-agnostic: switch execution backend with one config change
- Extensible: adding new brokers follows a clear 4-step pattern
- Async: all methods are `async` for non-blocking I/O
- Comprehensive: covers orders, positions, balances, tickers

**Weaknesses:**
- No order routing logic (direct market orders only)
- No smart order routing across exchanges
- FIX simulator is simplified (no real FIX session management)
- No order book reconstruction from fills

## Related

- [[signal-aggregation-logic]] — aggregated signals drive order decisions
- [[brain-ecosystem]] — brains produce signals, broker executes them
- [[local-trading-ai-architecture]] — overall system architecture

## Sources

- Internal code: `orchestrator/broker/` package
- Internal design: AGENTS.md §BROKER ABSTRACTION LAYER
