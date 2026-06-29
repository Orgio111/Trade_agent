"""
QUANTEX Broker Abstraction Layer — Unified interface for all exchange access.

Provides three broker implementations behind a common abstract interface:
  - BinanceBroker:     Real Binance REST + WebSocket trading
  - FIXSimulator:      Institutional FIX protocol simulation
  - PaperBroker:       Simulated paper trading for backtesting

Usage:
    from orchestrator.broker import get_broker

    # Auto-select based on config
    broker = get_broker(broker_type="binance", api_key="...", api_secret="...")
    broker.connect()
    order = broker.place_order("BTCUSDT", "buy", 0.01)
    broker.disconnect()
"""

from .base import BaseBroker, BrokerConfig, BrokerOrder, BrokerPosition, OrderStatus, OrderSide, OrderType
from .binance_broker import BinanceBroker
from .fix_simulator import FIXSimulator
from .paper_broker import PaperBroker
from .factory import get_broker

__all__ = [
    "BaseBroker",
    "BrokerConfig",
    "BrokerOrder",
    "BrokerPosition",
    "OrderStatus",
    "OrderSide",
    "OrderType",
    "BinanceBroker",
    "FIXSimulator",
    "PaperBroker",
    "get_broker",
]
