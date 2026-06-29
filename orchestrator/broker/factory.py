"""
QUANTEX Broker Factory — Create broker instances from config.

Provides a unified factory function that returns the correct broker
implementation based on a type string or configuration dict.

Usage:
    from orchestrator.broker import get_broker

    # Auto-select by type
    broker = get_broker("binance", api_key="...", api_secret="...")
    broker = get_broker("fix")
    broker = get_broker("paper", initial_balance=50000)

    # Or from config dict
    broker = get_broker_from_config({
        "type": "binance",
        "testnet": True,
        "api_key": "...",
        "api_secret": "...",
    })
"""

from __future__ import annotations

import logging
from typing import Optional

from .base import BaseBroker, BrokerConfig
from .binance_broker import BinanceBroker
from .fix_simulator import FIXSimulator
from .paper_broker import PaperBroker

logger = logging.getLogger("quantex.broker.factory")


def get_broker(
    broker_type: str = "paper",
    api_key: str = "",
    api_secret: str = "",
    testnet: bool = True,
    initial_balance: float = 10000.0,
    **kwargs,
) -> BaseBroker:
    """Create a broker instance by type.

    Args:
        broker_type: One of "binance", "fix", "paper" (default: "paper")
        api_key: Binance API key (binance only)
        api_secret: Binance API secret (binance only)
        testnet: Use testnet (binance only, default: True)
        initial_balance: Starting balance (paper only)
        **kwargs: Additional broker-specific config passed to extra dict

    Returns:
        BaseBroker instance.

    Raises:
        ValueError: If broker_type is unknown
    """
    broker_type = broker_type.lower().strip()

    if broker_type == "binance":
        config = BrokerConfig(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
            extra=kwargs,
        )
        return BinanceBroker(config)

    elif broker_type == "fix":
        config = BrokerConfig(
            testnet=True,
            extra={
                "fix_latency_min_ms": kwargs.get("fix_latency_min_ms", 50),
                "fix_latency_max_ms": kwargs.get("fix_latency_max_ms", 500),
                "fix_fill_probability": kwargs.get("fix_fill_probability", 0.7),
                "slippage_bps": kwargs.get("slippage_bps", 0.5),
            },
        )
        return FIXSimulator(config)

    elif broker_type == "paper":
        config = BrokerConfig(
            testnet=True,
            extra={
                "initial_balance": initial_balance,
                "maker_fee": kwargs.get("maker_fee", 0.0002),
                "taker_fee": kwargs.get("taker_fee", 0.0004),
                "slippage_bps": kwargs.get("slippage_bps", 1.0),
                "fill_probability": kwargs.get("fill_probability", 0.5),
            },
        )
        return PaperBroker(config)

    else:
        raise ValueError(
            f"Unknown broker type: '{broker_type}'. "
            f"Choose from: 'binance', 'fix', 'paper'"
        )


def get_broker_from_config(config: dict) -> BaseBroker:
    """Create a broker instance from a configuration dict.

    Expected format:
        {
            "type": "binance",          # Required: "binance", "fix", "paper"
            "api_key": "...",           # Binance API key
            "api_secret": "...",        # Binance API secret
            "testnet": true,            # Use testnet (binance only)
            "initial_balance": 10000,   # Starting balance (paper only)
            ...                         # Additional broker-specific settings
        }

    Args:
        config: Broker configuration dictionary.

    Returns:
        BaseBroker instance.
    """
    broker_type = config.get("type", "paper")
    return get_broker(
        broker_type=broker_type,
        api_key=config.get("api_key", ""),
        api_secret=config.get("api_secret", ""),
        testnet=config.get("testnet", True),
        initial_balance=config.get("initial_balance", 10000.0),
        **{k: v for k, v in config.items() if k not in ("type", "api_key", "api_secret", "testnet", "initial_balance")},
    )


# ── Convenience ─────────────────────────────────────────────

def create_paper_broker(initial_balance: float = 10000.0) -> PaperBroker:
    """Quick-create a paper broker (most common use case)."""
    return get_broker("paper", initial_balance=initial_balance)  # type: ignore


def create_binance_testnet(api_key: str, api_secret: str) -> BinanceBroker:
    """Quick-create a Binance testnet broker."""
    return get_broker("binance", api_key=api_key, api_secret=api_secret, testnet=True)  # type: ignore


def create_fix_simulator() -> FIXSimulator:
    """Quick-create a FIX simulator."""
    return get_broker("fix")  # type: ignore
