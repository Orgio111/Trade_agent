"""
Shared application configuration with validated Binance/trading defaults.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class ExchangeConfig:
    name: str
    rest_base: str
    ws_base: str
    symbols: tuple[str, ...]
    timeframes: tuple[str, ...]
    default_symbol: str
    default_timeframe: str

    @classmethod
    def binance_spot(cls) -> "ExchangeConfig":
        return cls(
            name="binance_spot",
            rest_base="https://api.binance.com/api/v3",
            ws_base="wss://stream.binance.com:9443/stream",
            symbols=("BTC/USDT", "ETH/USDT", "SOL/USDT"),
            timeframes=("1h", "4h", "15m"),
            default_symbol="BTC/USDT",
            default_timeframe="1h",
        )


@dataclass(frozen=True)
class RiskConfig:
    initial_balance: float = 1000.0
    max_open_positions: int = 5
    max_consecutive_losses: int = 5
    hard_cap_daily_loss_pct: float = 0.05
    hard_cap_max_drawdown_pct: float = 0.20
    maker_fee: float = 0.0002
    taker_fee: float = 0.0004
    slippage_bps: float = 0.3
    max_leverage: int = 10


@dataclass(frozen=True)
class InferenceConfig:
    primary_provider: str = "local_ollama"
    fallback_providers: tuple[str, ...] = ("nvidia_nim", "groq", "openrouter")
    budget_tier: str = "free"
    cache_enabled: bool = True
    cache_similarity_threshold: float = 0.92
    nvidia_api_key: str = ""
    groq_api_key: str = ""
    openrouter_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"


@dataclass(frozen=True)
class AppConfig:
    app_name: str = "quantex"
    mode: Literal["paper", "live"] = "paper"
    environment: str = "development"
    log_level: str = "INFO"
    exchange: ExchangeConfig = ExchangeConfig.binance_spot()
    risk: RiskConfig = RiskConfig()
    inference: InferenceConfig = InferenceConfig()

