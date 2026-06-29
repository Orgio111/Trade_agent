"""Configuration loader with environment variable support."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file with environment variable overrides."""
    config_path = Path(path)
    if not config_path.exists():
        # Return defaults if config file doesn't exist
        return get_default_config()
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}
    
    # Apply environment variable overrides
    config = apply_env_overrides(config)
    
    return config


def get_default_config() -> Dict[str, Any]:
    """Return default configuration."""
    return {
        "system": {
            "symbol": "BTCUSDT",
            "timeframe": "1m",
            "paper_mode": True,
            "exchange": "binance_testnet",
            "testnet": True,
        },
        "model": {
            "warm_model": "phi3:3.8b",
            "reasoning_model": "qwen3:8b",
            "deep_model": "deepseek-r1:8b",
            "execution_model": "mistral:7b",
            "vision_model": "moondream",
            "embedding_model": "nomic-embed-text",
            "ollama_host": "http://localhost:11434",
            "num_predict": 512,
            "temperature": 0.3,
            "top_p": 0.9,
        },
        "risk": {
            "max_position_pct": 0.05,
            "max_daily_drawdown": 0.03,
            "max_concurrent": 3,
            "kill_streak": 5,
            "min_confidence": 0.6,
            "volatility_filter": True,
            "atr_threshold_mult": 2.5,
        },
        "memory": {
            "chroma_path": "./memory",
            "buffer_size": 200,
            "similarity_top_k": 5,
        },
        "execution": {
            "maker_preference": True,
            "default_type": "LIMIT",
            "post_only": True,
            "max_slippage_bps": 5,
        },
        "data": {
            "ws_url": "wss://stream.binance.com:9443/ws/btcusdt@kline_1m",
            "testnet_ws_url": "wss://stream.binancefuture.com/ws/btcusdt@kline_1m",
            "reconnect_interval": 5,
            "buffer_maxlen": 1000,
        },
        "logging": {
            "level": "INFO",
            "file": "./logs/trading_ai.log",
            "max_bytes": 10485760,
            "backup_count": 5,
        },
    }


def apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides to config."""
    # Ollama host
    if ollama_host := os.getenv("OLLAMA_HOST"):
        config.setdefault("model", {})["ollama_host"] = ollama_host
    
    # Binance API keys
    if api_key := os.getenv("BINANCE_API_KEY"):
        config.setdefault("binance", {})["api_key"] = api_key
    if api_secret := os.getenv("BINANCE_API_SECRET"):
        config.setdefault("binance", {})["api_secret"] = api_secret
    
    # Paper mode
    if paper_mode := os.getenv("PAPER_MODE"):
        config.setdefault("system", {})["paper_mode"] = paper_mode.lower() == "true"
    
    # Symbol
    if symbol := os.getenv("TRADING_SYMBOL"):
        config.setdefault("system", {})["symbol"] = symbol
    
    # Log level
    if log_level := os.getenv("LOG_LEVEL"):
        config.setdefault("logging", {})["level"] = log_level
    
    return config


def save_config(config: Dict[str, Any], path: str = "config.yaml") -> None:
    """Save configuration to YAML file."""
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    # Test config loading
    cfg = load_config()
    print(yaml.dump(cfg, default_flow_style=False))