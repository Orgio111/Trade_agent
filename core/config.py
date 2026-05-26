"""System-wide configuration loaded from environment variables."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=(),
    )

    # ── NVIDIA NIM ──────────────────────────────────────────────────────────
    nim_api_key: str = Field(..., alias="NIM_API_KEY")
    nim_base_url: str = Field(
        "https://integrate.api.nvidia.com/v1", alias="NIM_BASE_URL"
    )
    nim_model: str = Field("meta/llama-3.1-70b-instruct", alias="NIM_MODEL")
    nim_embed_model: str = Field(
        "nvidia/nv-embedqa-e5-v5", alias="NIM_EMBED_MODEL"
    )
    nim_timeout: float = Field(30.0, alias="NIM_TIMEOUT")

    # ── Market data ──────────────────────────────────────────────────────────
    exchange: Literal["binance", "coinbase", "kraken"] = Field(
        "binance", alias="EXCHANGE"
    )
    """Primary exchange (legacy — kept for backward compat)."""

    exchanges_raw: str = Field(
        default="binance,bybit,hyperliquid", alias="EXCHANGES"
    )
    """Comma-separated list of exchanges to connect in priority order."""

    symbols_raw: str = Field(
        default='BTC/USDT,ETH/USDT', alias="SYMBOLS"
    )
    ws_reconnect_delay: float = Field(2.0, alias="WS_RECONNECT_DELAY")

    # ── Redis Streams ─────────────────────────────────────────────────────────
    redis_url: str = Field("redis://redis:6379/0", alias="REDIS_URL")
    stream_market_data: str = "stream:market_data"
    stream_signals: str = "stream:signals"
    stream_orders: str = "stream:orders"
    stream_risk: str = "stream:risk"
    stream_memory: str = "stream:memory"
    redis_max_stream_len: int = 100_000

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    pg_dsn: str = Field(
        "postgresql+asyncpg://trader:changeme@postgres:5432/tradingdb",
        alias="PG_DSN",
    )

    # ── Risk limits ───────────────────────────────────────────────────────────
    max_daily_drawdown_pct: float = Field(0.05, alias="MAX_DAILY_DRAWDOWN_PCT")
    kelly_fraction: float = Field(0.25, alias="KELLY_FRACTION")
    max_position_pct: float = Field(0.10, alias="MAX_POSITION_PCT")
    var_confidence: float = Field(0.99, alias="VAR_CONFIDENCE")
    var_lookback_days: int = Field(252, alias="VAR_LOOKBACK_DAYS")
    atr_stop_multiplier: float = Field(2.0, alias="ATR_STOP_MULTIPLIER")

    # ── Execution ─────────────────────────────────────────────────────────────
    paper_trading: bool = Field(True, alias="PAPER_TRADING")
    initial_capital: float = Field(100_000.0, alias="INITIAL_CAPITAL")
    max_slippage_bps: int = Field(10, alias="MAX_SLIPPAGE_BPS")
    order_timeout_s: float = Field(5.0, alias="ORDER_TIMEOUT_S")

    # ── RL Execution agent ────────────────────────────────────────────────────
    ppo_model_path: str = Field(
        "/models/ppo_execution/latest.zip", alias="PPO_MODEL_PATH"
    )
    ppo_retrain_interval_hours: int = Field(24, alias="PPO_RETRAIN_INTERVAL_HOURS")

    # ── MLOps ─────────────────────────────────────────────────────────────────
    psi_drift_threshold: float = Field(0.2, alias="PSI_DRIFT_THRESHOLD")
    wfo_train_months: int = Field(12, alias="WFO_TRAIN_MONTHS")
    wfo_test_months: int = Field(3, alias="WFO_TEST_MONTHS")
    model_registry_path: str = Field(
        "/models/mlops/registry", alias="MODEL_REGISTRY_PATH"
    )
    """Directory where versioned model checkpoints are stored."""
    mlops_check_interval_s: int = Field(
        3600, alias="MLOPS_CHECK_INTERVAL_S"
    )
    """How often the MLOps scheduler checks for drift (seconds)."""
    mlops_retrain_interval_hours: int = Field(
        24, alias="MLOPS_RETRAIN_INTERVAL_HOURS"
    )
    """Maximum age of a model before forced retrain (hours)."""
    mlops_retrain_cooldown_hours: int = Field(
        4, alias="MLOPS_RETRAIN_COOLDOWN_HOURS"
    )
    """Minimum time between retrain cycles (hours)."""
    mlops_retrain_timesteps: int = Field(
        500_000, alias="MLOPS_RETRAIN_TIMESTEPS"
    )
    """Total timesteps for each PPO retraining run."""
    mlops_enabled: bool = Field(
        True, alias="MLOPS_ENABLED"
    )
    """Toggle the entire MLOps pipeline on/off."""

    # ── Ray Serve Inference ───────────────────────────────────────────────────
    ray_serve_url: str = Field(
        "", alias="RAY_SERVE_URL"
    )
    """Base URL of the Ray Serve cluster (e.g. ``http://ray-head:8765``).
    Leave empty to disable remote PPO inference."""

    ray_serve_auto_reload_s: int = Field(
        300, alias="RAY_SERVE_AUTO_RELOAD_S"
    )
    """How often the Ray Serve client polls the model registry for newer versions."""

    # ── Observability ─────────────────────────────────────────────────────────
    prometheus_port: int = Field(8000, alias="PROMETHEUS_PORT")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    # ── Triton Inference Server ───────────────────────────────────────────────
    triton_url: str = Field("triton:8001", alias="TRITON_URL")

    # ── Sentinel-X integration ─────────────────────────────────────────────────
    sentinelx_risk_addr: str = Field(
        "", alias="SENTINELX_RISK_ADDR"
    )
    """gRPC address of the Rust Risk Engine, e.g. "sentinel-rust:50051".
    Leave empty to use the pure-Python risk engine."""

    sentinelx_gateway_url: str = Field(
        "", alias="SENTINELX_GATEWAY_URL"
    )
    """HTTP base URL of the Go Gateway, e.g. "http://sentinel-go:8080".
    Leave empty to use the local execution engine."""

    sentinelx_timeout_s: float = Field(8.0, alias="SENTINELX_TIMEOUT_S")

    # ── Council consensus ─────────────────────────────────────────────────────
    min_consensus_score: float = Field(0.65, alias="MIN_CONSENSUS_SCORE")
    council_timeout_s: float = Field(30.0, alias="COUNCIL_TIMEOUT_S")

    # ── Feature Extraction Engine ──────────────────────────────────────────────
    features_enabled: bool = Field(True, alias="FEATURES_ENABLED")
    """Toggle order-flow and funding rate feature extraction."""

    feature_trade_window_seconds: int = Field(
        60, alias="FEATURE_TRADE_WINDOW_S"
    )
    """Rolling window in seconds for accumulating trades (matches OHLCV bar)."""

    feature_funding_poll_seconds: int = Field(
        300, alias="FEATURE_FUNDING_POLL_S"
    )
    """How often to poll funding rates (seconds)."""

    feature_oi_poll_seconds: int = Field(
        60, alias="FEATURE_OI_POLL_S"
    )
    """How often to poll open interest (seconds)."""

    feature_cvd_lookback: int = Field(
        20, alias="FEATURE_CVD_LOOKBACK"
    )
    """Number of bars for rolling CVD normalisation / OI-price correlation."""

    # ── Backtesting ───────────────────────────────────────────────────────────
    backtest_mode: bool = Field(False, alias="BACKTEST")
    """Enable backtest mode — load historical data instead of live feeds."""

    backtest_start: str = Field("2024-01-01", alias="BACKTEST_START")
    """Start date for backtest data (ISO format, e.g. ``"2024-01-01"``)."""

    backtest_end: str = Field("2024-12-31", alias="BACKTEST_END")
    """End date for backtest data (ISO format)."""

    backtest_timeframe: str = Field("1h", alias="BACKTEST_TIMEFRAME")
    """Candle interval for backtest replay (e.g. ``"1h"``, ``"1m"``)."""

    backtest_slippage_bps: float = Field(5.0, alias="BACKTEST_SLIPPAGE_BPS")
    """Simulated slippage in basis points for backtest fills."""

    backtest_max_cycles: int = Field(0, alias="BACKTEST_MAX_CYCLES")
    """Limit bars processed in backtest (0 = all).  Useful for quick tests."""

    # ── TurboVec Memory Engine ─────────────────────────────────────────────────
    turbovec_index_dir: str = Field(
        ".turbovec", alias="TURBOVEC_INDEX_DIR"
    )
    """Directory where the TurboVec FAISS index and metadata JSONL are stored."""

    turbovec_dimension: int = Field(768, alias="TURBOVEC_DIMENSION")
    """Embedding dimension (nv-embedqa-e5-v5 outputs 768)."""

    turbovec_max_results: int = Field(8, alias="TURBOVEC_MAX_RESULTS")
    """Number of similar states to retrieve for memory context."""

    turbovec_compact_interval_hours: float = Field(1.0, alias="TURBOVEC_COMPACT_INTERVAL_HOURS")
    """How often to compact / re-train the FAISS index."""

    memory_enabled: bool = Field(True, alias="MEMORY_ENABLED")
    """Toggle semantic memory pipeline on/off."""

    @property
    def exchanges(self) -> list[str]:
        """Parse exchanges_raw into a priority-ordered list of exchange IDs.
        Falls back to the legacy `exchange` field if EXCHANGES is empty.
        """
        raw = self.exchanges_raw.strip()
        if raw:
            parts = [e.strip() for e in raw.split(",") if e.strip()]
            if parts:
                return parts
        return [self.exchange]

    @property
    def symbols(self) -> list[str]:
        """Parse symbols_raw into a list, supporting both JSON and comma-separated formats."""
        raw = self.symbols_raw.strip()
        if not raw:
            return []
        if raw.startswith("[") and raw.endswith("]"):
            import json
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                pass
        return [s.strip().strip('"').strip("'") for s in raw.split(",") if s.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
