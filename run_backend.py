"""
Quantex backend runner — lightweight dispatch over existing components.
"""
from __future__ import annotations

import asyncio
import logging
import signal
import sys
from dataclasses import dataclass
from typing import Iterable

from config import AppConfig, ExchangeConfig, InferenceConfig, RiskConfig
from binance.adapter import to_binance_symbol, to_binance_interval, normalize_symbols, normalize_timeframes
from binance.client import BinanceClient


def _build_exchange_config() -> ExchangeConfig:
    return ExchangeConfig(
        name="binance_spot",
        rest_base="https://api.binance.com/api/v3",
        ws_base="wss://stream.binance.com:9443/stream",
        symbols=tuple(normalize_symbols(["BTC/USDT", "ETH/USDT", "SOL/USDT"])),
        timeframes=tuple(normalize_timeframes(["1h", "4h", "15m"])),
        default_symbol="BTC/USDT",
        default_timeframe="1h",
    )


def build_config() -> AppConfig:
    return AppConfig(
        app_name="quantex",
        mode="paper",
        environment="development",
        log_level="INFO",
        exchange=_build_exchange_config(),
        risk=RiskConfig(),
        inference=InferenceConfig(
            primary_provider="nvidia_nim",
            fallback_providers=("local_ollama",),
            budget_tier="free",
        ),
    )


@dataclass(frozen=True)
class SymbolTimeframe:
    symbol: str
    timeframe: str


class BackendRunner:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or build_config()
        self.logger = logging.getLogger("quantex.backend")
        self.client = BinanceClient(base_url=self.config.exchange.rest_base)
        self._shutdown = asyncio.Event()
        self._planned: list[SymbolTimeframe] = []
        self._running_tasks: list[asyncio.Task] = []

    def _plan(self) -> list[SymbolTimeframe]:
        if self._planned:
            return self._planned
        out: list[SymbolTimeframe] = []
        for symbol in self.config.exchange.symbols:
            for timeframe in self.config.exchange.timeframes:
                out.append(SymbolTimeframe(symbol=symbol, timeframe=timeframe))
        self._planned = out
        return out

    async def _run_symbol_pipeline(self, st: SymbolTimeframe) -> None:
        binance_symbol = to_binance_symbol(st.symbol)
        binance_interval = to_binance_interval(st.timeframe)
        self.logger.info("pipeline start symbol=%s timeframe=%s binance=%s", st.symbol, st.timeframe, binance_symbol)
        while not self._shutdown.is_set():
            try:
                health = await self.client.health()
                if health.get("status") != "ok":
                    self.logger.warning("exchange health degraded: %s", health)
                candles = await self.client.fetch_klines(binance_symbol, binance_interval, limit=200)
                ticker = await self.client.fetch_24h_ticker(binance_symbol)
                self.logger.debug(
                    "pipeline update symbol=%s timeframe=%s candles=%s last_price=%s",
                    st.symbol,
                    st.timeframe,
                    len(candles),
                    ticker.get("lastPrice"),
                )
            except Exception as exc:
                self.logger.exception("pipeline error symbol=%s timeframe=%s: %s", st.symbol, st.timeframe, exc)
            await asyncio.sleep(30)

    async def _run_inference_health_loop(self) -> None:
        self.logger.info("inference health loop starting")
        while not self._shutdown.is_set():
            try:
                from orchestrator.inference_integration import InferenceIntegration
                integration = InferenceIntegration(budget_tier=self.config.inference.budget_tier)
                health = await integration.get_provider_health()
                self.logger.info("inference provider health: %s", health)
            except Exception as exc:
                self.logger.warning("inference health check failed: %s", exc)
            await asyncio.sleep(60)

    async def start(self) -> None:
        logging.basicConfig(level=getattr(logging, self.config.log_level.upper(), logging.INFO))
        self.logger.info("starting backend mode=%s", self.config.mode)
        plan = self._plan()
        self.logger.info("markets configured symbols=%s timeframes=%s", self.config.exchange.symbols, self.config.exchange.timeframes)
        symbol_tasks = [asyncio.create_task(self._run_symbol_pipeline(st)) for st in plan]
        inference_task = asyncio.create_task(self._run_inference_health_loop())
        self._running_tasks = [*symbol_tasks, inference_task]
        try:
            await asyncio.gather(*self._running_tasks, return_exceptions=True)
        except asyncio.CancelledError:
            pass

    async def stop(self) -> None:
        self.logger.info("shutting down backend")
        self._shutdown.set()
        for task in self._running_tasks:
            task.cancel()
        await asyncio.gather(*self._running_tasks, return_exceptions=True)


async def _serve() -> None:
    runner = BackendRunner()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: asyncio.ensure_future(runner.stop()))
        except NotImplementedError:
            pass
    await runner.start()


if __name__ == "__main__":
    asyncio.run(_serve())
