"""Fundamental Agent: on-chain metrics, NVT, MVRV, earnings, revenue growth."""
from __future__ import annotations

import logging
from datetime import datetime

import httpx

from core.config import get_settings
from core.messaging import MsgType, get_bus
from core.models import FundamentalSignal, Side
from core.nim_client import nim_json
from core.observability import AGENT_LATENCY, SIGNAL_COUNTER

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a fundamental analyst at a top-tier crypto/equities hedge fund.
Given on-chain metrics and macro indicators, output a JSON with:
{
  "on_chain_score": <float -1 to 1>,
  "nvt_ratio": <float or null>,
  "mvrv_zscore": <float or null>,
  "trend": "BUY" | "SELL" | "HOLD",
  "confidence": <float 0 to 1>,
  "rationale": "<one-sentence>"
}
Be precise and conservative. Negative scores indicate bearish fundamentals."""


class FundamentalAgent:
    """Fetches on-chain/macro data and uses NIM to score fundamentals."""

    async def analyze(self, symbol: str) -> FundamentalSignal:
        with AGENT_LATENCY.labels(agent="fundamental").time():
            metrics = await self._fetch_on_chain(symbol)
            result = await nim_json(
                [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Symbol: {symbol}\nMetrics: {metrics}\n"
                            "Analyze fundamentals and return JSON."
                        ),
                    },
                ]
            )

        trend_str = result.get("trend", "HOLD")
        signal = FundamentalSignal(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            on_chain_score=float(result.get("on_chain_score", 0.0)),
            nvt_ratio=result.get("nvt_ratio"),
            mvrv_zscore=result.get("mvrv_zscore"),
            trend=Side(trend_str),
            confidence=float(result.get("confidence", 0.5)),
            rationale=result.get("rationale", ""),
        )

        cfg = get_settings()
        bus = await get_bus()
        await bus.publish(
            cfg.stream_signals, MsgType.FUNDAMENTAL_SIGNAL, signal.model_dump(mode="json")
        )
        SIGNAL_COUNTER.labels(agent="fundamental", symbol=symbol, side=trend_str).inc()
        return signal

    async def _fetch_on_chain(self, symbol: str) -> dict:
        """Fetch on-chain metrics from public APIs (CryptoCompare / Glassnode proxy)."""
        base_asset = symbol.split("/")[0].upper()
        metrics: dict = {"symbol": symbol, "base_asset": base_asset}
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                # CryptoCompare social/blockchain stats (no API key needed for basic)
                url = (
                    f"https://min-api.cryptocompare.com/data/blockchain/histo/day"
                    f"?fsym={base_asset}&limit=7"
                )
                r = await client.get(url)
                if r.status_code == 200:
                    data = r.json().get("Data", {}).get("Data", [])
                    if data:
                        latest = data[-1]
                        metrics["active_addresses"] = latest.get("active_addresses", 0)
                        metrics["transaction_count"] = latest.get("transaction_count", 0)
                        metrics["hashrate"] = latest.get("hashrate", None)
        except Exception as exc:
            logger.warning("On-chain fetch failed for %s: %s", symbol, exc)
        return metrics
