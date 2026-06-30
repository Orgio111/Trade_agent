"""
QUANTEX NATS↔LangGraph Bridge — Subscribes to NATS market.candle events
and triggers the multi-agent LangGraph pipeline.

Architecture:
  NATS market.candle.<symbol> → CandleSubscriber → MultiAgentPipeline → NATS signals.>/ws.>

Usage:
    bridge = NATSLangGraphBridge(nats_url="nats://localhost:4222")
    await bridge.start()
    # Bridge listens for candle_close events and runs the full pipeline
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Optional

import nats

from .langgraph_pipeline import MultiAgentPipeline
from .candle_buffer import CandleBuffer
from .events import (
    TradeSignalEvent,
    AggregatedSignalEvent,
    RiskCheckEvent,
    RiskApprovedEvent,
    RiskRejectedEvent,
    SystemEvent,
    WSEvent,
    MarketDataEvent,
    ScalpDecisionEvent,
    quantex_event,
)

logger = logging.getLogger("quantex.nats_bridge")

DEFAULT_NATS_URL = os.getenv("NATS_URL", "nats://localhost:4222")


class NATSLangGraphBridge:
    """
    Bridge between NATS JetStream and LangGraph pipeline.

    Subscribes to:
      - market.candle.<symbol>  → triggers pipeline
      - signals.raw.>           → feeds into candle buffer context

    Publishes:
      - signals.raw.<source>.<symbol>  → pipeline output signals
      - system.risk.check/approved/rejected → risk decisions
      - system.pipeline.stage           → pipeline metrics
      - ws.update                       → frontend updates
    """

    def __init__(
        self,
        nats_url: str = DEFAULT_NATS_URL,
        symbols: list[str] | None = None,
    ):
        self.nats_url = nats_url
        self.symbols = symbols or ["BTCUSDT"]
        self.nc: Optional[nats.NATS] = None
        self.pipeline = MultiAgentPipeline()
        self.candle_buffer = CandleBuffer(max_candles=200)
        self._running = False
        self._processed_count = 0
        self._error_count = 0

    async def start(self) -> bool:
        """Connect to NATS and start subscribing."""
        try:
            self.nc = await nats.connect(
                self.nats_url,
                name="quantex-langgraph-bridge",
                reconnect_time_wait=2,
                max_reconnect_attempts=60,
            )
            logger.info(f"Connected to NATS at {self.nats_url}")

            # Connect candle buffer
            await self.candle_buffer.connect()

            # Subscribe to candle close events (wildcard covers all symbols)
            await self.nc.subscribe("market.candle.>", cb=self._on_candle_close)
            logger.info(f"Subscribed to market.candle.> (covers {self.symbols})")

            self._running = True
            logger.info(
                f"NATS↔LangGraph bridge started — "
                f"symbols={self.symbols}, pipeline ready"
            )
            return True

        except Exception as e:
            logger.error(f"NATS bridge failed to start: {e}")
            return False

    async def _on_candle_close(self, msg):
        """Handle incoming candle close event from NATS."""
        try:
            try:
                data = json.loads(msg.data.decode())
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.warning(f"[Bridge] Invalid NATS message: {e}")
                return

            # Extract candle data
            if "data" in data:
                candle = data["data"]
            else:
                candle = data

            symbol = candle.get("symbol", msg.subject.split(".")[-1])

            # Update candle buffer
            await self.candle_buffer.push(symbol, candle)

            # Get candle history for features
            history = await self.candle_buffer.get(symbol, limit=200)

            # Run pipeline
            start = time.perf_counter()
            candle_input = {
                **candle,
                "symbol": symbol,
                "interval": candle.get("interval", "1m"),
            }

            result = await self.pipeline.process_candle(candle_input)
            total_ms = (time.perf_counter() - start) * 1000

            self._processed_count += 1

            # Publish results to NATS
            await self._publish_results(symbol, result)

            logger.info(
                f"[Bridge] Processed {symbol} candle: "
                f"stage={result['stage']} "
                f"strategy={result['strategy'].get('direction', '?')} "
                f"total={total_ms:.0f}ms"
            )

        except Exception as e:
            self._error_count += 1
            logger.error(f"[Bridge] Candle processing error: {e}")

    async def _publish_results(self, symbol: str, result: dict):
        """Publish pipeline results as NATS events."""
        strategy = result.get("strategy", {})
        risk = result.get("risk", {})
        execution = result.get("execution", {})
        trace_id = result.get("trace_id", "")

        # 0. Publish ScalpDecisionEvent when scalping fast-path triggered
        if strategy.get("source") == "scalping_fast_path":
            scalp_event = ScalpDecisionEvent(
                source="scalping_engine",
                symbol=symbol,
                action="BUY" if strategy.get("direction") == "long" else "SELL",
                confidence=int(strategy.get("confidence", 0) * 100),
                size_pct=strategy.get("size_pct", 0),
                reason=strategy.get("reasoning", "")[:200],
                latency_ms=result.get("latency_ms", {}).get("scalping", 0),
                trace_id=trace_id,
                priority="high",
                context={
                    "threshold": result.get("context", {}).get("scalping_threshold", 80),
                    "fast_path": True,
                    "vlm_skipped": True,
                    "rag_skipped": True,
                    "swarm_skipped": True,
                },
            )
            await self.nc.publish(scalp_event.subject, scalp_event.to_json().encode())
            logger.debug(
                f"[Bridge] Published ScalpDecisionEvent: {scalp_event.action} "
                f"conf={scalp_event.confidence}% size={scalp_event.size_pct}%"
            )

        # 1. Publish strategy signal as TradeSignalEvent
        if strategy.get("direction", "hold") != "hold":
            signal_event = TradeSignalEvent(
                source="langgraph_pipeline",
                symbol=symbol,
                signal=strategy["direction"],
                confidence=strategy.get("confidence", 0),
                price=result.get("execution", {}).get("price", 0),
                reason=strategy.get("reasoning", "")[:200],
                trace_id=trace_id,
                priority="high",
                context=result.get("context", {}),
            )
            await self.nc.publish(signal_event.subject, signal_event.to_json().encode())

        # 2. Publish risk decision events
        risk_check = RiskCheckEvent(
            source="langgraph_pipeline",
            symbol=symbol,
            signal=strategy.get("direction", "hold"),
            confidence=strategy.get("confidence", 0),
            trace_id=trace_id,
        )
        await self.nc.publish(risk_check.subject, risk_check.to_json().encode())

        if risk.get("allowed", False):
            risk_approved = RiskApprovedEvent(
                source="langgraph_pipeline",
                symbol=symbol,
                signal=strategy.get("direction", "hold"),
                approved_size=execution.get("size_pct", 0),
                max_leverage=risk.get("max_leverage", 1),
                risk_score=risk.get("size_multiplier", 1.0),
                reasoning=risk.get("reason", ""),
                trace_id=trace_id,
            )
            await self.nc.publish(risk_approved.subject, risk_approved.to_json().encode())
        else:
            risk_rejected = RiskRejectedEvent(
                source="langgraph_pipeline",
                symbol=symbol,
                signal=strategy.get("direction", "hold"),
                reason=risk.get("reason", ""),
                severity=risk.get("severity", "hard"),
                trace_id=trace_id,
            )
            await self.nc.publish(risk_rejected.subject, risk_rejected.to_json().encode())

        # 3. Publish pipeline metrics as SystemEvent
        metrics_event = SystemEvent(
            source="langgraph_pipeline",
            event_type="info",
            level="info",
            message=json.dumps({
                "trace_id": trace_id,
                "symbol": symbol,
                "stage": result.get("stage"),
                "total_latency_ms": result.get("total_latency_ms", 0),
                "strategy_source": strategy.get("source"),
                "risk_allowed": risk.get("allowed", False),
                "errors": result.get("errors", []),
            }),
            component="langgraph_pipeline",
            trace_id=trace_id,
            priority="low",
        )
        await self.nc.publish(metrics_event.subject, metrics_event.to_json().encode())

        # 4. Publish WS update for frontend
        ws_event = WSEvent(
            source="langgraph_pipeline",
            event_type="update",
            payload={
                "type": "pipeline_result",
                "symbol": symbol,
                "strategy": strategy,
                "risk": risk,
                "execution": execution,
                "latency_ms": result.get("latency_ms", {}),
                "trace_id": trace_id,
            },
            trace_id=trace_id,
        )
        await self.nc.publish(ws_event.subject, ws_event.to_json().encode())

    async def process_single(self, symbol: str = "BTCUSDT", candle: dict | None = None) -> dict:
        """Process a single candle manually (for testing)."""
        if candle is None:
            candle = {
                "symbol": symbol,
                "open": 50000, "high": 50500, "low": 49800,
                "close": 50200, "volume": 100.0,
                "interval": "1m",
            }

        await self.candle_buffer.push(symbol, candle)
        history = await self.candle_buffer.get(symbol, limit=200)
        result = await self.pipeline.process_candle(candle)
        return result

    def get_stats(self) -> dict:
        """Get bridge statistics."""
        return {
            "processed": self._processed_count,
            "errors": self._error_count,
            "running": self._running,
            "nats_connected": self.nc is not None and self.nc.is_connected,
            "symbols": self.symbols,
        }

    async def stop(self):
        """Stop the bridge and disconnect."""
        self._running = False
        self.pipeline.stop()
        await self.candle_buffer.close()
        if self.nc:
            await self.nc.drain()
            await self.nc.close()
        logger.info("NATS↔LangGraph bridge stopped")
