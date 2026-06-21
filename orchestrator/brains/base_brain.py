"""Base class for all Trinity Architecture AI Brains.

Every brain must implement `compute_score()` which returns a BrainSignal.
The BrainRunner orchestrates NATS publishing on a fixed cadence.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import nats
from nats.js.client import JetStreamContext

logger = logging.getLogger(__name__)

# ── Signal Data Structure ────────────────────────────────────

@dataclass
class BrainSignal:
    """Signal published by each brain to NATS signals.raw."""
    brain_id: str
    symbol: str
    score: float          # -1.0 (strong sell) to 1.0 (strong buy)
    confidence: float     # 0.0 to 1.0
    weight: float = 0.0   # brain weight from Go orchestrator config
    direction: int = 0    # +1 buy, -1 sell, 0 hold
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> bytes:
        return json.dumps({
            "brain_id": self.brain_id,
            "symbol": self.symbol,
            "score": round(self.score, 6),
            "confidence": round(self.confidence, 4),
            "weight": self.weight,
            "direction": self.direction,
            "timestamp_ms": self.timestamp_ms,
            "metadata": self.metadata,
        }).encode()


# ── Abstract Brain ───────────────────────────────────────────

class BaseBrain(ABC):
    """Abstract base for all Trinity Architecture brains.

    Subclasses MUST implement:
        - brain_id: str property
        - compute_score(symbol: str) -> BrainSignal

    Subclasses MAY override:
        - warmup() — async init (model loading, etc.)
        - cooldown() — async cleanup
    """

    @property
    @abstractmethod
    def brain_id(self) -> str:
        """Unique identifier matching Go orchestrator weight key."""
        ...

    @abstractmethod
    async def compute_score(self, symbol: str) -> BrainSignal:
        """Compute and return a trading signal for the given symbol."""
        ...

    async def warmup(self) -> None:
        """Optional: load models, warm caches, etc. Called once at startup."""
        pass

    async def cooldown(self) -> None:
        """Optional: release resources. Called once on shutdown."""
        pass


# ── NATS Publisher ───────────────────────────────────────────

class BrainNATSPublisher:
    """Publishes brain signals to NATS JetStream `signals.raw`."""

    def __init__(self, nats_url: str | None = None, subject: str = "signals.raw"):
        self.nats_url = nats_url or os.getenv("NATS_URL", "nats://localhost:4222")
        self.subject = subject
        self.nc: nats.NATS | None = None
        self.js: JetStreamContext | None = None

    async def connect(self) -> None:
        """Connect to NATS server and ensure JetStream stream exists."""
        self.nc = await nats.connect(
            self.nats_url,
            name="quantex-brain-publisher",
            reconnect_wait=2,
            max_reconnect=60,
        )
        self.js = self.nc.jetstream()

        # Ensure stream exists
        try:
            await self.js.stream_info("signals")
        except Exception:
            await self.js.add_stream(
                name="signals",
                subjects=["signals.raw", "signals.aggregated", "signals.executed"],
                retention="limits",
                max_msgs=100_000,
                max_age=86400 * 3,  # 3 days
                storage="file",
            )
            logger.info("Created NATS JetStream stream 'signals'")

        logger.info(f"Connected to NATS at {self.nats_url}")

    async def publish(self, signal: BrainSignal) -> None:
        """Publish a brain signal to signals.raw."""
        if self.js is None:
            await self.connect()
        assert self.js is not None

        ack = await self.js.publish(self.subject, signal.to_json())
        logger.debug(
            f"[{signal.brain_id}] Published score={signal.score:.4f} "
            f"conf={signal.confidence:.2f} for {signal.symbol} (seq={ack.seq})"
        )

    async def close(self) -> None:
        """Drain and close NATS connection."""
        if self.nc is not None:
            await self.nc.drain()
            await self.nc.close()


# ── Brain Runner ─────────────────────────────────────────────

class BrainRunner:
    """Orchestrates one brain: calls compute_score on a cadence and
    publishes the result to NATS JetStream."""

    def __init__(
        self,
        brain: BaseBrain,
        publisher: BrainNATSPublisher,
        interval_secs: float = 15.0,
    ):
        self.brain = brain
        self.publisher = publisher
        self.interval_secs = interval_secs
        self._running = False

    async def run(self, symbol: str) -> None:
        """Run the brain loop: compute → publish → sleep → repeat."""
        self._running = True
        logger.info(f"[{self.brain.brain_id}] Starting (interval={self.interval_secs}s)")

        await self.brain.warmup()

        while self._running:
            try:
                signal = await asyncio.wait_for(
                    self.brain.compute_score(symbol),
                    timeout=self.interval_secs * 0.8,
                )
                await self.publisher.publish(signal)
            except asyncio.TimeoutError:
                logger.warning(f"[{self.brain.brain_id}] compute_score timed out")
            except Exception as e:
                logger.error(f"[{self.brain.brain_id}] Error: {e}", exc_info=True)

            await asyncio.sleep(self.interval_secs)

    def stop(self) -> None:
        self._running = False

    async def shutdown(self) -> None:
        self.stop()
        await self.brain.cooldown()
