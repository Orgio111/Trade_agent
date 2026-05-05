"""Redis Streams inter-agent message bus with SBE-inspired zero-copy framing."""
from __future__ import annotations

import asyncio
import json
import logging
import struct
import time
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Callable

import redis.asyncio as aioredis
from pydantic import BaseModel

from core.config import get_settings

logger = logging.getLogger(__name__)

# ── Simple Binary Encoding header (8 bytes): type(2) | version(1) | flags(1) | length(4)
_SBE_MAGIC = 0xAB
_HEADER_FMT = "!BBHi"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)


def encode_sbe(msg_type: int, payload: bytes) -> bytes:
    header = struct.pack(_HEADER_FMT, _SBE_MAGIC, msg_type, 1, len(payload))
    return header + payload


def decode_sbe(data: bytes) -> tuple[int, bytes]:
    magic, msg_type, _version, length = struct.unpack_from(_HEADER_FMT, data)
    if magic != _SBE_MAGIC:
        raise ValueError(f"Invalid SBE magic byte: {magic:#x}")
    return msg_type, data[_HEADER_SIZE : _HEADER_SIZE + length]


@dataclass
class StreamMessage:
    stream: str
    message_id: str
    msg_type: int
    data: dict


class MessageBus:
    """Redis Streams-backed async message bus."""

    def __init__(self) -> None:
        self._pool: aioredis.Redis | None = None

    async def connect(self) -> None:
        cfg = get_settings()
        self._pool = await aioredis.from_url(
            cfg.redis_url,
            encoding="utf-8",
            decode_responses=False,
            socket_keepalive=True,
        )
        logger.info("MessageBus connected to Redis")

    async def close(self) -> None:
        if self._pool:
            await self._pool.aclose()

    @property
    def redis(self) -> aioredis.Redis:
        if self._pool is None:
            raise RuntimeError("MessageBus not connected — call await connect()")
        return self._pool

    async def publish(
        self, stream: str, msg_type: int, data: dict, maxlen: int | None = None
    ) -> str:
        cfg = get_settings()
        maxlen = maxlen or cfg.redis_max_stream_len
        payload = json.dumps(data).encode()
        frame = encode_sbe(msg_type, payload)
        msg_id = await self.redis.xadd(
            stream, {"sbe": frame}, maxlen=maxlen, approximate=True
        )
        return msg_id.decode() if isinstance(msg_id, bytes) else msg_id

    async def subscribe(
        self,
        streams: list[str],
        group: str,
        consumer: str,
        batch: int = 10,
        block_ms: int = 500,
    ) -> AsyncGenerator[StreamMessage, None]:
        for stream in streams:
            try:
                await self.redis.xgroup_create(stream, group, id="0", mkstream=True)
            except aioredis.ResponseError:
                pass  # group already exists

        last_ids = {s: ">" for s in streams}
        while True:
            try:
                results = await self.redis.xreadgroup(
                    group,
                    consumer,
                    {s: last_ids[s] for s in streams},
                    count=batch,
                    block=block_ms,
                )
                if not results:
                    continue
                for stream_name, messages in results:
                    sname = (
                        stream_name.decode()
                        if isinstance(stream_name, bytes)
                        else stream_name
                    )
                    for msg_id, fields in messages:
                        mid = (
                            msg_id.decode() if isinstance(msg_id, bytes) else msg_id
                        )
                        raw = fields.get(b"sbe") or fields.get("sbe")
                        if not raw:
                            continue
                        msg_type, payload = decode_sbe(raw)
                        data = json.loads(payload)
                        yield StreamMessage(
                            stream=sname,
                            message_id=mid,
                            msg_type=msg_type,
                            data=data,
                        )
                        await self.redis.xack(sname, group, mid)
            except Exception as exc:
                logger.error("Stream read error: %s — retrying in 1s", exc)
                await asyncio.sleep(1)


# ── Message type registry ─────────────────────────────────────────────────────
class MsgType:
    TICK = 1
    TECHNICAL_SIGNAL = 2
    FUNDAMENTAL_SIGNAL = 3
    SENTIMENT_SIGNAL = 4
    COUNCIL_DECISION = 5
    RISK_REPORT = 6
    ORDER = 7
    TRADE_OUTCOME = 8
    KILL_SWITCH = 9
    ALPHA_FACTORS = 10
    MEMORY_UPDATE = 11
    PORTFOLIO_STATE = 12


_bus: MessageBus | None = None


async def get_bus() -> MessageBus:
    global _bus
    if _bus is None:
        _bus = MessageBus()
        await _bus.connect()
    return _bus
