"""
Memory Agent: PostgreSQL-backed post-trade reflection.
Analyzes trade outcomes and updates system prompts for the Analyst agents.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

import asyncpg  # type: ignore[import]

from core.config import get_settings
from core.messaging import MsgType, get_bus
from core.models import Order, OrderStatus, Side, TradeOutcome
from core.nim_client import nim_json
from core.observability import AGENT_LATENCY

logger = logging.getLogger(__name__)

_REFLECTION_SYSTEM = """You are a trading post-mortem analyst at a top hedge fund.
Given a trade's setup signals and outcome, analyze what went wrong (or right).
Output JSON:
{
  "failure_analysis": "<2-3 sentences explaining the root cause of loss or confirming win>",
  "updated_prompts": {
    "technical": "<refined instruction for the technical analyst, or empty string>",
    "fundamental": "<refined instruction, or empty string>",
    "sentiment": "<refined instruction, or empty string>"
  }
}
For winning trades, focus on what signals were most predictive."""


class MemoryAgent:
    """Persists trades to PostgreSQL and performs post-trade reflection via NIM."""

    def __init__(self) -> None:
        self._pool: asyncpg.Pool | None = None

    def is_connected(self) -> bool:
        return self._pool is not None

    async def connect(self) -> None:
        cfg = get_settings()
        # asyncpg uses a different DSN format (no +asyncpg)
        dsn = cfg.pg_dsn.replace("postgresql+asyncpg://", "postgresql://")
        self._pool = await asyncpg.create_pool(dsn, min_size=2, max_size=10)
        await self._ensure_schema()
        logger.info("MemoryAgent connected to PostgreSQL")

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()

    async def _ensure_schema(self) -> None:
        async with self._pool.acquire() as conn:  # type: ignore[union-attr]
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    order_id      TEXT PRIMARY KEY,
                    session_id    TEXT NOT NULL,
                    symbol        TEXT NOT NULL,
                    side          TEXT NOT NULL,
                    entry_price   DOUBLE PRECISION,
                    exit_price    DOUBLE PRECISION,
                    quantity      DOUBLE PRECISION,
                    pnl           DOUBLE PRECISION,
                    pnl_pct       DOUBLE PRECISION,
                    win           BOOLEAN,
                    created_at    TIMESTAMPTZ DEFAULT NOW(),
                    closed_at     TIMESTAMPTZ,
                    council_json  JSONB,
                    risk_json     JSONB,
                    failure_analysis TEXT,
                    updated_prompts  JSONB
                );

                CREATE TABLE IF NOT EXISTS agent_prompts (
                    agent_name    TEXT PRIMARY KEY,
                    base_prompt   TEXT NOT NULL,
                    refinements   TEXT[] DEFAULT '{}',
                    updated_at    TIMESTAMPTZ DEFAULT NOW()
                );

                CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
                CREATE INDEX IF NOT EXISTS idx_trades_win ON trades(win);
            """)

    async def record_open(self, order: Order, council_json: dict, risk_json: dict) -> None:
        if not self._pool:
            return
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO trades
                   (order_id, session_id, symbol, side, entry_price, quantity, council_json, risk_json)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
                   ON CONFLICT (order_id) DO NOTHING""",
                order.order_id,
                order.session_id,
                order.symbol,
                order.side.value,
                order.avg_fill_price,
                order.quantity,
                json.dumps(council_json),
                json.dumps(risk_json),
            )

    async def record_close(self, order_id: str, exit_price: float) -> TradeOutcome | None:
        if not self._pool:
            return None
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM trades WHERE order_id=$1", order_id
            )
            if not row:
                return None

            entry_price = row["entry_price"] or exit_price
            quantity = row["quantity"] or 0.0
            side = row["side"]
            pnl_per_unit = (exit_price - entry_price) if side == "BUY" else (entry_price - exit_price)
            pnl = pnl_per_unit * quantity
            pnl_pct = pnl_per_unit / (entry_price + 1e-10)

            await conn.execute(
                """UPDATE trades SET exit_price=$1, pnl=$2, pnl_pct=$3, win=$4,
                   closed_at=NOW() WHERE order_id=$5""",
                exit_price,
                pnl,
                pnl_pct,
                pnl > 0,
                order_id,
            )

            outcome = TradeOutcome(
                order_id=order_id,
                session_id=row["session_id"],
                symbol=row["symbol"],
                side=Side(side),
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=quantity,
                pnl=pnl,
                pnl_pct=pnl_pct,
                win=pnl > 0,
            )

        await self._reflect(outcome, row)
        return outcome

    async def _reflect(self, outcome: TradeOutcome, row: Any) -> None:
        with AGENT_LATENCY.labels(agent="memory").time():
            council_json = row["council_json"] or {}
            context = (
                f"Symbol: {outcome.symbol}\n"
                f"Side: {outcome.side.value}\n"
                f"Entry: {outcome.entry_price:.4f} → Exit: {outcome.exit_price:.4f}\n"
                f"PnL: {outcome.pnl:.2f} ({outcome.pnl_pct:.2%})\n"
                f"Win: {outcome.win}\n"
                f"Council context: {council_json}"
            )
            result = await nim_json(
                [
                    {"role": "system", "content": _REFLECTION_SYSTEM},
                    {"role": "user", "content": context},
                ]
            )

        outcome.failure_analysis = result.get("failure_analysis", "")
        outcome.updated_prompts = result.get("updated_prompts", {})

        if not self._pool:
            return

        async with self._pool.acquire() as conn:
            await conn.execute(
                """UPDATE trades SET failure_analysis=$1, updated_prompts=$2
                   WHERE order_id=$3""",
                outcome.failure_analysis,
                json.dumps(outcome.updated_prompts),
                outcome.order_id,
            )
            # Persist refined prompts for each agent
            for agent_name, refinement in outcome.updated_prompts.items():
                if refinement:
                    await conn.execute(
                        """INSERT INTO agent_prompts (agent_name, base_prompt, refinements)
                           VALUES ($1, '', ARRAY[$2::TEXT])
                           ON CONFLICT (agent_name) DO UPDATE
                           SET refinements = agent_prompts.refinements || $2::TEXT,
                               updated_at = NOW()""",
                        agent_name,
                        refinement,
                    )

        cfg = get_settings()
        bus = await get_bus()
        await bus.publish(
            cfg.stream_memory,
            MsgType.TRADE_OUTCOME,
            outcome.model_dump(mode="json"),
        )
        logger.info(
            "Memory reflection: %s %s PnL=%.2f  %s",
            outcome.symbol,
            outcome.side.value,
            outcome.pnl or 0,
            "WIN" if outcome.win else "LOSS",
        )

    async def get_win_rate(self, symbol: str, lookback: int = 100) -> float:
        if not self._pool:
            return 0.55
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT AVG(win::int) as wr FROM (
                     SELECT win FROM trades WHERE symbol=$1 AND win IS NOT NULL
                     ORDER BY closed_at DESC LIMIT $2
                   ) sub""",
                symbol,
                lookback,
            )
            return float(row["wr"] or 0.55)

    async def get_avg_pnl_stats(self, symbol: str, lookback: int = 100) -> tuple[float, float]:
        if not self._pool:
            return 0.015, 0.010
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT
                     AVG(CASE WHEN win THEN pnl_pct END) as avg_win,
                     AVG(CASE WHEN NOT win THEN ABS(pnl_pct) END) as avg_loss
                   FROM (
                     SELECT win, pnl_pct FROM trades WHERE symbol=$1 AND win IS NOT NULL
                     ORDER BY closed_at DESC LIMIT $2
                   ) sub""",
                symbol,
                lookback,
            )
            return float(row["avg_win"] or 0.015), float(row["avg_loss"] or 0.010)
