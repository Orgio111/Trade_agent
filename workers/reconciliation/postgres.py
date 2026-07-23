"""PostgreSQL authority for paper portfolio reconciliation."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
import hashlib
import json
from typing import Any

from packages.risk import PortfolioState
from workers.reconciliation.projection import FillRow


class PostgresReconciliationRepository:
    def __init__(self, pool: Any) -> None:
        self._pool = pool

    async def load_fills(self, account_id: str) -> tuple[FillRow, ...]:
        async with self._pool.acquire() as connection:
            rows = await connection.fetch(
                """
                SELECT f.id, oi.instrument_id, oi.side, f.quantity, f.price,
                       f.fee, COALESCE(f.exchange_ts, f.received_ts) AS occurred_at
                FROM fills f
                JOIN broker_orders bo ON bo.id = f.order_id
                JOIN order_intents oi ON oi.id = bo.order_intent_id
                WHERE oi.account_id = $1
                  AND oi.source_mode = 'paper_live'
                ORDER BY COALESCE(f.exchange_ts, f.received_ts), f.id
                """,
                account_id,
            )
        return tuple(
            FillRow(
                fill_id=str(row["id"]),
                instrument=str(row["instrument_id"]),
                side=str(row["side"]),  # type: ignore[arg-type]
                quantity=Decimal(str(row["quantity"])),
                price=Decimal(str(row["price"])),
                fee=Decimal(str(row["fee"])),
                occurred_at=row["occurred_at"],
            )
            for row in rows
        )

    async def is_manual_kill_switch_active(self, account_id: str) -> bool:
        async with self._pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                SELECT active, changed_by FROM kill_switch_state
                WHERE account_id = $1
                """,
                account_id,
            )
        return bool(
            row is not None
            and row["active"]
            and row["changed_by"] not in {"system", "reconciliation-worker"}
        )

    async def persist_success(self, state: PortfolioState) -> None:
        payload = state.model_dump(mode="json")
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        checksum = hashlib.sha256(encoded.encode()).hexdigest()
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    "SELECT pg_advisory_xact_lock(hashtext($1))",
                    f"portfolio:{state.account_id}",
                )
                sequence = await connection.fetchval(
                    """
                    SELECT COALESCE(MAX(source_sequence), 0) + 1
                    FROM portfolio_snapshots WHERE account_id = $1
                    """,
                    state.account_id,
                )
                await connection.execute(
                    """
                    INSERT INTO portfolio_snapshots (
                        account_id, state_id, source_sequence, equity, cash,
                        daily_pnl, weekly_pnl, consecutive_losses,
                        open_positions, gross_exposure, kill_switch_active,
                        reconciled_at, payload, payload_checksum
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                        $11, $12, $13::jsonb, $14
                    ) ON CONFLICT (account_id, state_id) DO NOTHING
                    """,
                    state.account_id,
                    state.state_id,
                    sequence,
                    state.equity,
                    state.cash,
                    state.daily_pnl,
                    state.weekly_pnl,
                    state.consecutive_losses,
                    state.open_positions,
                    state.gross_exposure,
                    state.kill_switch_active,
                    state.reconciled_at,
                    encoded,
                    checksum,
                )
                await connection.execute(
                    """
                    INSERT INTO kill_switch_state (
                        account_id, active, reason, changed_by, changed_at, version
                    ) VALUES ($1, $2, 'reconciliation_clean',
                              'reconciliation-worker', $3, 1)
                    ON CONFLICT (account_id) DO UPDATE SET
                        active = EXCLUDED.active,
                        reason = CASE WHEN EXCLUDED.active
                            THEN kill_switch_state.reason
                            ELSE 'reconciliation_clean' END,
                        changed_by = CASE WHEN EXCLUDED.active
                            THEN kill_switch_state.changed_by
                            ELSE 'reconciliation-worker' END,
                        changed_at = EXCLUDED.changed_at,
                        version = kill_switch_state.version + 1
                    """,
                    state.account_id,
                    state.kill_switch_active,
                    state.reconciled_at,
                )
                await connection.execute(
                    """
                    INSERT INTO reconciliation_runs (
                        account_id, venue, source_mode, completed_at,
                        status, discrepancy_count, report
                    ) VALUES ($1, 'binance', 'paper_live', $2,
                              'matched', 0, $3::jsonb)
                    """,
                    state.account_id,
                    state.reconciled_at,
                    json.dumps({"state_id": state.state_id}),
                )

    async def persist_failure(self, *, account_id: str, reason: str) -> None:
        now = datetime.now(UTC)
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    """
                    INSERT INTO kill_switch_state (
                        account_id, active, reason, changed_by, changed_at, version
                    ) VALUES ($1, TRUE, $2, 'reconciliation-worker', $3, 1)
                    ON CONFLICT (account_id) DO UPDATE SET
                        active = TRUE,
                        reason = EXCLUDED.reason,
                        changed_by = 'reconciliation-worker',
                        changed_at = EXCLUDED.changed_at,
                        version = kill_switch_state.version + 1
                    """,
                    account_id,
                    reason,
                    now,
                )
                await connection.execute(
                    """
                    INSERT INTO reconciliation_runs (
                        account_id, venue, source_mode, completed_at,
                        status, discrepancy_count, report
                    ) VALUES ($1, 'binance', 'paper_live', $2,
                              'failed', 1, $3::jsonb)
                    """,
                    account_id,
                    now,
                    json.dumps({"reason": reason}),
                )
