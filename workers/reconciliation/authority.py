"""Authoritative public Binance inputs for canonical paper risk."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from typing import Any, Mapping

import httpx

from packages.risk import InstrumentConstraints, RiskPolicy


_SYMBOLS = ("BTCUSDT", "ETHUSDT")


def constraints_from_exchange_info(
    payload: Mapping[str, Any], *, expected_symbol: str
) -> InstrumentConstraints:
    if expected_symbol not in _SYMBOLS:
        raise ValueError("symbol is outside canonical allowlist")
    if (
        payload.get("symbol") != expected_symbol
        or payload.get("status") != "TRADING"
        or payload.get("isSpotTradingAllowed") is not True
    ):
        raise ValueError("Binance symbol is not authoritative active spot")
    filters = payload.get("filters")
    if not isinstance(filters, list):
        raise ValueError("Binance symbol filters are missing")
    by_type = {
        item.get("filterType"): item for item in filters if isinstance(item, Mapping)
    }
    price = by_type.get("PRICE_FILTER")
    lot = by_type.get("LOT_SIZE")
    notional = by_type.get("NOTIONAL") or by_type.get("MIN_NOTIONAL")
    if (
        not isinstance(price, Mapping)
        or not isinstance(lot, Mapping)
        or not isinstance(notional, Mapping)
    ):
        raise ValueError("required Binance filters are missing")
    return InstrumentConstraints(
        venue="binance",
        market_type="spot",
        instrument=expected_symbol,
        tick_size=price["tickSize"],
        step_size=lot["stepSize"],
        min_quantity=lot["minQty"],
        min_notional=notional["minNotional"],
    )


class RiskAuthorityBootstrap:
    def __init__(self, pool: Any, *, client: httpx.AsyncClient | None = None) -> None:
        self._pool = pool
        self._client = client or httpx.AsyncClient(
            base_url="https://api.binance.com",
            timeout=httpx.Timeout(5),
            trust_env=False,
        )
        self._owns_client = client is None

    async def ensure(self) -> None:
        policy = RiskPolicy(version="paper-v1")
        policy_json = json.dumps(
            policy.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        policy_hash = hashlib.sha256(policy_json.encode()).hexdigest()
        constraints: list[tuple[InstrumentConstraints, str, str, str]] = []
        for symbol in _SYMBOLS:
            response = await self._client.get("/api/v3/exchangeInfo", params={"symbol": symbol})
            response.raise_for_status()
            body = response.json()
            symbols = body.get("symbols")
            if not isinstance(symbols, list) or len(symbols) != 1:
                raise ValueError("Binance exchange info is ambiguous")
            model = constraints_from_exchange_info(symbols[0], expected_symbol=symbol)
            raw = json.dumps(symbols[0], sort_keys=True, separators=(",", ":"))
            payload = json.dumps(
                model.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            )
            constraints.append(
                (
                    model,
                    payload,
                    hashlib.sha256(raw.encode()).hexdigest(),
                    hashlib.sha256(payload.encode()).hexdigest(),
                )
            )

        now = datetime.now(UTC)
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                active = await connection.fetchrow(
                    """
                    SELECT version, policy_hash FROM risk_policies
                    WHERE active = TRUE
                    """
                )
                if active is not None and (
                    active["version"] != policy.version
                    or active["policy_hash"] != policy_hash
                ):
                    raise ValueError("active risk policy differs from canonical paper policy")
                await connection.execute(
                    """
                    INSERT INTO risk_policies (
                        version, paper_only, policy, policy_hash, active
                    ) VALUES ($1, TRUE, $2::jsonb, $3, TRUE)
                    ON CONFLICT (version) DO NOTHING
                    """,
                    policy.version,
                    policy_json,
                    policy_hash,
                )
                await connection.execute(
                    """
                    UPDATE risk_policies SET active = TRUE
                    WHERE version = $1 AND policy_hash = $2
                      AND NOT EXISTS (
                          SELECT 1 FROM risk_policies
                          WHERE active = TRUE AND version <> $1
                      )
                    """,
                    policy.version,
                    policy_hash,
                )
                for model, payload, source_checksum, payload_checksum in constraints:
                    await connection.execute(
                        """
                        INSERT INTO instrument_constraints (
                            venue, market_type, instrument_id, version,
                            tick_size, step_size, min_quantity, min_notional,
                            effective_at, payload, payload_checksum
                        ) VALUES (
                            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, $11
                        ) ON CONFLICT (venue, market_type, instrument_id, version)
                          DO NOTHING
                        """,
                        model.venue,
                        model.market_type,
                        model.instrument,
                        f"binance-{source_checksum[:16]}",
                        model.tick_size,
                        model.step_size,
                        model.min_quantity,
                        model.min_notional,
                        now,
                        payload,
                        payload_checksum,
                    )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
