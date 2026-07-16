"""Read-only PostgreSQL repository for deterministic risk inputs."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
import json
from typing import Any, Mapping, Protocol

from packages.risk.policy import InstrumentConstraints, RiskPolicy
from packages.risk.state import PortfolioState


class RiskInputError(RuntimeError):
    """Base class for fail-closed risk-input lookup errors."""


class RiskInputUnavailable(RiskInputError):
    """PostgreSQL could not authoritatively answer the lookup."""


class RiskInputMissing(RiskInputError):
    """A required versioned risk input does not exist."""


class RiskInputAmbiguous(RiskInputError):
    """More than one row could authoritatively satisfy the lookup."""


class RiskInputStale(RiskInputError):
    """The authoritative row is outside its usable time window."""


@dataclass(frozen=True, slots=True)
class VersionedInstrumentConstraints:
    version: str
    constraints: InstrumentConstraints
    effective_at: datetime
    expires_at: datetime | None


class _Connection(Protocol):
    async def fetch(self, query: str, *args: Any) -> Sequence[Mapping[str, Any]]: ...


class _Pool(Protocol):
    def acquire(self) -> AbstractAsyncContextManager[_Connection]: ...


class PostgresRiskInputRepository:
    """Load exact, immutable inputs; never synthesize policy or portfolio state."""

    def __init__(self, pool: _Pool) -> None:
        self._pool = pool

    async def get_active_policy(self) -> RiskPolicy:
        rows = await self._fetch(
            """
            SELECT version, policy
            FROM risk_policies
            WHERE active = TRUE
            ORDER BY created_at DESC
            LIMIT 2
            """
        )
        if not rows:
            raise RiskInputMissing("no active risk policy exists")
        if len(rows) != 1:
            raise RiskInputAmbiguous("multiple active risk policies exist")
        try:
            policy = RiskPolicy.model_validate(
                self._json_object(rows[0]["policy"], field_name="risk_policy.policy")
            )
        except RiskInputError:
            raise
        except Exception as exc:
            raise RiskInputUnavailable("active risk policy payload is invalid") from exc
        if policy.version != str(rows[0]["version"]):
            raise RiskInputAmbiguous(
                "risk policy payload version differs from its database key"
            )
        return policy

    async def get_portfolio_state(
        self,
        account_id: str,
        *,
        as_of: datetime,
        max_age_seconds: Decimal | int | str,
        state_id: str | None = None,
    ) -> PortfolioState:
        account = account_id.strip()
        if not account:
            raise RiskInputMissing("account_id cannot be blank")
        now = self._aware_utc(as_of, field_name="as_of")
        max_age = self._decimal(max_age_seconds, field_name="max_age_seconds")
        if max_age < 0:
            raise RiskInputStale("max_age_seconds cannot be negative")

        if state_id is None:
            rows = await self._fetch(
                """
                SELECT state_id, source_sequence, reconciled_at, payload
                FROM portfolio_snapshots
                WHERE account_id = $1
                ORDER BY source_sequence DESC
                LIMIT 2
                """,
                account,
            )
            if len(rows) > 1 and int(rows[0]["source_sequence"]) == int(
                rows[1]["source_sequence"]
            ):
                raise RiskInputAmbiguous(
                    "latest portfolio snapshot sequence is ambiguous"
                )
        else:
            resolved_state_id = state_id.strip()
            if not resolved_state_id:
                raise RiskInputMissing("state_id cannot be blank")
            rows = await self._fetch(
                """
                SELECT state_id, source_sequence, reconciled_at, payload
                FROM portfolio_snapshots
                WHERE account_id = $1 AND state_id = $2
                LIMIT 2
                """,
                account,
                resolved_state_id,
            )

        if not rows:
            raise RiskInputMissing("authoritative portfolio snapshot is missing")
        if state_id is not None and len(rows) != 1:
            raise RiskInputAmbiguous("portfolio snapshot identity is ambiguous")
        try:
            state = PortfolioState.model_validate(
                self._json_object(
                    rows[0]["payload"], field_name="portfolio_snapshot.payload"
                )
            )
        except RiskInputError:
            raise
        except Exception as exc:
            raise RiskInputUnavailable("portfolio snapshot payload is invalid") from exc
        if state.account_id != account or state.state_id != str(rows[0]["state_id"]):
            raise RiskInputAmbiguous(
                "portfolio snapshot payload identity differs from database key"
            )
        if state.reconciled_at is None:
            raise RiskInputStale("portfolio snapshot has not been reconciled")
        age = self._decimal(
            (now - state.reconciled_at).total_seconds(), field_name="snapshot_age"
        )
        if age < 0:
            raise RiskInputStale("portfolio snapshot is from the future")
        if age > max_age:
            raise RiskInputStale("portfolio snapshot exceeds max_age_seconds")
        return state

    async def get_instrument_constraints(
        self,
        *,
        venue: str,
        market_type: str,
        instrument: str,
        as_of: datetime,
        version: str | None = None,
    ) -> VersionedInstrumentConstraints:
        resolved_venue = venue.strip().lower()
        resolved_market = market_type.strip().lower()
        resolved_instrument = instrument.strip().upper()
        if not resolved_venue or not resolved_market or not resolved_instrument:
            raise RiskInputMissing("instrument identity cannot be blank")
        effective_at = self._aware_utc(as_of, field_name="as_of")

        if version is None:
            rows = await self._fetch(
                """
                SELECT version, effective_at, expires_at, payload
                FROM instrument_constraints
                WHERE venue = $1
                  AND market_type = $2
                  AND instrument_id = $3
                  AND effective_at <= $4
                  AND (expires_at IS NULL OR expires_at > $4)
                ORDER BY effective_at DESC, created_at DESC
                LIMIT 2
                """,
                resolved_venue,
                resolved_market,
                resolved_instrument,
                effective_at,
            )
            if len(rows) > 1 and rows[0]["effective_at"] == rows[1]["effective_at"]:
                raise RiskInputAmbiguous(
                    "effective instrument-constraint version is ambiguous"
                )
        else:
            resolved_version = version.strip()
            if not resolved_version:
                raise RiskInputMissing("constraint version cannot be blank")
            rows = await self._fetch(
                """
                SELECT version, effective_at, expires_at, payload
                FROM instrument_constraints
                WHERE venue = $1
                  AND market_type = $2
                  AND instrument_id = $3
                  AND version = $4
                LIMIT 2
                """,
                resolved_venue,
                resolved_market,
                resolved_instrument,
                resolved_version,
            )

        if not rows:
            raise RiskInputMissing("instrument constraints are missing")
        if version is not None and len(rows) != 1:
            raise RiskInputAmbiguous("constraint version is ambiguous")
        row = rows[0]
        row_effective = self._aware_utc(row["effective_at"], field_name="effective_at")
        row_expires = (
            None
            if row["expires_at"] is None
            else self._aware_utc(row["expires_at"], field_name="expires_at")
        )
        if effective_at < row_effective or (
            row_expires is not None and effective_at >= row_expires
        ):
            raise RiskInputStale(
                "instrument-constraint version is not effective at as_of"
            )
        try:
            constraints = InstrumentConstraints.model_validate(
                self._json_object(
                    row["payload"], field_name="instrument_constraints.payload"
                )
            )
        except RiskInputError:
            raise
        except Exception as exc:
            raise RiskInputUnavailable(
                "instrument-constraint payload is invalid"
            ) from exc
        if (
            constraints.venue != resolved_venue
            or constraints.market_type != resolved_market
            or constraints.instrument != resolved_instrument
        ):
            raise RiskInputAmbiguous(
                "instrument-constraint payload identity differs from database key"
            )
        return VersionedInstrumentConstraints(
            version=str(row["version"]),
            constraints=constraints,
            effective_at=row_effective,
            expires_at=row_expires,
        )

    async def _fetch(self, query: str, *args: object) -> Sequence[Mapping[str, Any]]:
        try:
            async with self._pool.acquire() as connection:
                return await connection.fetch(query, *args)
        except RiskInputError:
            raise
        except Exception as exc:
            raise RiskInputUnavailable("risk-input query failed closed") from exc

    @staticmethod
    def _json_object(value: object, *, field_name: str) -> dict[str, Any]:
        parsed = json.loads(value) if isinstance(value, str) else value
        if not isinstance(parsed, Mapping):
            raise RiskInputUnavailable(f"{field_name} must be a JSON object")
        return {str(key): item for key, item in parsed.items()}

    @staticmethod
    def _aware_utc(value: object, *, field_name: str) -> datetime:
        if not isinstance(value, datetime):
            raise RiskInputUnavailable(f"{field_name} must be a datetime")
        if value.tzinfo is None or value.utcoffset() is None:
            raise RiskInputUnavailable(f"{field_name} must be timezone-aware")
        return value.astimezone(UTC)

    @staticmethod
    def _decimal(value: object, *, field_name: str) -> Decimal:
        if isinstance(value, bool):
            raise RiskInputUnavailable(f"{field_name} must be decimal-compatible")
        try:
            result = Decimal(str(value))
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise RiskInputUnavailable(
                f"{field_name} must be decimal-compatible"
            ) from exc
        if not result.is_finite():
            raise RiskInputUnavailable(f"{field_name} must be finite")
        return result
