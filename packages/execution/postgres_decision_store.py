"""PostgreSQL authorization store for immutable risk decisions."""

from __future__ import annotations

import json
from uuid import UUID

from packages.execution._postgres import PostgresConnection, PostgresPool, json_object
from packages.execution.decision_store import DecisionAuthorizationError
from packages.risk import CandidateSignal, PortfolioState, RiskDecision


class DecisionStoreUnavailable(DecisionAuthorizationError):
    """The durable authority cannot be verified, so execution must stop."""


class PostgresDecisionStore:
    """Async append-only implementation of the risk-decision store port.

    ``decision_payload`` is the authoritative representation.  Indexed columns
    support audit queries, but approval is granted only after the complete
    payload round-trips to the exact immutable :class:`RiskDecision`.
    """

    _INSERT = """
        INSERT INTO risk_decisions (
            id, signal_id, trace_id, account_id, policy_version,
            state_snapshot_id, approved, reason_codes, candidate_hash,
            approved_quantity, approved_risk_pct, state_snapshot,
            decision_payload, created_at
        )
        VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, $10, $11,
            $12::jsonb, $13::jsonb, $14
        )
        ON CONFLICT DO NOTHING
        RETURNING decision_payload, state_snapshot
    """
    _BY_ID = """
        SELECT decision_payload, state_snapshot
        FROM risk_decisions
        WHERE id = $1
    """
    _BY_SIGNAL = """
        SELECT
            rd.decision_payload,
            rd.state_snapshot,
            s.market_event_id,
            s.candidate_event_id
        FROM risk_decisions AS rd
        JOIN signals AS s ON s.id = rd.signal_id
        WHERE rd.signal_id = $1
        LIMIT 2
    """
    _BY_BINDING = """
        SELECT id
        FROM risk_decisions
        WHERE signal_id = $1
          AND policy_version = $2
          AND state_snapshot_id = $3
    """
    _INSERT_SIGNAL = """
        INSERT INTO signals (
            id, trace_id, strategy_version_id, instrument_id, side,
            source_mode, reference_price, stop_price, take_profit_price,
            confidence, feature_snapshot_id, market_event_id,
            candidate_event_id, payload, created_at
        )
        VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12,
            $13, $14::jsonb, $15
        )
        ON CONFLICT (id) DO NOTHING
        RETURNING payload, market_event_id, candidate_event_id,
                  strategy_version_id, feature_snapshot_id
    """
    _SIGNAL_BY_ID = """
        SELECT payload, market_event_id, candidate_event_id,
               strategy_version_id, feature_snapshot_id
        FROM signals
        WHERE id = $1
    """
    _SNAPSHOT_BY_ID = """
        SELECT payload
        FROM portfolio_snapshots
        WHERE account_id = $1 AND state_id = $2
    """

    def __init__(self, pool: PostgresPool) -> None:
        self._pool = pool

    async def record(self, decision: RiskDecision) -> RiskDecision:
        """Record a decision whose canonical signal row already exists.

        New decision workers should call :meth:`record_candidate_and_decision`
        so the signal FK and decision commit atomically.
        """

        try:
            async with self._pool.acquire() as connection:
                async with connection.transaction():
                    snapshot = await self._require_snapshot(connection, decision)
                    return await self._record_on(connection, decision, snapshot)
        except DecisionAuthorizationError:
            raise
        except Exception as exc:
            raise DecisionStoreUnavailable(
                "durable risk-decision record failed closed"
            ) from exc

    async def record_candidate_and_decision(
        self,
        candidate: CandidateSignal,
        decision: RiskDecision,
        *,
        portfolio_state: PortfolioState,
        market_event_id: str,
        candidate_event_id: str,
        strategy_version_id: UUID | None = None,
        feature_snapshot_id: str | None = None,
    ) -> RiskDecision:
        """Atomically persist the candidate signal and its exact risk decision."""

        if not market_event_id.strip():
            raise DecisionAuthorizationError("market_event_id cannot be blank")
        if not candidate_event_id.strip():
            raise DecisionAuthorizationError("candidate_event_id cannot be blank")
        if candidate.signal_id != decision.signal_id:
            raise DecisionAuthorizationError(
                "risk decision signal_id does not match candidate"
            )
        if candidate.trace_id != decision.trace_id:
            raise DecisionAuthorizationError(
                "risk decision trace_id does not match candidate"
            )
        if candidate.content_hash() != decision.candidate_hash:
            raise DecisionAuthorizationError(
                "risk decision candidate_hash does not match candidate"
            )
        if portfolio_state.account_id != decision.account_id:
            raise DecisionAuthorizationError(
                "risk decision account_id does not match portfolio snapshot"
            )
        if portfolio_state.state_id != decision.portfolio_state_id:
            raise DecisionAuthorizationError(
                "risk decision portfolio_state_id does not match snapshot"
            )

        payload = candidate.model_dump_json()
        try:
            async with self._pool.acquire() as connection:
                async with connection.transaction():
                    stored_snapshot = await self._require_snapshot(connection, decision)
                    if stored_snapshot != portfolio_state:
                        raise DecisionAuthorizationError(
                            "portfolio snapshot differs from authoritative record"
                        )
                    row = await connection.fetchrow(
                        self._INSERT_SIGNAL,
                        candidate.signal_id,
                        candidate.trace_id,
                        strategy_version_id,
                        candidate.instrument,
                        candidate.side,
                        candidate.source_mode.value,
                        candidate.reference_price,
                        candidate.stop_price,
                        candidate.take_profit_price,
                        candidate.confidence,
                        feature_snapshot_id,
                        market_event_id,
                        candidate_event_id,
                        payload,
                        decision.evaluated_at,
                    )
                    if row is None:
                        row = await connection.fetchrow(
                            self._SIGNAL_BY_ID, candidate.signal_id
                        )
                    if row is None:
                        raise DecisionAuthorizationError(
                            "candidate signal could not be persisted"
                        )
                    issued_candidate = CandidateSignal.model_validate(
                        json_object(row["payload"], field_name="signal.payload")
                    )
                    if issued_candidate != candidate:
                        raise DecisionAuthorizationError(
                            "signal_id already exists with different immutable content"
                        )
                    if str(row["market_event_id"]) != market_event_id:
                        raise DecisionAuthorizationError(
                            "signal_id is bound to a different market event"
                        )
                    if str(row["candidate_event_id"]) != candidate_event_id:
                        raise DecisionAuthorizationError(
                            "signal_id is bound to a different candidate event"
                        )
                    if row["strategy_version_id"] != strategy_version_id:
                        raise DecisionAuthorizationError(
                            "signal_id is bound to a different strategy version"
                        )
                    if row["feature_snapshot_id"] != feature_snapshot_id:
                        raise DecisionAuthorizationError(
                            "signal_id is bound to a different feature snapshot"
                        )
                    return await self._record_on(connection, decision, stored_snapshot)
        except DecisionAuthorizationError:
            raise
        except Exception as exc:
            raise DecisionStoreUnavailable(
                "candidate and risk-decision transaction failed closed"
            ) from exc

    async def _record_on(
        self,
        connection: PostgresConnection,
        decision: RiskDecision,
        portfolio_state: PortfolioState,
    ) -> RiskDecision:
        payload = decision.model_dump_json()
        reasons = json.dumps([reason.value for reason in decision.reason_codes])
        state_payload = portfolio_state.model_dump_json()
        try:
            row = await connection.fetchrow(
                self._INSERT,
                decision.decision_id,
                decision.signal_id,
                decision.trace_id,
                decision.account_id,
                decision.policy_version,
                decision.portfolio_state_id,
                decision.approved,
                reasons,
                decision.candidate_hash,
                decision.approved_quantity,
                decision.approved_risk_fraction,
                state_payload,
                payload,
                decision.evaluated_at,
            )
            if row is None:
                row = await connection.fetchrow(self._BY_ID, decision.decision_id)
                if row is None:
                    conflict = await connection.fetchrow(
                        self._BY_BINDING,
                        decision.signal_id,
                        decision.policy_version,
                        decision.portfolio_state_id,
                    )
                    conflict_id = (
                        str(conflict["id"]) if conflict is not None else "unknown"
                    )
                    raise DecisionAuthorizationError(
                        "risk-decision binding already belongs to "
                        f"a different decision ({conflict_id})"
                    )
            issued = self._decision_from_row(row)
            if issued != decision:
                raise DecisionAuthorizationError(
                    "decision_id already exists with different immutable content"
                )
            issued_snapshot = self._snapshot_from_row(row)
            if issued_snapshot != portfolio_state:
                raise DecisionAuthorizationError(
                    "decision_id is bound to a different portfolio snapshot"
                )
            return issued
        except DecisionAuthorizationError:
            raise
        except Exception as exc:
            raise DecisionStoreUnavailable(
                "durable risk-decision write failed closed"
            ) from exc

    async def _require_snapshot(
        self,
        connection: PostgresConnection,
        decision: RiskDecision,
    ) -> PortfolioState:
        row = await connection.fetchrow(
            self._SNAPSHOT_BY_ID,
            decision.account_id,
            decision.portfolio_state_id,
        )
        if row is None:
            raise DecisionAuthorizationError(
                "authoritative portfolio snapshot is missing"
            )
        try:
            snapshot = PortfolioState.model_validate(
                json_object(row["payload"], field_name="portfolio_snapshot.payload")
            )
        except Exception as exc:
            raise DecisionStoreUnavailable(
                "authoritative portfolio snapshot payload is invalid"
            ) from exc
        if (
            snapshot.account_id != decision.account_id
            or snapshot.state_id != decision.portfolio_state_id
        ):
            raise DecisionAuthorizationError(
                "portfolio snapshot identity differs from risk decision"
            )
        return snapshot

    async def require_exact_approval(self, decision: RiskDecision) -> RiskDecision:
        issued = await self.get(decision.decision_id)
        if issued is None:
            raise DecisionAuthorizationError(
                "risk decision was not issued by the configured authority"
            )
        if issued != decision:
            raise DecisionAuthorizationError(
                "risk decision content differs from the issued record"
            )
        if not issued.approved:
            raise DecisionAuthorizationError("issued risk decision is not approved")
        return issued

    async def get(self, decision_id: str) -> RiskDecision | None:
        if not decision_id.strip():
            raise DecisionAuthorizationError("decision_id cannot be blank")
        try:
            async with self._pool.acquire() as connection:
                row = await connection.fetchrow(self._BY_ID, decision_id)
            return None if row is None else self._decision_from_row(row)
        except DecisionAuthorizationError:
            raise
        except Exception as exc:
            raise DecisionStoreUnavailable(
                "durable risk-decision lookup failed closed"
            ) from exc

    async def get_for_signal(
        self,
        signal_id: str,
        *,
        candidate_hash: str,
        market_event_id: str,
        candidate_event_id: str,
    ) -> RiskDecision | None:
        """Return the single immutable verdict already issued for a signal.

        JetStream may redeliver an event after policy or portfolio state has
        advanced. Reusing the first persisted verdict prevents a retry from
        silently authorizing the same candidate against different inputs.
        """

        resolved_signal_id = signal_id.strip()
        if not resolved_signal_id:
            raise DecisionAuthorizationError("signal_id cannot be blank")
        if len(candidate_hash) != 64:
            raise DecisionAuthorizationError("candidate_hash must be a SHA-256 digest")
        if not market_event_id.strip() or not candidate_event_id.strip():
            raise DecisionAuthorizationError(
                "market_event_id and candidate_event_id cannot be blank"
            )
        try:
            async with self._pool.acquire() as connection:
                rows = await connection.fetch(
                    self._BY_SIGNAL,
                    resolved_signal_id,
                )
            if len(rows) > 1:
                raise DecisionAuthorizationError(
                    "more than one risk decision exists for the signal"
                )
            if not rows:
                return None
            issued = self._decision_from_row(rows[0])
            if issued.candidate_hash != candidate_hash:
                raise DecisionAuthorizationError(
                    "persisted signal verdict belongs to different candidate content"
                )
            if str(rows[0]["market_event_id"]) != market_event_id:
                raise DecisionAuthorizationError(
                    "persisted signal verdict belongs to a different market event"
                )
            if str(rows[0]["candidate_event_id"]) != candidate_event_id:
                raise DecisionAuthorizationError(
                    "persisted signal verdict belongs to a different candidate event"
                )
            return issued
        except DecisionAuthorizationError:
            raise
        except Exception as exc:
            raise DecisionStoreUnavailable(
                "durable risk-decision signal lookup failed closed"
            ) from exc

    async def count(self) -> int:
        try:
            async with self._pool.acquire() as connection:
                row = await connection.fetchrow(
                    "SELECT COUNT(*) AS count FROM risk_decisions"
                )
            if row is None:
                raise RuntimeError("count query returned no row")
            return int(row["count"])
        except Exception as exc:
            raise DecisionStoreUnavailable(
                "durable risk-decision count failed closed"
            ) from exc

    @staticmethod
    def _decision_from_row(row: object) -> RiskDecision:
        try:
            payload = row["decision_payload"]  # type: ignore[index]
            return RiskDecision.model_validate(
                json_object(payload, field_name="decision_payload")
            )
        except Exception as exc:
            raise DecisionStoreUnavailable(
                "stored risk-decision payload is invalid"
            ) from exc

    @staticmethod
    def _snapshot_from_row(row: object) -> PortfolioState:
        try:
            payload = row["state_snapshot"]  # type: ignore[index]
            return PortfolioState.model_validate(
                json_object(payload, field_name="risk_decision.state_snapshot")
            )
        except Exception as exc:
            raise DecisionStoreUnavailable(
                "stored risk-decision snapshot is invalid"
            ) from exc
