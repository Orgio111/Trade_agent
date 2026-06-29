"""Brain #12: Odoo ERP Business Intelligence Brain.

Polls Odoo ERP via XML-RPC for business state changes (inventory, sales,
purchase orders, revenue, CRM pipeline) and converts them into trading
signals. Treats Odoo as a "real-world business state engine" — ERP
changes become market signals.

Weight in Go orchestrator: 0.05.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from datetime import datetime, timedelta
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from .base_brain import BaseBrain, BrainSignal

logger = logging.getLogger(__name__)


class OdooBrain(BaseBrain):
    """Odoo ERP business intelligence brain.

    Monitors:
      - Inventory changes (supply pressure)
      - Sales order velocity (demand strength)
      - Purchase order surges (supply chain activity)
      - Revenue trends (business health)
      - CRM pipeline conversion (growth trajectory)

    Falls back to neutral if Odoo is not configured or unreachable.

    Environment variables:
      ODOO_URL       — Odoo server URL (e.g. http://localhost:8069)
      ODOO_DB        — Database name
      ODOO_USER      — XML-RPC username (email)
      ODOO_PASSWORD  — XML-RPC password or API key
      ODOO_COMPANY_ID — Company ID (default: 1)
    """

    @property
    def brain_id(self) -> str:
        return "odoo_erp"

    def __init__(self) -> None:
        self._odoo_url = os.getenv("ODOO_URL", "")
        self._odoo_db = os.getenv("ODOO_DB", "")
        self._odoo_user = os.getenv("ODOO_USER", "")
        self._odoo_password = os.getenv("ODOO_PASSWORD", "")
        self._odoo_company_id = int(os.getenv("ODOO_COMPANY_ID", "1"))

        # Buffered business signals
        self._inventory_changes: deque[dict] = deque(maxlen=100)
        self._sales_orders: deque[dict] = deque(maxlen=100)
        self._purchase_orders: deque[dict] = deque(maxlen=100)
        self._revenue_snapshots: deque[dict] = deque(maxlen=50)
        self._crm_pipeline: deque[dict] = deque(maxlen=50)

        # Connection state
        self._connected = False
        self._uid: int | None = None
        self._last_poll_ts: float = 0.0

    @property
    def _configured(self) -> bool:
        return bool(self._odoo_url and self._odoo_db and self._odoo_user and self._odoo_password)

    async def warmup(self) -> None:
        if not self._configured:
            logger.warning(
                "[odoo_erp] Not configured — set ODOO_URL, ODOO_DB, ODOO_USER, ODOO_PASSWORD"
            )
            return

        try:
            await self._connect()
            logger.info(f"[odoo_erp] Connected to Odoo at {self._odoo_url}")
        except Exception as e:
            logger.warning(f"[odoo_erp] Connection failed: {e}")

    async def _connect(self) -> None:
        """Establish XML-RPC connection to Odoo."""
        import xmlrpc.client

        common = xmlrpc.client.ServerProxy(f"{self._odoo_url}/xmlrpc/2/common")
        self._uid = common.authenticate(self._odoo_db, self._odoo_user, self._odoo_password, {})
        if not self._uid:
            raise ConnectionError("Odoo authentication failed — check credentials")
        self._connected = True

    async def _rpc(self, model: str, method: str, args: list | None = None, kwargs: dict | None = None) -> Any:
        """Execute an XML-RPC call against Odoo."""
        import xmlrpc.client

        if not self._connected or not self._uid:
            await self._connect()

        models = xmlrpc.client.ServerProxy(f"{self._odoo_url}/xmlrpc/2/object")
        func = getattr(models, method)
        return func(self._odoo_db, self._uid, self._odoo_password, model, args or [], kwargs or {})

    async def compute_score(self, symbol: str) -> BrainSignal:
        if not self._configured or not self._connected:
            return BrainSignal(
                brain_id=self.brain_id,
                symbol=symbol,
                score=0.0,
                confidence=0.1,
                metadata={"configured": False, "reason": "ODOO env vars not set"},
            )

        # Poll Odoo for latest data
        try:
            await self._poll_odoo_data()
        except Exception as e:
            logger.warning(f"[odoo_erp] Poll failed: {e}")
            return BrainSignal(
                brain_id=self.brain_id,
                symbol=symbol,
                score=0.0,
                confidence=0.1,
                metadata={"error": str(e)},
            )

        # Compute sub-scores
        inventory_score = self._inventory_score()
        sales_score = self._sales_score()
        purchase_score = self._purchase_score()
        revenue_score = self._revenue_score()
        crm_score = self._crm_score()

        # Weighted combination
        combined = (
            0.25 * inventory_score
            + 0.30 * sales_score
            + 0.15 * purchase_score
            + 0.20 * revenue_score
            + 0.10 * crm_score
        )

        # Confidence based on data availability
        data_points = (
            len(self._inventory_changes)
            + len(self._sales_orders)
            + len(self._purchase_orders)
            + len(self._revenue_snapshots)
            + len(self._crm_pipeline)
        )
        confidence = min(0.85, 0.2 + data_points * 0.005)

        return BrainSignal(
            brain_id=self.brain_id,
            symbol=symbol,
            score=float(np.clip(combined, -1.0, 1.0)),
            confidence=confidence,
            metadata={
                "inventory": round(inventory_score, 4),
                "sales": round(sales_score, 4),
                "purchase": round(purchase_score, 4),
                "revenue": round(revenue_score, 4),
                "crm": round(crm_score, 4),
                "data_points": data_points,
                "connected": self._connected,
            },
        )

    async def _poll_odoo_data(self) -> None:
        """Fetch latest business data from Odoo."""
        try:
            # 1. Recent stock moves (inventory changes)
            stock_moves = await self._rpc(
                "stock.move", "search_read",
                [[["date", ">=", ts_24h]]],
                {"fields": ["product_id", "product_uom_qty", "location_dest_id", "state"], "limit": 50},
            )
            for move in stock_moves:
                self._inventory_changes.append({
                    "qty": move.get("product_uom_qty", 0),
                    "state": move.get("state", "draft"),
                })

            # 2. Recent sale orders
            sale_orders = await self._rpc(
                "sale.order", "search_read",
                [[[("date_order", ">=", ts_24h)]]],
                {"fields": ["amount_total", "state", "partner_id"], "limit": 50},
            )
            for order in sale_orders:
                self._sales_orders.append({
                    "amount": order.get("amount_total", 0),
                    "state": order.get("state", "draft"),
                })

            # 3. Recent purchase orders
            purchase_orders = await self._rpc(
                "purchase.order", "search_read",
                [[[("date_order", ">=", ts_24h)]]],
                {"fields": ["amount_total", "state"], "limit": 50},
            )
            for order in purchase_orders:
                self._purchase_orders.append({
                    "amount": order.get("amount_total", 0),
                    "state": order.get("state", "draft"),
                })

            # 4. Account move totals (revenue proxy)
            account_moves = await self._rpc(
                "account.move", "search_read",
                [[["move_type", "=", "out_invoice"], ["invoice_date", ">=", ts_7d]]],
                {"fields": ["amount_total", "invoice_date"], "limit": 30},
            )
            total_revenue = sum(m.get("amount_total", 0) for m in account_moves)
            self._revenue_snapshots.append({"revenue_7d": total_revenue, "count": len(account_moves)})

            # 5. CRM leads / opportunities
            crm_leads = await self._rpc(
                "crm.lead", "search_read",
                [[[("create_date", ">=", ts_24h)]]],
                {"fields": ["expected_revenue", "stage_id", "probability"], "limit": 50},
            )
            for lead in crm_leads:
                self._crm_pipeline.append({
                    "expected_revenue": lead.get("expected_revenue", 0),
                    "probability": lead.get("probability", 0),
                })

        except Exception as e:
            logger.debug(f"[odoo_erp] RPC partial failure: {e}")

    def _inventory_score(self) -> float:
        """Inventory increase sharply → weak demand → bearish.
        Inventory decrease → strong demand → bullish."""
        if not self._inventory_changes:
            return 0.0
        recent = list(self._inventory_changes)[-20:]
        total_qty = sum(m["qty"] for m in recent)
        # High inventory buildup = bearish (supply > demand)
        return float(np.tanh(-total_qty / 1000.0)) * 0.5

    def _sales_score(self) -> float:
        """Sales spike → demand strength ↑ → bullish macro signal."""
        if not self._sales_orders:
            return 0.0
        recent = list(self._sales_orders)[-20:]
        confirmed = [o for o in recent if o["state"] in ("sale", "done")]
        total = sum(o["amount"] for o in confirmed)
        # High sales = bullish
        return float(np.tanh(total / 50000.0)) * 0.5

    def _purchase_score(self) -> float:
        """Purchase order surge → supply chain activity ↑ → sector signal.
        High purchases can mean expansion (bullish) or cost pressure (neutral)."""
        if not self._purchase_orders:
            return 0.0
        recent = list(self._purchase_orders)[-20:]
        total = sum(o["amount"] for o in recent)
        # Moderate signal — purchases are ambiguous
        return float(np.tanh(total / 100000.0)) * 0.3

    def _revenue_score(self) -> float:
        """Revenue drop → risk-off mode → reduce exposure.
        Revenue growth → bullish."""
        if not self._revenue_snapshots:
            return 0.0
        latest = self._revenue_snapshots[-1]
        revenue = latest["revenue_7d"]
        # Compare with historical average if available
        if len(self._revenue_snapshots) > 3:
            avg = np.mean([s["revenue_7d"] for s in list(self._revenue_snapshots)[:-1]])
            if avg > 0:
                growth = (revenue - avg) / avg
                return float(np.tanh(growth * 5.0)) * 0.5
        # Absolute signal: high revenue = slightly bullish
        return float(np.tanh(revenue / 100000.0)) * 0.3

    def _crm_score(self) -> float:
        """CRM pipeline up → growth trajectory ↑ → bullish."""
        if not self._crm_pipeline:
            return 0.0
        recent = list(self._crm_pipeline)[-20:]
        weighted_rev = sum(
            l["expected_revenue"] * l["probability"] / 100.0 for l in recent
        )
        return float(np.tanh(weighted_rev / 50000.0)) * 0.4

    async def cooldown(self) -> None:
        self._connected = False
        self._uid = None
        logger.info("[odoo_erp] Disconnected")
