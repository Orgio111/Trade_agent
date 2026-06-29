"use client";

// ── Types ──────────────────────────────────────────────────────────────────

interface OdooSignal {
  demand_index: number;
  supply_pressure: number;
  cash_flow: number;
  business_health: number;
  combined_score: number;
  direction: "long" | "short" | "hold";
  confidence: number;
}

interface InventoryData {
  total_changes: number;
  net_delta: number;
  products_moved: number;
  recent_moves: { qty: number; state: string; timestamp?: string }[];
}

interface SalesData {
  total_revenue: number;
  order_count: number;
  avg_order_value: number;
  growth_rate: number;
  recent_orders: { amount: number; state: string }[];
}

interface RevenueData {
  daily: number;
  weekly: number;
  trend: "up" | "down" | "stable";
  change_pct: number;
}

interface CrmData {
  pipeline_value: number;
  conversion_rate: number;
  active_leads: number;
  closed_deals: number;
}

interface OdooErpData {
  connected: boolean;
  last_sync: string | null;
  signal: OdooSignal | null;
  inventory: InventoryData | null;
  sales: SalesData | null;
  revenue: RevenueData | null;
  crm: CrmData | null;
}

// ── Sub-components ──────────────────────────────────────────────────────────

function SignalCard({ signal }: { signal: OdooSignal }) {
  const dirColor =
    signal.direction === "long"
      ? "#00ff88"
      : signal.direction === "short"
        ? "#ff0044"
        : "#888";

  return (
    <div
      style={{
        background: "#0a0a0f",
        borderRadius: 10,
        padding: 16,
        border: `1px solid ${dirColor}33`,
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          marginBottom: 12,
        }}
      >
        <div
          style={{
            width: 36,
            height: 36,
            borderRadius: 8,
            background: `${dirColor}18`,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 18,
            color: dirColor,
          }}
        >
          🏢
        </div>
        <div>
          <div style={{ fontSize: 13, fontWeight: 600, color: "#ccc" }}>
            ERP Business Signal
          </div>
          <div style={{ fontSize: 10, color: "#555", marginTop: 2 }}>
            Odoo → Trading Intelligence
          </div>
        </div>
        <span
          className={`badge ${signal.direction === "long" ? "badge-green" : signal.direction === "short" ? "badge-red" : "badge-gray"}`}
          style={{ fontSize: 12, fontWeight: 700, padding: "6px 16px", marginLeft: "auto" }}
        >
          {signal.direction.toUpperCase()}
        </span>
      </div>

      {/* Score bars */}
      <div style={{ display: "grid", gap: 8 }}>
        {[
          { label: "Demand Index", value: signal.demand_index, color: "#00ff88" },
          { label: "Supply Pressure", value: signal.supply_pressure, color: "#ff0044" },
          { label: "Cash Flow", value: signal.cash_flow, color: "#00aaff" },
          { label: "Business Health", value: signal.business_health, color: "#aa00ff" },
        ].map((item) => (
          <div key={item.label}>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                fontSize: 10,
                color: "#888",
                marginBottom: 3,
              }}
            >
              <span>{item.label}</span>
              <span style={{ color: item.color, fontFamily: "var(--font-mono)" }}>
                {item.value >= 0 ? "+" : ""}{item.value.toFixed(3)}
              </span>
            </div>
            <div
              style={{
                height: 4,
                background: "#1a1a2e",
                borderRadius: 2,
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  width: `${Math.min(Math.abs(item.value) * 200, 100)}%`,
                  height: "100%",
                  background: item.value >= 0 ? item.color : "#ff0044",
                  borderRadius: 2,
                  transition: "width 0.5s ease",
                }}
              />
            </div>
          </div>
        ))}
      </div>

      {/* Combined score */}
      <div
        style={{
          marginTop: 12,
          paddingTop: 10,
          borderTop: "1px solid #1a1a2e",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          fontSize: 11,
        }}
      >
        <span style={{ color: "#888" }}>Combined Score</span>
        <span style={{ color: dirColor, fontWeight: 700, fontFamily: "var(--font-mono)" }}>
          {signal.combined_score >= 0 ? "+" : ""}{(signal.combined_score * 100).toFixed(1)}%
        </span>
      </div>
      <div style={{ marginTop: 4, fontSize: 10, color: "#555" }}>
        Confidence: {(signal.confidence * 100).toFixed(0)}%
      </div>
    </div>
  );
}

function MetricRow({ label, value, color = "#ccc", mono = true }: { label: string; value: string; color?: string; mono?: boolean }) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        padding: "5px 0",
        borderBottom: "1px solid #111",
        fontSize: 10,
      }}
    >
      <span style={{ color: "#888" }}>{label}</span>
      <span
        style={{
          color,
          fontWeight: 600,
          fontFamily: mono ? "var(--font-mono)" : "inherit",
        }}
      >
        {value}
      </span>
    </div>
  );
}

// ── Main Component ──────────────────────────────────────────────────────────

export default function OdooErpPanel({ data }: { data: OdooErpData | null }) {
  if (!data) {
    return (
      <div style={{ display: "grid", gap: 16, gridColumn: "1 / -1" }}>
        <div className="panel">
          <div className="panel-title">🏢 Odoo ERP → Trading Intelligence</div>
          <p style={{ color: "#444", fontSize: 12 }}>Waiting for Odoo ERP data...</p>
        </div>
      </div>
    );
  }

  const { connected, last_sync, signal, inventory, sales, revenue, crm } = data;

  return (
    <div style={{ display: "grid", gap: 16, gridColumn: "1 / -1" }}>
      {/* ═══ Connection Status Bar ═══ */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr 1fr 1fr",
          gap: 14,
          animation: "slideUp 0.3s ease",
        }}
      >
        <div className="panel stat-card">
          <div className="panel-title">Odoo Connection</div>
          <div
            className="stat-value"
            style={{ color: connected ? "#00ff88" : "#ff0044" }}
          >
            {connected ? "● ONLINE" : "○ OFFLINE"}
          </div>
          <div className="stat-subtitle">
            {last_sync
              ? `Last sync: ${new Date(last_sync).toLocaleTimeString()}`
              : "No sync yet"}
          </div>
        </div>
        <div className="panel stat-card">
          <div className="panel-title">Inventory Signal</div>
          <div
            className="stat-value"
            style={{ color: inventory?.net_delta !== undefined && inventory.net_delta < 0 ? "#ff0044" : "#00ff88" }}
          >
            {inventory?.net_delta !== undefined ? `${inventory.net_delta > 0 ? "+" : ""}${inventory.net_delta}` : "—"}
          </div>
          <div className="stat-subtitle">
            {inventory?.products_moved ?? 0} products moved
          </div>
        </div>
        <div className="panel stat-card">
          <div className="panel-title">Sales Velocity</div>
          <div
            className="stat-value"
            style={{ color: sales?.growth_rate !== undefined && sales.growth_rate > 0 ? "#00ff88" : "#ff0044" }}
          >
            {sales?.order_count ?? 0} orders
          </div>
          <div className="stat-subtitle">
            {sales?.growth_rate !== undefined ? `${sales.growth_rate > 0 ? "+" : ""}${(sales.growth_rate * 100).toFixed(1)}% growth` : "—"}
          </div>
        </div>
        <div className="panel stat-card">
          <div className="panel-title">Revenue Trend</div>
          <div
            className="stat-value"
            style={{
              color: revenue?.trend === "up" ? "#00ff88" : revenue?.trend === "down" ? "#ff0044" : "#888",
            }}
          >
            {revenue?.daily !== undefined ? `$${revenue.daily.toLocaleString()}` : "—"}
          </div>
          <div className="stat-subtitle">
            {revenue?.trend === "up" ? "▲" : revenue?.trend === "down" ? "▼" : "—"}{" "}
            {revenue?.change_pct !== undefined ? `${revenue.change_pct > 0 ? "+" : ""}${revenue.change_pct.toFixed(1)}%` : ""}
          </div>
        </div>
      </div>

      {/* ═══ Signal + Details Grid ═══ */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 14 }}>
        {/* Business Signal Card */}
        {signal ? (
          <SignalCard signal={signal} />
        ) : (
          <div className="panel" style={{ display: "flex", alignItems: "center", justifyContent: "center" }}>
            <p style={{ color: "#555", fontSize: 12 }}>No signal computed yet</p>
          </div>
        )}

        {/* Inventory & Sales Detail */}
        <div className="panel">
          <div className="panel-title">Inventory & Sales Detail</div>

          {/* Inventory */}
          <div style={{ marginBottom: 12 }}>
            <div
              style={{
                fontSize: 9,
                color: "#555",
                textTransform: "uppercase",
                letterSpacing: 1,
                marginBottom: 6,
              }}
            >
              Recent Inventory
            </div>
            <MetricRow label="Total Changes" value={`${inventory?.total_changes ?? 0}`} />
            <MetricRow
              label="Net Delta"
              value={`${inventory?.net_delta !== undefined && inventory.net_delta >= 0 ? "+" : ""}${inventory?.net_delta ?? 0}`}
              color={inventory?.net_delta !== undefined && inventory.net_delta < 0 ? "#ff0044" : "#00ff88"}
            />
            <MetricRow label="Products Moved" value={`${inventory?.products_moved ?? 0}`} />
          </div>

          {/* Sales */}
          <div>
            <div
              style={{
                fontSize: 9,
                color: "#555",
                textTransform: "uppercase",
                letterSpacing: 1,
                marginBottom: 6,
              }}
            >
              Sales Summary
            </div>
            <MetricRow
              label="Revenue"
              value={`$${sales?.total_revenue?.toLocaleString() ?? "0"}`}
              color="#00aaff"
            />
            <MetricRow
              label="Avg Order"
              value={`$${sales?.avg_order_value?.toFixed(2) ?? "0"}`}
            />
            <MetricRow
              label="Growth Rate"
              value={sales?.growth_rate !== undefined ? `${sales.growth_rate > 0 ? "+" : ""}${(sales.growth_rate * 100).toFixed(1)}%` : "—"}
              color={sales?.growth_rate !== undefined && sales.growth_rate > 0 ? "#00ff88" : "#ff0044"}
            />
          </div>
        </div>

        {/* Revenue & CRM */}
        <div className="panel">
          <div className="panel-title">Revenue & CRM</div>

          {/* Revenue */}
          <div style={{ marginBottom: 12 }}>
            <div
              style={{
                fontSize: 9,
                color: "#555",
                textTransform: "uppercase",
                letterSpacing: 1,
                marginBottom: 6,
              }}
            >
              Revenue Trend
            </div>
            <MetricRow
              label="Daily"
              value={`$${revenue?.daily?.toLocaleString() ?? "0"}`}
              color="#00aaff"
            />
            <MetricRow
              label="Weekly"
              value={`$${revenue?.weekly?.toLocaleString() ?? "0"}`}
              color="#00aaff"
            />
            <MetricRow
              label="Trend"
              value={revenue?.trend?.toUpperCase() ?? "—"}
              color={
                revenue?.trend === "up"
                  ? "#00ff88"
                  : revenue?.trend === "down"
                    ? "#ff0044"
                    : "#888"
              }
              mono={false}
            />
          </div>

          {/* CRM */}
          <div>
            <div
              style={{
                fontSize: 9,
                color: "#555",
                textTransform: "uppercase",
                letterSpacing: 1,
                marginBottom: 6,
              }}
            >
              CRM Pipeline
            </div>
            <MetricRow
              label="Pipeline Value"
              value={`$${crm?.pipeline_value?.toLocaleString() ?? "0"}`}
              color="#aa00ff"
            />
            <MetricRow
              label="Conversion"
              value={crm?.conversion_rate !== undefined ? `${(crm.conversion_rate * 100).toFixed(1)}%` : "—"}
              color={crm?.conversion_rate !== undefined && crm.conversion_rate > 0.3 ? "#00ff88" : "#ffaa00"}
            />
            <MetricRow label="Active Leads" value={`${crm?.active_leads ?? 0}`} />
            <MetricRow label="Closed Deals" value={`${crm?.closed_deals ?? 0}`} color="#00ff88" />
          </div>
        </div>
      </div>

      {/* ═══ Signal Mapping Reference ═══ */}
      <div className="panel" style={{ animation: "slideUp 0.4s ease" }}>
        <div className="panel-title">
          Odoo → Trading Signal Mapping
          <span style={{ marginLeft: 8, fontWeight: 400, color: "#555" }}>
            — How ERP data translates to market signals
          </span>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 12 }}>
          {[
            { odoo: "📦 Inventory ↑", trading: "Weak demand → bearish", color: "#ff0044" },
            { odoo: "📈 Sales ↑", trading: "Demand strength → bullish", color: "#00ff88" },
            { odoo: "💰 Revenue ↓", trading: "Risk-off → reduce exposure", color: "#ff0044" },
            { odoo: "🤝 CRM ↑", trading: "Pipeline strength → bullish", color: "#00ff88" },
          ].map((item) => (
            <div
              key={item.odoo}
              style={{
                background: "#0a0a0f",
                borderRadius: 8,
                padding: 12,
                borderLeft: `3px solid ${item.color}44`,
              }}
            >
              <div style={{ fontSize: 11, fontWeight: 600, color: "#ccc", marginBottom: 4 }}>
                {item.odoo}
              </div>
              <div style={{ fontSize: 10, color: "#888" }}>{item.trading}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
