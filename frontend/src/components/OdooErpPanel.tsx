"use client";

// ── Types matching the OdooErpData from page.tsx ────────────────────────────

interface OdooSignal {
  demand_index: number;
  supply_pressure: number;
  cash_flow: number;
  business_health: number;
  combined_score: number;
  direction: "long" | "short" | "hold";
  confidence: number;
}

interface OdooErpData {
  connected: boolean;
  last_sync: string | null;
  signal: OdooSignal | null;
  inventory: {
    total_changes: number;
    net_delta: number;
    products_moved: number;
    recent_moves: { qty: number; state: string }[];
  } | null;
  sales: {
    total_revenue: number;
    order_count: number;
    avg_order_value: number;
    growth_rate: number;
    recent_orders: { amount: number; state: string }[];
  } | null;
  revenue: {
    daily: number;
    weekly: number;
    trend: "up" | "down" | "stable";
    change_pct: number;
  } | null;
  crm: {
    pipeline_value: number;
    conversion_rate: number;
    active_leads: number;
    closed_deals: number;
  } | null;
}

// ── Helpers ─────────────────────────────────────────────────────────────────

function fmtPct(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

function fmtCurrency(v: number): string {
  if (v >= 1e6) return `$${(v / 1e6).toFixed(2)}M`;
  if (v >= 1e3) return `$${(v / 1e3).toFixed(1)}K`;
  return `$${v.toFixed(2)}`;
}

function trendColor(trend: "up" | "down" | "stable"): string {
  return trend === "up" ? "#00ff88" : trend === "down" ? "#ff0044" : "#888";
}

function trendIcon(trend: "up" | "down" | "stable"): string {
  return trend === "up" ? "▲" : trend === "down" ? "▼" : "—";
}

// ── Sub-component: Metric Card ──────────────────────────────────────────────

function MetricCard({
  label,
  value,
  sub,
  color,
  barPct,
}: {
  label: string;
  value: string;
  sub?: string;
  color: string;
  barPct?: number;
}) {
  return (
    <div style={{ background: "#0a0a0f", borderRadius: 8, padding: 12 }}>
      <div style={{ fontSize: 9, color: "#555", textTransform: "uppercase", letterSpacing: 1, marginBottom: 8 }}>
        {label}
      </div>
      <div style={{ fontSize: 20, fontWeight: 700, color, marginBottom: 4 }}>
        {value}
      </div>
      {barPct != null && (
        <div style={{ background: "#1a1a2e", borderRadius: 3, height: 4, overflow: "hidden", marginBottom: 6 }}>
          <div
            style={{
              width: `${Math.min(Math.max(barPct, 0), 100)}%`,
              height: "100%",
              background: color,
              borderRadius: 3,
              transition: "width 0.5s ease",
            }}
          />
        </div>
      )}
      {sub && <div style={{ fontSize: 10, color: "#555" }}>{sub}</div>}
    </div>
  );
}

// ── Main Component ──────────────────────────────────────────────────────────

export default function OdooErpPanel({ data }: { data: OdooErpData | null }) {
  if (!data) {
    return (
      <div className="panel" style={{ gridColumn: "1 / -1" }}>
        <div className="panel-title">Odoo ERP Signals</div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 12 }}>
          <MetricCard label="Demand Index" value="—" color="#888" />
          <MetricCard label="Supply Pressure" value="—" color="#888" />
          <MetricCard label="Cash Flow Trend" value="—" color="#888" />
          <MetricCard label="Business Health" value="—" color="#888" />
        </div>
        <p style={{ color: "#444", fontSize: 12, marginTop: 12, textAlign: "center" }}>
          Waiting for Odoo ERP data...
        </p>
      </div>
    );
  }

  const { connected, last_sync, signal, inventory, sales, revenue, crm } = data;

  // Signal-derived values
  const demandPct = signal ? (signal.demand_index + 1) / 2 * 100 : 50; // -1..1 → 0..100
  const supplyPct = signal ? (signal.supply_pressure + 1) / 2 * 100 : 50;
  const cashPct = signal ? (signal.cash_flow + 1) / 2 * 100 : 50;
  const healthPct = signal ? signal.business_health * 100 : 50;

  const demandColor = signal && signal.demand_index > 0.2 ? "#00ff88" : signal && signal.demand_index < -0.2 ? "#ff0044" : "#888";
  const supplyColor = signal && signal.supply_pressure > 0.2 ? "#ff0044" : signal && signal.supply_pressure < -0.2 ? "#00ff88" : "#888";
  const cashColor = signal && signal.cash_flow > 0.2 ? "#00ff88" : signal && signal.cash_flow < -0.2 ? "#ff0044" : "#888";
  const healthColor = signal && signal.business_health > 0.6 ? "#00ff88" : signal && signal.business_health > 0.3 ? "#ffaa00" : "#ff0044";

  return (
    <div style={{ display: "grid", gap: 16, gridColumn: "1 / -1" }}>
      {/* ═══ Connection Status ═══ */}
      <div style={{
        display: "flex", alignItems: "center", gap: 12,
        padding: "10px 16px", background: "#0a0a0f", borderRadius: 8,
        border: `1px solid ${connected ? "#00ff8833" : "#ff004433"}`,
      }}>
        <div style={{
          width: 10, height: 10, borderRadius: "50%",
          background: connected ? "#00ff88" : "#ff0044",
          animation: connected ? "none" : "pulse 1s ease-in-out infinite",
        }} />
        <span style={{ fontSize: 12, fontWeight: 600, color: connected ? "#00ff88" : "#ff0044" }}>
          {connected ? "CONNECTED" : "DISCONNECTED"}
        </span>
        {last_sync && (
          <span style={{ fontSize: 10, color: "#555", marginLeft: "auto" }}>
            Last sync: {new Date(last_sync).toLocaleTimeString()}
          </span>
        )}
        <span style={{ fontSize: 9, color: "#333", fontFamily: "var(--font-mono)" }}>
          Brain #12 · weight 0.05
        </span>
      </div>

      {/* ═══ 4 Signal Cards ═══ */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 12 }}>
        <MetricCard
          label="Demand Index"
          value={signal ? fmtPct(Math.abs(signal.demand_index)) : "—"}
          sub={sales ? `${sales.order_count} orders · ${fmtCurrency(sales.total_revenue)}` : undefined}
          color={demandColor}
          barPct={demandPct}
        />
        <MetricCard
          label="Supply Pressure"
          value={signal ? fmtPct(Math.abs(signal.supply_pressure)) : "—"}
          sub={inventory ? `${inventory.net_delta > 0 ? "+" : ""}${inventory.net_delta.toFixed(0)} units net` : undefined}
          color={supplyColor}
          barPct={supplyPct}
        />
        <MetricCard
          label="Cash Flow Trend"
          value={revenue ? `${trendIcon(revenue.trend)} ${(revenue.change_pct * 100).toFixed(1)}%` : "—"}
          sub={revenue ? `${fmtCurrency(revenue.daily)}/day · ${fmtCurrency(revenue.weekly)}/wk` : undefined}
          color={cashColor}
          barPct={cashPct}
        />
        <MetricCard
          label="Business Health"
          value={signal ? fmtPct(signal.business_health) : "—"}
          sub={crm ? `${crm.active_leads} leads · ${crm.closed_deals} closed` : undefined}
          color={healthColor}
          barPct={healthPct}
        />
      </div>

      {/* ═══ Combined Signal + CRM Pipeline ═══ */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        {/* Combined ERP Signal */}
        <div style={{ background: "#0a0a0f", borderRadius: 8, padding: 16 }}>
          <div style={{ fontSize: 9, color: "#555", textTransform: "uppercase", letterSpacing: 1, marginBottom: 12 }}>
            Combined ERP Signal
          </div>
          {signal ? (
            <>
              <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12 }}>
                <span style={{
                  fontSize: 14, fontWeight: 700, padding: "4px 12px", borderRadius: 6,
                  background: signal.direction === "long" ? "#00ff8822" : signal.direction === "short" ? "#ff004422" : "#88888822",
                  color: signal.direction === "long" ? "#00ff88" : signal.direction === "short" ? "#ff0044" : "#888",
                }}>
                  {signal.direction.toUpperCase()}
                </span>
                <span style={{ fontSize: 12, color: "#888" }}>
                  {(signal.confidence * 100).toFixed(0)}% confidence
                </span>
              </div>
              <div style={{
                fontSize: 28, fontWeight: 700, fontFamily: "var(--font-mono)",
                color: signal.combined_score > 0 ? "#00ff88" : signal.combined_score < 0 ? "#ff0044" : "#888",
                marginBottom: 4,
              }}>
                {signal.combined_score > 0 ? "+" : ""}{signal.combined_score.toFixed(4)}
              </div>
              <div style={{ fontSize: 10, color: "#555" }}>
                Composite of 4 ERP sub-signals
              </div>
              {/* Sub-signal breakdown */}
              <div style={{ marginTop: 12, display: "grid", gap: 6 }}>
                {[
                  { label: "Demand", value: signal.demand_index, weight: "25%" },
                  { label: "Supply", value: signal.supply_pressure, weight: "20%" },
                  { label: "Revenue", value: signal.cash_flow, weight: "30%" },
                  { label: "CRM", value: signal.business_health, weight: "25%" },
                ].map((s) => (
                  <div key={s.label} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 10 }}>
                    <span style={{ width: 60, color: "#555" }}>{s.label}</span>
                    <div style={{ flex: 1, height: 4, background: "#1a1a2e", borderRadius: 2, overflow: "hidden" }}>
                      <div style={{
                        width: `${Math.abs(s.value) * 50 + 50}%`,
                        height: "100%",
                        background: s.value > 0 ? "#00ff88" : s.value < 0 ? "#ff0044" : "#888",
                        borderRadius: 2,
                        marginLeft: s.value > 0 ? "50%" : undefined,
                        transition: "width 0.5s ease",
                      }} />
                    </div>
                    <span style={{ width: 30, color: "#555", fontFamily: "var(--font-mono)" }}>{s.weight}</span>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <p style={{ color: "#444", fontSize: 12 }}>No signal data yet</p>
          )}
        </div>

        {/* CRM Pipeline */}
        <div style={{ background: "#0a0a0f", borderRadius: 8, padding: 16 }}>
          <div style={{ fontSize: 9, color: "#555", textTransform: "uppercase", letterSpacing: 1, marginBottom: 12 }}>
            CRM Pipeline
          </div>
          {crm ? (
            <>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 16 }}>
                <div>
                  <div style={{ fontSize: 9, color: "#555" }}>Pipeline Value</div>
                  <div style={{ fontSize: 16, fontWeight: 700, color: "#00aaff" }}>
                    {fmtCurrency(crm.pipeline_value)}
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: 9, color: "#555" }}>Conversion Rate</div>
                  <div style={{ fontSize: 16, fontWeight: 700, color: crm.conversion_rate > 0.2 ? "#00ff88" : "#ffaa00" }}>
                    {fmtPct(crm.conversion_rate)}
                  </div>
                </div>
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                <div style={{ background: "#111", borderRadius: 6, padding: 10, textAlign: "center" }}>
                  <div style={{ fontSize: 20, fontWeight: 700, color: "#00aaff" }}>{crm.active_leads}</div>
                  <div style={{ fontSize: 9, color: "#555", marginTop: 2 }}>Active Leads</div>
                </div>
                <div style={{ background: "#111", borderRadius: 6, padding: 10, textAlign: "center" }}>
                  <div style={{ fontSize: 20, fontWeight: 700, color: "#00ff88" }}>{crm.closed_deals}</div>
                  <div style={{ fontSize: 9, color: "#555", marginTop: 2 }}>Closed Deals</div>
                </div>
              </div>
            </>
          ) : (
            <p style={{ color: "#444", fontSize: 12 }}>No CRM data yet</p>
          )}
        </div>
      </div>

      {/* ═══ Inventory + Sales Detail ═══ */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        {/* Inventory */}
        <div style={{ background: "#0a0a0f", borderRadius: 8, padding: 16 }}>
          <div style={{ fontSize: 9, color: "#555", textTransform: "uppercase", letterSpacing: 1, marginBottom: 12 }}>
            Inventory Activity
          </div>
          {inventory ? (
            <>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8, marginBottom: 12 }}>
                <div>
                  <div style={{ fontSize: 9, color: "#555" }}>Changes</div>
                  <div style={{ fontSize: 16, fontWeight: 700, color: "#00aaff" }}>{inventory.total_changes}</div>
                </div>
                <div>
                  <div style={{ fontSize: 9, color: "#555" }}>Net Delta</div>
                  <div style={{ fontSize: 16, fontWeight: 700, color: inventory.net_delta > 0 ? "#ff0044" : "#00ff88" }}>
                    {inventory.net_delta > 0 ? "+" : ""}{inventory.net_delta}
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: 9, color: "#555" }}>Products Moved</div>
                  <div style={{ fontSize: 16, fontWeight: 700, color: "#888" }}>{inventory.products_moved}</div>
                </div>
              </div>
              {inventory.recent_moves.length > 0 && (
                <div style={{ borderTop: "1px solid #1a1a2e", paddingTop: 8 }}>
                  <div style={{ fontSize: 9, color: "#444", marginBottom: 4 }}>Recent Moves</div>
                  {inventory.recent_moves.slice(0, 4).map((m, i) => (
                    <div key={i} style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#666", padding: "2px 0" }}>
                      <span>{m.state}</span>
                      <span style={{ color: m.qty > 0 ? "#ff0044" : "#00ff88", fontFamily: "var(--font-mono)" }}>
                        {m.qty > 0 ? "+" : ""}{m.qty}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </>
          ) : (
            <p style={{ color: "#444", fontSize: 12 }}>No inventory data yet</p>
          )}
        </div>

        {/* Sales */}
        <div style={{ background: "#0a0a0f", borderRadius: 8, padding: 16 }}>
          <div style={{ fontSize: 9, color: "#555", textTransform: "uppercase", letterSpacing: 1, marginBottom: 12 }}>
            Sales Performance
          </div>
          {sales ? (
            <>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 12 }}>
                <div>
                  <div style={{ fontSize: 9, color: "#555" }}>Total Revenue</div>
                  <div style={{ fontSize: 16, fontWeight: 700, color: "#00ff88" }}>{fmtCurrency(sales.total_revenue)}</div>
                </div>
                <div>
                  <div style={{ fontSize: 9, color: "#555" }}>Growth Rate</div>
                  <div style={{ fontSize: 16, fontWeight: 700, color: trendColor(sales.growth_rate > 0 ? "up" : sales.growth_rate < 0 ? "down" : "stable") }}>
                    {sales.growth_rate > 0 ? "+" : ""}{(sales.growth_rate * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                <div style={{ background: "#111", borderRadius: 6, padding: 10, textAlign: "center" }}>
                  <div style={{ fontSize: 20, fontWeight: 700, color: "#00aaff" }}>{sales.order_count}</div>
                  <div style={{ fontSize: 9, color: "#555", marginTop: 2 }}>Orders</div>
                </div>
                <div style={{ background: "#111", borderRadius: 6, padding: 10, textAlign: "center" }}>
                  <div style={{ fontSize: 20, fontWeight: 700, color: "#888" }}>{fmtCurrency(sales.avg_order_value)}</div>
                  <div style={{ fontSize: 9, color: "#555", marginTop: 2 }}>Avg Order</div>
                </div>
              </div>
              {sales.recent_orders.length > 0 && (
                <div style={{ borderTop: "1px solid #1a1a2e", paddingTop: 8, marginTop: 8 }}>
                  <div style={{ fontSize: 9, color: "#444", marginBottom: 4 }}>Recent Orders</div>
                  {sales.recent_orders.slice(0, 4).map((o, i) => (
                    <div key={i} style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#666", padding: "2px 0" }}>
                      <span>{o.state}</span>
                      <span style={{ color: "#00ff88", fontFamily: "var(--font-mono)" }}>{fmtCurrency(o.amount)}</span>
                    </div>
                  ))}
                </div>
              )}
            </>
          ) : (
            <p style={{ color: "#444", fontSize: 12 }}>No sales data yet</p>
          )}
        </div>
      </div>

      {/* ═══ Signal Mapping Reference ═══ */}
      <div style={{ background: "#0a0a0f", borderRadius: 8, padding: 16 }}>
        <div style={{ fontSize: 9, color: "#555", textTransform: "uppercase", letterSpacing: 1, marginBottom: 12 }}>
          Signal Mapping Reference
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 8, fontSize: 10, color: "#888" }}>
          <div style={{ borderLeft: "2px solid #00aaff", paddingLeft: 8 }}>
            <div style={{ color: "#00aaff", fontWeight: 600 }}>Inventory ↓</div>
            <div style={{ color: "#555" }}>demand ↓ → bearish</div>
          </div>
          <div style={{ borderLeft: "2px solid #00ff88", paddingLeft: 8 }}>
            <div style={{ color: "#00ff88", fontWeight: 600 }}>Sales ↑</div>
            <div style={{ color: "#555" }}>demand strength ↑ → bullish</div>
          </div>
          <div style={{ borderLeft: "2px solid #ff0044", paddingLeft: 8 }}>
            <div style={{ color: "#ff0044", fontWeight: 600 }}>Revenue ↓</div>
            <div style={{ color: "#555" }}>risk-off → reduce exposure</div>
          </div>
          <div style={{ borderLeft: "2px solid #aa00ff", paddingLeft: 8 }}>
            <div style={{ color: "#aa00ff", fontWeight: 600 }}>CRM Pipeline ↑</div>
            <div style={{ color: "#555" }}>growth trajectory ↑ → bullish</div>
          </div>
        </div>
      </div>
    </div>
  );
}
