"use client";

interface MicroData {
  orderbook: { imbalance: number; bid_volume: number; ask_volume: number; signal: string };
  delta: { divergence: string; strength: number; cvd: number; delta_trend: string };
  spoofing: { spoofing_detected: boolean; confidence: number; cancel_to_order_ratio: number };
  cascade: { cascade_risk: number; severity: string; reasoning: string };
}

export default function MicrostructurePanel({ data }: { data: MicroData | null }) {
  if (!data) {
    return (
      <div className="panel">
        <div className="panel-title">Microstructure Analysis</div>
        <p style={{ color: "#444", fontSize: 12 }}>Waiting for data...</p>
      </div>
    );
  }

  const { orderbook, delta, spoofing, cascade } = data;

  return (
    <div className="panel">
      <div className="panel-title">Microstructure Analysis</div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        {/* Orderbook Imbalance */}
        <div style={{ background: "#0a0a0f", borderRadius: 8, padding: 12 }}>
          <div style={{ fontSize: 9, color: "#555", textTransform: "uppercase", letterSpacing: 1, marginBottom: 8 }}>
            Orderbook Imbalance
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
            <div style={{
              flex: 1, height: 6, background: "#1a1a2e", borderRadius: 3, position: "relative", overflow: "hidden",
            }}>
              <div style={{
                width: `${Math.abs(orderbook.imbalance) * 100}%`,
                height: "100%",
                background: orderbook.imbalance > 0 ? "#00ff88" : "#ff0044",
                borderRadius: 3,
                marginLeft: orderbook.imbalance > 0 ? "50%" : `${50 - Math.abs(orderbook.imbalance) * 50}%`,
                transition: "all 0.5s ease",
              }} />
            </div>
            <span style={{ fontSize: 11, color: orderbook.imbalance > 0.3 ? "#00ff88" : orderbook.imbalance < -0.3 ? "#ff0044" : "#888", fontWeight: 600 }}>
              {(orderbook.imbalance * 100).toFixed(0)}%
            </span>
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#555" }}>
            <span>Bids: {orderbook.bid_volume.toFixed(1)}</span>
            <span>Asks: {orderbook.ask_volume.toFixed(1)}</span>
          </div>
          <div style={{ fontSize: 10, color: orderbook.signal === "bullish" ? "#00ff88" : orderbook.signal === "bearish" ? "#ff0044" : "#555", marginTop: 6, fontWeight: 500 }}>
            {orderbook.signal.toUpperCase()}
          </div>
        </div>

        {/* Delta Divergence */}
        <div style={{ background: "#0a0a0f", borderRadius: 8, padding: 12 }}>
          <div style={{ fontSize: 9, color: "#555", textTransform: "uppercase", letterSpacing: 1, marginBottom: 8 }}>
            Delta / CVD
          </div>
          <div style={{ fontSize: 18, fontWeight: 700, color: delta.divergence === "bullish" ? "#00ff88" : delta.divergence === "bearish" ? "#ff0044" : "#888", marginBottom: 4 }}>
            {delta.divergence === "none" ? "NEUTRAL" : delta.divergence.toUpperCase()}
          </div>
          <div style={{ display: "flex", gap: 12, fontSize: 10, color: "#555" }}>
            <span>CVD: <strong style={{ color: delta.cvd > 0 ? "#00ff88" : "#ff0044" }}>{delta.cvd.toFixed(2)}</strong></span>
            <span>Strength: <strong>{(delta.strength * 100).toFixed(0)}%</strong></span>
          </div>
          <div style={{ display: "flex", gap: 12, fontSize: 10, color: "#555", marginTop: 4 }}>
            <span>Delta: <strong style={{ color: delta.delta_trend === "rising" ? "#00ff88" : delta.delta_trend === "falling" ? "#ff0044" : "#888" }}>
              {delta.delta_trend.toUpperCase()}
            </strong></span>
          </div>
        </div>

        {/* Spoofing Alert */}
        <div style={{ background: "#0a0a0f", borderRadius: 8, padding: 12 }}>
          <div style={{ fontSize: 9, color: "#555", textTransform: "uppercase", letterSpacing: 1, marginBottom: 8 }}>
            Spoofing Detection
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
            <div style={{
              width: 8, height: 8, borderRadius: "50%",
              background: spoofing.spoofing_detected ? "#ff0044" : "#00ff88",
              animation: spoofing.spoofing_detected ? "pulse 1s ease-in-out infinite" : "none",
            }} />
            <span style={{ fontSize: 12, fontWeight: 600, color: spoofing.spoofing_detected ? "#ff0044" : "#888" }}>
              {spoofing.spoofing_detected ? "ALERT: Spoofing" : "Clean"}
            </span>
          </div>
          <div style={{ fontSize: 10, color: "#555" }}>
            Cancel/Order Ratio: <strong>{spoofing.cancel_to_order_ratio.toFixed(1)}x</strong>
          </div>
          <div style={{ fontSize: 10, color: "#555", marginTop: 2 }}>
            Confidence: {(spoofing.confidence * 100).toFixed(0)}%
          </div>
        </div>

        {/* Liquidation Cascade */}
        <div style={{ background: "#0a0a0f", borderRadius: 8, padding: 12 }}>
          <div style={{ fontSize: 9, color: "#555", textTransform: "uppercase", letterSpacing: 1, marginBottom: 8 }}>
            Liquidation Cascade Risk
          </div>
          <div style={{
            fontSize: 18, fontWeight: 700,
            color: cascade.severity === "critical" ? "#ff0044" : cascade.severity === "high" ? "#ff4400" : cascade.severity === "elevated" ? "#ffaa00" : cascade.severity === "low" ? "#00aaff" : "#888",
            marginBottom: 4,
          }}>
            {cascade.severity.toUpperCase()}
          </div>
          <div style={{ background: "#1a1a2e", borderRadius: 3, height: 6, width: "100%", overflow: "hidden" }}>
            <div style={{
              width: `${cascade.cascade_risk * 100}%`,
              height: "100%",
              background: cascade.cascade_risk > 0.7 ? "#ff0044" : cascade.cascade_risk > 0.4 ? "#ffaa00" : "#00aaff",
              borderRadius: 3,
              transition: "width 0.5s ease",
            }} />
          </div>
          <div style={{ fontSize: 9, color: "#555", marginTop: 4 }}>
            Score: {(cascade.cascade_risk * 100).toFixed(0)}%
          </div>
        </div>
      </div>
    </div>
  );
}
