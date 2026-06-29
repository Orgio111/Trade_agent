"use client";

/**
 * QUANTEX Orderbook Heatmap — Real-time L2 orderbook depth visualization.
 *
 * Visualizes bid/ask depth as an interactive heatmap with:
 *   - Bid depth (green) / Ask depth (red) bars
 *   - Cumulative depth line
 *   - Price axis with mid-price marker
 *   - Imbalance indicator
 *   - Liquidity clusters (whale detection)
 *
 * Data format:
 *   bids: [[price, quantity], ...]  (sorted descending by price)
 *   asks: [[price, quantity], ...]  (sorted ascending by price)
 */

import { useMemo } from "react";

// ── Types ──────────────────────────────────────────────────────

export interface OrderbookData {
  symbol: string;
  bids: [number, number][];  // [price, quantity]
  asks: [number, number][];  // [price, quantity]
  spread: number;
  mid_price: number;
  imbalance: number;
  timestamp: number;
}

// ── Config ──────────────────────────────────────────────────────

const HEATMAP_LEVELS = 20;       // Number of price levels to show
const MIN_BAR_HEIGHT = 4;        // Minimum bar height in px
const MAX_BAR_HEIGHT = 40;       // Maximum bar height in px
const MAX_CUMULATIVE_WIDTH = 120; // Max cumulative depth bar width

// ── Helpers ─────────────────────────────────────────────────────

function formatPrice(p: number): string {
  if (p >= 1000) return p.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  if (p >= 1) return p.toFixed(2);
  return p.toFixed(6);
}

function formatQty(q: number): string {
  if (q >= 1000) return `${(q / 1000).toFixed(1)}K`;
  if (q >= 1) return q.toFixed(2);
  return q.toFixed(4);
}

/** Compute the color intensity for a depth level (0 → transparent, 1 → full). */
function depthColor(side: "bid" | "ask", intensity: number): string {
  const alpha = Math.min(1, Math.max(0.1, intensity));
  if (side === "bid") {
    return `rgba(0, 255, 136, ${alpha})`;
  }
  return `rgba(255, 0, 68, ${alpha})`;
}

/** Background glow for the mid-price line. */
function midPriceGlow(side: "bid" | "ask"): string {
  return side === "bid" ? "rgba(0, 255, 136, 0.08)" : "rgba(255, 0, 68, 0.08)";
}

/** Compute depth bars from orderbook data. */
function computeDepthBars(
  bids: [number, number][],
  asks: [number, number][],
  midPrice: number,
  levels: number,
) {
  // Take the top N levels closest to mid-price
  const nearBids = bids.slice(0, levels);
  const nearAsks = asks.slice(0, levels);

  // Compute max quantities for normalization
  const maxBidQty = Math.max(...nearBids.map(([, q]) => q), 0.0001);
  const maxAskQty = Math.max(...nearAsks.map(([, q]) => q), 0.0001);
  const globalMaxQty = Math.max(maxBidQty, maxAskQty);

  // Build bars: value ranges from 0 to 1 for cumulative opacity
  let cumBid = 0;
  const bidBars = nearBids.map(([price, qty]) => {
    cumBid += qty;
    return {
      price,
      qty,
      side: "bid" as const,
      intensity: Math.min(1, qty / globalMaxQty),
      cumulative: cumBid,
      cumulativeNormalized: Math.min(1, cumBid / (globalMaxQty * 3)),
    };
  });

  let cumAsk = 0;
  const askBars = nearAsks.map(([price, qty]) => {
    cumAsk += qty;
    return {
      price,
      qty,
      side: "ask" as const,
      intensity: Math.min(1, qty / globalMaxQty),
      cumulative: cumAsk,
      cumulativeNormalized: Math.min(1, cumAsk / (globalMaxQty * 3)),
    };
  });

  return { bidBars, askBars, maxCumulative: Math.max(cumBid, cumAsk) };
}

// ── Sub-components ─────────────────────────────────────────────

function DepthBar({
  price,
  qty,
  side,
  intensity,
  cumulativeNormalized,
  isHighest,
}: {
  price: number;
  qty: number;
  side: "bid" | "ask";
  intensity: number;
  cumulativeNormalized: number;
  isHighest: boolean;
}) {
  const barHeight = Math.max(MIN_BAR_HEIGHT, intensity * MAX_BAR_HEIGHT);
  const barColor = depthColor(side, intensity);
  const cumWidth = Math.max(4, cumulativeNormalized * MAX_CUMULATIVE_WIDTH);

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        height: barHeight + 2,
        marginBottom: 2,
        position: "relative",
        transition: "all 0.3s ease",
      }}
    >
      {/* Price label */}
      <div
        style={{
          width: 80,
          textAlign: "right",
          paddingRight: 8,
          fontSize: 10,
          fontFamily: "var(--font-mono)",
          color: isHighest ? "#fff" : "#666",
          fontWeight: isHighest ? 700 : 400,
          flexShrink: 0,
        }}
      >
        {formatPrice(price)}
      </div>

      {/* Depth bar */}
      <div
        style={{
          flex: 1,
          height: barHeight,
          borderRadius: 3,
          background: barColor,
          transition: "all 0.3s ease",
          position: "relative",
          minWidth: 10,
        }}
        title={`${side === "bid" ? "Bid" : "Ask"} ${formatPrice(price)} — ${formatQty(qty)}`}
      >
        {/* Quantity label on bar */}
        <span
          style={{
            position: "absolute",
            left: 6,
            top: "50%",
            transform: "translateY(-50%)",
            fontSize: 8,
            color: "rgba(255,255,255,0.7)",
            fontFamily: "var(--font-mono)",
            whiteSpace: "nowrap",
          }}
        >
          {formatQty(qty)}
        </span>
      </div>

      {/* Cumulative depth bar (thin line behind) */}
      <div
        style={{
          position: "absolute",
          right: 0,
          height: barHeight,
          width: cumWidth,
          borderRadius: 3,
          background: side === "bid"
            ? "linear-gradient(to left, rgba(0,255,136,0.15), transparent)"
            : "linear-gradient(to right, rgba(255,0,68,0.15), transparent)",
          pointerEvents: "none",
          transition: "all 0.3s ease",
        }}
      />
    </div>
  );
}

function LiquidityCluster({
  price,
  size,
  style,
}: {
  price: number;
  size: number;
  style: "bid" | "ask";
}) {
  const color = style === "bid" ? "#00ff88" : "#ff0044";
  const glowSize = Math.min(40, 10 + size * 30);

  return (
    <div
      style={{
        position: "absolute",
        left: `${50 + (style === "bid" ? -5 : 5)}%`,
        top: "50%",
        width: glowSize,
        height: glowSize,
        borderRadius: "50%",
        background: `radial-gradient(circle, ${color}44, transparent)`,
        transform: "translate(-50%, -50%)",
        pointerEvents: "none",
        animation: "pulse 2s ease-in-out infinite",
      }}
      title={`Whale cluster at ${formatPrice(price)} (${formatQty(size)})`}
    />
  );
}

// ── Main Component ────────────────────────────────────────────

export default function OrderbookHeatmap({
  data,
  compact = false,
}: {
  data: OrderbookData | null;
  compact?: boolean;
}) {
  const depthBars = useMemo(() => {
    if (!data) return null;
    return computeDepthBars(data.bids, data.asks, data.mid_price, HEATMAP_LEVELS);
  }, [data]);

  // Detect whale clusters (levels with unusually large quantity)
  const whaleClusters = useMemo(() => {
    if (!depthBars) return [];
    const allBars = [
      ...depthBars.bidBars.map((b) => ({ ...b, style: "bid" as const })),
      ...depthBars.askBars.map((a) => ({ ...a, style: "ask" as const })),
    ];
    const mean = allBars.reduce((s, b) => s + b.qty, 0) / Math.max(allBars.length, 1);
    const std = Math.sqrt(
      allBars.reduce((s, b) => s + (b.qty - mean) ** 2, 0) / Math.max(allBars.length, 1),
    );
    return allBars
      .filter((b) => b.qty > mean + 2 * std && std > 0)
      .map((b) => ({ price: b.price, size: b.qty, style: b.style }));
  }, [depthBars]);

  if (!data) {
    return (
      <div className="panel">
        <div className="panel-title">Orderbook Heatmap (L2)</div>
        <p style={{ color: "#444", fontSize: 12 }}>Waiting for orderbook data...</p>
      </div>
    );
  }

  const imbalancePct = (data.imbalance * 100).toFixed(0);
  const imbalanceColor =
    data.imbalance > 0.3 ? "#00ff88" : data.imbalance < -0.3 ? "#ff0044" : "#888";

  return (
    <div className="panel" style={{ position: "relative" }}>
      <div className="panel-title">
        Orderbook Heatmap (L2) — {data.symbol}
      </div>

      {/* ═══ Summary Bar ═══ */}
      <div
        style={{
          display: "flex",
          gap: 16,
          marginBottom: 14,
          fontSize: 10,
          color: "#888",
          padding: "8px 10px",
          background: "#0a0a0f",
          borderRadius: 6,
        }}
      >
        <div>
          Mid: <strong style={{ color: "#ccc", fontFamily: "var(--font-mono)" }}>
            ${formatPrice(data.mid_price)}
          </strong>
        </div>
        <div>
          Spread: <strong style={{ color: "#ccc", fontFamily: "var(--font-mono)" }}>
            ${formatPrice(data.spread)} ({(data.spread / data.mid_price * 10000).toFixed(1)}bps)
          </strong>
        </div>
        <div>
          Imbalance:{" "}
          <strong style={{ color: imbalanceColor, fontFamily: "var(--font-mono)" }}>
            {imbalancePct}%
          </strong>
        </div>
        <div style={{ marginLeft: "auto", display: "flex", gap: 4, alignItems: "center" }}>
          <span style={{ color: "#00ff88" }}>■</span> Bids{" "}
          <span style={{ color: "#ff0044", marginLeft: 6 }}>■</span> Asks
        </div>
      </div>

      {/* ═══ Depth Bars ═══ */}
      <div
        style={{
          display: "flex",
          gap: 0,
          position: "relative",
        }}
      >
        {/* Bids (left side) */}
        <div style={{ flex: 1 }}>
          <div
            style={{
              fontSize: 8,
              color: "#00ff88",
              textTransform: "uppercase",
              letterSpacing: 1,
              marginBottom: 4,
            }}
          >
            Bids
          </div>
          {depthBars?.bidBars.map((bar, i) => (
            <DepthBar
              key={`bid-${bar.price}-${i}`}
              {...bar}
              isHighest={i === depthBars.bidBars.length - 1}
            />
          ))}
          {(!depthBars || depthBars.bidBars.length === 0) && (
            <div style={{ color: "#333", fontSize: 10, padding: 10, textAlign: "center" }}>
              No bid data
            </div>
          )}
        </div>

        {/* Mid-price line */}
        <div
          style={{
            width: 2,
            background: `linear-gradient(to bottom, transparent, #444488, transparent)`,
            position: "relative",
            margin: "0 4px",
            flexShrink: 0,
          }}
        >
          {/* Mid-price marker */}
          <div
            style={{
              position: "absolute",
              width: 8,
              height: 8,
              borderRadius: "50%",
              background: "#444488",
              left: "50%",
              top: "50%",
              transform: "translate(-50%, -50%)",
              border: "2px solid #222244",
            }}
          />
        </div>

        {/* Asks (right side) */}
        <div style={{ flex: 1 }}>
          <div
            style={{
              fontSize: 8,
              color: "#ff0044",
              textTransform: "uppercase",
              letterSpacing: 1,
              marginBottom: 4,
            }}
          >
            Asks
          </div>
          {depthBars?.askBars.map((bar, i) => (
            <DepthBar
              key={`ask-${bar.price}-${i}`}
              {...bar}
              isHighest={i === 0}
            />
          ))}
          {(!depthBars || depthBars.askBars.length === 0) && (
            <div style={{ color: "#333", fontSize: 10, padding: 10, textAlign: "center" }}>
              No ask data
            </div>
          )}
        </div>
      </div>

      {/* ═══ Whale Clusters ═══ */}
      {whaleClusters.length > 0 && (
        <div
          style={{
            marginTop: 12,
            paddingTop: 10,
            borderTop: "1px solid #1a1a2e",
          }}
        >
          <div
            style={{
              fontSize: 9,
              color: "#555",
              textTransform: "uppercase",
              letterSpacing: 1,
              marginBottom: 8,
            }}
          >
            Whale Liquidity Clusters Detected
          </div>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
            {whaleClusters.map((wc, i) => (
              <div
                key={i}
                style={{
                  padding: "3px 10px",
                  borderRadius: 6,
                  fontSize: 10,
                  fontFamily: "var(--font-mono)",
                  background: wc.style === "bid" ? "rgba(0,255,136,0.1)" : "rgba(255,0,68,0.1)",
                  color: wc.style === "bid" ? "#00ff88" : "#ff0044",
                  border: `1px solid ${
                    wc.style === "bid" ? "rgba(0,255,136,0.2)" : "rgba(255,0,68,0.2)"
                  }`,
                }}
              >
                ${formatPrice(wc.price)} — {formatQty(wc.size)}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ═══ Legend ═══ */}
      <div
        style={{
          marginTop: 12,
          paddingTop: 8,
          borderTop: "1px solid #1a1a2e",
          display: "flex",
          gap: 16,
          fontSize: 8,
          color: "#444",
          justifyContent: "center",
        }}
      >
        <span>Bar width = depth at that level</span>
        <span>Thin bar = cumulative depth</span>
        <span>● = whale cluster</span>
      </div>
    </div>
  );
}
