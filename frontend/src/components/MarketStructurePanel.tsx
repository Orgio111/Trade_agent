"use client";

interface MarketStructureData {
  signal: { direction: string; confidence: number; reasoning: string; wyckoff_phase: string; has_liquidity_sweep: boolean; has_order_block: boolean; has_structure_break: boolean };
  structure: { swing_highs: number; swing_lows: number; bos_up: number; bos_down: number; choch: number; liq_sweeps: number; order_blocks: number; fvg_gaps: number; wyckoff_phase: string };
}

export default function MarketStructurePanel({ data }: { data: MarketStructureData | null }) {
  if (!data) {
    return (
      <div className="panel">
        <div className="panel-title">Market Structure (SMC / Wyckoff)</div>
        <p style={{ color: "#444", fontSize: 12 }}>Waiting for data...</p>
      </div>
    );
  }

  const { signal, structure } = data;

  const phaseColors: Record<string, string> = {
    markup: "#00ff88", markdown: "#ff0044", accumulation: "#00aaff", distribution: "#ffaa00", unknown: "#888",
  };

  return (
    <div className="panel">
      <div className="panel-title">Market Structure (SMC / Wyckoff)</div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12 }}>
        {/* Wyckoff Phase */}
        <div style={{ background: "#0a0a0f", borderRadius: 8, padding: 12 }}>
          <div style={{ fontSize: 9, color: "#555", textTransform: "uppercase", letterSpacing: 1, marginBottom: 8 }}>
            Wyckoff Phase
          </div>
          <div style={{ fontSize: 16, fontWeight: 700, color: phaseColors[structure.wyckoff_phase] || "#888", marginBottom: 4 }}>
            {structure.wyckoff_phase.toUpperCase()}
          </div>
          <div style={{ fontSize: 10, color: "#555" }}>
            Signal: <strong style={{ color: signal.direction === "long" ? "#00ff88" : signal.direction === "short" ? "#ff0044" : "#888" }}>
              {signal.direction.toUpperCase()}
            </strong> ({(signal.confidence * 100).toFixed(0)}%)
          </div>
        </div>

        {/* Structure Events */}
        <div style={{ background: "#0a0a0f", borderRadius: 8, padding: 12 }}>
          <div style={{ fontSize: 9, color: "#555", textTransform: "uppercase", letterSpacing: 1, marginBottom: 8 }}>
            Structure Events
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 4, fontSize: 10, color: "#888" }}>
            <span>BOS Up: <strong style={{ color: structure.bos_up > 0 ? "#00ff88" : "#555" }}>{structure.bos_up}</strong></span>
            <span>BOS Down: <strong style={{ color: structure.bos_down > 0 ? "#ff0044" : "#555" }}>{structure.bos_down}</strong></span>
            <span>CHOCH: <strong style={{ color: structure.choch > 0 ? "#ffaa00" : "#555" }}>{structure.choch}</strong></span>
            <span>Liq Sweeps: <strong style={{ color: structure.liq_sweeps > 0 ? "#ff0044" : "#555" }}>{structure.liq_sweeps}</strong></span>
          </div>
        </div>

        {/* Key Levels */}
        <div style={{ background: "#0a0a0f", borderRadius: 8, padding: 12 }}>
          <div style={{ fontSize: 9, color: "#555", textTransform: "uppercase", letterSpacing: 1, marginBottom: 8 }}>
            Key Levels
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 4, fontSize: 10, color: "#888" }}>
            <span>Order Blocks: <strong style={{ color: structure.order_blocks > 0 ? "#00aaff" : "#555" }}>{structure.order_blocks}</strong></span>
            <span>FVG Gaps: <strong style={{ color: structure.fvg_gaps > 0 ? "#aa00ff" : "#555" }}>{structure.fvg_gaps}</strong></span>
            <span>Swing Highs: <strong style={{ color: "#aaa" }}>{structure.swing_highs}</strong></span>
            <span>Swing Lows: <strong style={{ color: "#aaa" }}>{structure.swing_lows}</strong></span>
          </div>
        </div>
      </div>

      {/* Signal reasoning */}
      <div style={{ marginTop: 10, fontSize: 10, color: "#555", fontFamily: "var(--font-mono)", padding: 8, background: "#0a0a0f", borderRadius: 6 }}>
        {signal.reasoning}
      </div>
    </div>
  );
}
