"use client";

import { useEffect, useState, useCallback } from "react";
import "./globals.css";
import dynamic from "next/dynamic";

// Dynamic imports for client-side only components
const AgentSwarmVisor = dynamic(() => import("../components/AgentSwarmVisor"), { ssr: false });
const PriceChart = dynamic(() => import("../components/PriceChart"), { ssr: false });
const MicrostructurePanel = dynamic(() => import("../components/MicrostructurePanel"), { ssr: false });
const MarketStructurePanel = dynamic(() => import("../components/MarketStructurePanel"), { ssr: false });
const InferenceRoutingPanel = dynamic(() => import("../components/InferenceRoutingPanel"), { ssr: false });

// ── Types ─────────────────────────────────────────────────

interface Portfolio {
  balance: number; equity: number; unrealized_pnl: number;
  open_positions: number; drawdown: number; total_trades: number;
  winning_trades: number; win_rate: number; consecutive_losses: number;
  total_pnl: number;
}

interface Signal {
  symbol: string; source: string; signal: string; confidence: number;
  entry_price: number | null; stop_loss: number | null;
  take_profits: { level: number; price: number; qty_pct: number }[];
  reason: string; metadata: Record<string, unknown>;
}

interface Agent {
  id: string; label: string; color: string;
  signal: "long" | "short" | "hold"; confidence: number; active: boolean;
}

// ── Color Map ─────────────────────────────────────────────

const SIGNAL_COLORS: Record<string, string> = {
  long: "#00ff88", short: "#ff0044", hold: "#888",
};

// ── Simulated Data Generators ───────────────────────────

function generateMockCandles(count: number) {
  const candles = [];
  let price = 50000;
  const now = Math.floor(Date.now() / 1000);  // Current time in Unix seconds
  for (let i = count; i >= 0; i--) {
    const t = now - i * 3600;  // Hourly candles as Unix timestamps
    const change = (Math.random() - 0.5) * 400;
    const open = price;
    const close = price + change;
    const high = Math.max(open, close) + Math.random() * 200;
    const low = Math.min(open, close) - Math.random() * 200;
    candles.push({
      time: t,
      open, high, low, close,
      volume: Math.random() * 200 + 50,
    });
    price = close;
  }
  return candles;
}

function generateMicroData() {
  const ob_imbalance = (Math.random() - 0.5) * 2;
  return {
    orderbook: {
      imbalance: ob_imbalance,
      bid_volume: Math.random() * 50 + 10,
      ask_volume: Math.random() * 50 + 10,
      signal: ob_imbalance > 0.3 ? "bullish" : ob_imbalance < -0.3 ? "bearish" : "neutral",
    },
    delta: {
      divergence: Math.random() > 0.7 ? (Math.random() > 0.5 ? "bullish" : "bearish") : "none",
      strength: Math.random() * 0.8,
      cvd: (Math.random() - 0.5) * 10,
      delta_trend: ["rising", "falling", "neutral"][Math.floor(Math.random() * 3)],
    },
    spoofing: {
      spoofing_detected: Math.random() > 0.85,
      confidence: Math.random() * 0.5,
      cancel_to_order_ratio: Math.random() * 8 + 0.5,
    },
    cascade: {
      cascade_risk: Math.random() * 0.6,
      severity: ["none", "low", "elevated"][Math.floor(Math.random() * 3)],
      reasoning: "Normal market conditions",
    },
  };
}

function generateStructureData() {
  const phases = ["accumulation", "markup", "distribution", "markdown", "unknown"];
  const phase = phases[Math.floor(Math.random() * phases.length)];
  const dir = phase === "markup" ? "long" : phase === "markdown" ? "short" : phase === "accumulation" ? "long" : "hold";
  return {
    signal: {
      direction: dir,
      confidence: Math.random() * 0.4 + 0.3,
      reasoning: `Wyckoff ${phase} phase detected`,
      wyckoff_phase: phase,
      has_liquidity_sweep: Math.random() > 0.7,
      has_order_block: Math.random() > 0.5,
      has_structure_break: Math.random() > 0.6,
    },
    structure: {
      swing_highs: Math.floor(Math.random() * 15),
      swing_lows: Math.floor(Math.random() * 15),
      bos_up: Math.floor(Math.random() * 8),
      bos_down: Math.floor(Math.random() * 8),
      choch: Math.floor(Math.random() * 5),
      liq_sweeps: Math.floor(Math.random() * 6),
      order_blocks: Math.floor(Math.random() * 4),
      fvg_gaps: Math.floor(Math.random() * 3),
      wyckoff_phase: phase,
    },
  };
}

// ── Inference Routing Types ──────────────────────────────

interface ProviderHealth {
  available: boolean;
  avg_latency_ms?: number;
  error_rate?: number;
  [key: string]: unknown;
}

interface AgentChain {
  base_chain: string[];
  adapted_chain: string[] | null;
  task_override: string | null;
  adapted: boolean;
}

interface ProviderPerf {
  ema_latency_ms: number;
  p50_latency_ms: number;
  samples: number;
  successes: number;
  failures: number;
  success_rate: number;
  last_updated: number;
}

interface AdaptiveAgentData {
  [provider: string]: ProviderPerf | boolean | string[] | undefined;
  _adaptive_ready?: boolean;
  _adapted_chain?: string[];
  _chain_adapted?: boolean;
}

interface AgentMapping {
  agent_id: string;
  agent_name: string;
  primary_model: string;
  primary_provider: string;
  primary_free: boolean;
  fallback_model: string | null;
  latency_sensitive: boolean;
  reasoning_score: number;
  speed_score: number;
}

interface CostUsage {
  today?: Record<string, unknown>;
  cache?: Record<string, unknown>;
  over_budget?: boolean;
}

interface InferenceRoutingData {
  provider_health: Record<string, ProviderHealth>;
  agent_chains: Record<string, AgentChain>;
  adaptive_routing: Record<string, AdaptiveAgentData>;
  agent_mappings: AgentMapping[];
  cost_usage: CostUsage;
}

// ── Main Component ────────────────────────────────────────

export default function Dashboard() {
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [signal, setSignal] = useState<Signal | null>(null);
  const [agents, setAgents] = useState<Agent[]>([]);
  const [candles, setCandles] = useState(generateMockCandles(100));
  const [microData, setMicroData] = useState(generateMicroData());
  const [structureData, setStructureData] = useState(generateStructureData());
  const [inferenceRouting, setInferenceRouting] = useState<InferenceRoutingData | null>(null);
  const [wsStatus, setWsStatus] = useState("disconnected");
  const [events, setEvents] = useState<string[]>([]);
  const [riskScore, setRiskScore] = useState(42);
  const [showReasoning, setShowReasoning] = useState(false);
  const [activeTab, setActiveTab] = useState("overview");

  // Agent definitions
  const defaultAgents: Agent[] = [
    { id: "SCALP", label: "Scalping", color: "#00ff88", signal: "hold", confidence: 0.5, active: true },
    { id: "SWING", label: "Swing", color: "#00aaff", signal: "hold", confidence: 0.6, active: true },
    { id: "RISK", label: "Risk Guard", color: "#ff0044", signal: "hold", confidence: 0.8, active: true },
    { id: "SENT", label: "Sentiment", color: "#aa00ff", signal: "hold", confidence: 0.4, active: true },
    { id: "REGIME", label: "Regime", color: "#ffaa00", signal: "hold", confidence: 0.7, active: true },
    { id: "ML", label: "ML Model", color: "#00ff88", signal: "hold", confidence: 0.55, active: true },
    { id: "EXEC", label: "Execution", color: "#e0e0e0", signal: "hold", confidence: 0.5, active: true },
    { id: "MEM", label: "Memory", color: "#888", signal: "hold", confidence: 0.3, active: false },
  ];

  // ── Data Fetching ──────────────────────────────────────

  const fetchPortfolio = useCallback(async () => {
    try {
      const res = await fetch("/api/portfolio");
      if (res.ok) {
        const data = await res.json();
        setPortfolio(data);
        setRiskScore(Math.round((data.drawdown || 0) * 100 + (data.consecutive_losses || 0) * 5));
      }
    } catch { /* use defaults */ }
  }, []);

  const fetchSignal = useCallback(async () => {
    try {
      const res = await fetch("/api/signal?symbol=BTCUSDT&source=ml");
      if (res.ok) {
        const data = await res.json();
        setSignal(data);
        // Update agent signals
        setAgents((prev) =>
          prev.map((a) =>
            a.id === "ML"
              ? { ...a, signal: data.signal as any, confidence: data.confidence }
              : a
          )
        );
      }
    } catch { /* ignore */ }
  }, []);

  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch("/api/status");
      if (res.ok) {
        const data = await res.json();
        const p = data.portfolio;
        if (p) {
          setPortfolio(p);
          setRiskScore(Math.round((p.drawdown || 0) * 100 + (p.consecutive_losses || 0) * 5));
        }
      }
    } catch { /* ignore */ }
  }, []);

  // ── Effects ─────────────────────────────────────────

  useEffect(() => {
    fetchPortfolio();
    fetchSignal();
    fetchStatus();

    const agentInterval = setInterval(() => {
      // Simulate agent state changes
      setAgents((prev) =>
        prev.map((a) => ({
          ...a,
          signal: Math.random() > 0.7
            ? (["long", "short", "hold"] as const)[Math.floor(Math.random() * 2)]
            : "hold",
          confidence: Math.min(1, Math.max(0, a.confidence + (Math.random() - 0.5) * 0.2)),
        }))
      );
    }, 5000);

    // Simulate real-time data
    const tickInterval = setInterval(() => {
      setCandles((prev) => {
        const last = prev[prev.length - 1];
        const change = (Math.random() - 0.5) * 200;
        const newCandle = {
          time: Math.floor(Date.now() / 1000),
          open: last.close,
          high: Math.max(last.close, last.close + change) + Math.random() * 100,
          low: Math.min(last.close, last.close + change) - Math.random() * 100,
          close: last.close + change,
          volume: Math.random() * 200 + 50,
        };
        return [...prev.slice(-99), newCandle];
      });
      setMicroData(generateMicroData());
      setStructureData(generateStructureData());
      setEvents((prev) => [
        `BTC $${(50000 + (Math.random() - 0.5) * 2000).toFixed(2)} ${Math.random() > 0.5 ? "▲" : "▼"}`,
        ...prev.slice(0, 49),
      ]);
    }, 3000);

    // WebSocket connection
    const connectWs = () => {
      try {
        const ws = new WebSocket("ws://localhost:8001/ws");
        ws.onopen = () => setWsStatus("connected");
        ws.onclose = () => {
          setWsStatus("disconnected");
          setTimeout(connectWs, 3000);
        };
        ws.onmessage = (msg) => {
          try {
            const data = JSON.parse(msg.data);
            // Handle typed WebSocket events
            if (data.type === "inference_routing") {
              setInferenceRouting(data);
            } else if (data.type === "portfolio") {
              setPortfolio(data);
              setRiskScore(Math.round((data.drawdown || 0) * 100 + (data.consecutive_losses || 0) * 5));
            } else if (data.type === "signal") {
              setSignal(data);
              setAgents((prev) =>
                prev.map((a) =>
                  a.id === "ML"
                    ? { ...a, signal: data.signal as any, confidence: data.confidence }
                    : a
                )
              );
            } else {
              setEvents((prev) => [JSON.stringify(data).slice(0, 80), ...prev.slice(0, 49)]);
            }
          } catch { /* ignore */ }
        };
      } catch { /* ignore */ }
    };
    connectWs();

    // Initialize agents
    setAgents(defaultAgents);

    return () => {
      clearInterval(agentInterval);
      clearInterval(tickInterval);
    };
  }, [fetchPortfolio, fetchSignal, fetchStatus]);

  const pnlColor = (portfolio?.total_pnl ?? 0) >= 0 ? "#00ff88" : "#ff0044";

  // ── Tab click handler ──────────────────────────────

  const handleTabClick = (tab: string) => {
    setActiveTab(tab);
    if (tab === "inference") {
      // Data is pushed via WebSocket every 5s — no fetch needed
    }
  };
  const riskLabel = riskScore < 30 ? "Low" : riskScore < 60 ? "Medium" : "High";
  const riskColor = riskScore < 30 ? "#00ff88" : riskScore < 60 ? "#ffaa00" : "#ff0044";

  return (
    <div className="cockpit-grid">
      {/* ═══ Header ═══ */}
      <header className="cockpit-header">
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <div style={{
            width: 36, height: 36, borderRadius: 8,
            background: "linear-gradient(135deg, #00ff88, #00aaff)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 14, fontWeight: 700, color: "#0a0a0f",
          }}>
            Q
          </div>
          <div>
            <h1 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: "#eee", letterSpacing: 1 }}>
              QUANTEX COCKPIT
            </h1>
            <p style={{ margin: "2px 0 0", fontSize: 10, color: "#555", letterSpacing: 2 }}>
              AUTONOMOUS AI TRADING SYSTEM v1.0
            </p>
          </div>
        </div>
        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          <div className="badge" style={{
            background: `${riskColor}22`, color: riskColor,
            border: `1px solid ${riskColor}44`,
            padding: "6px 14px", fontSize: 10,
          }}>
            RISK: {riskLabel.toUpperCase()}
            <span style={{ marginLeft: 6, fontWeight: 700 }}>{riskScore}%</span>
          </div>
          <div className={`badge ${wsStatus === "connected" ? "badge-green" : "badge-red"}`}
               style={{ padding: "6px 14px", fontSize: 10 }}>
            {wsStatus === "connected" ? "● LIVE" : "○ OFFLINE"}
          </div>
          <div className="badge badge-gray" style={{ padding: "6px 14px", fontSize: 10 }}>
            PAPER MODE
          </div>
        </div>
      </header>

      {/* ═══ Top Row: Portfolio Stats ═══ */}
      <div className="top-row">
        <div className="panel stat-card">
          <div className="panel-title">Account Balance</div>
          <div className="stat-value green">
            ${portfolio?.balance.toFixed(2) ?? "1,000.00"}
          </div>
          <div className="stat-subtitle" style={{ color: pnlColor }}>
            PnL: {portfolio ? `${portfolio.total_pnl >= 0 ? "+" : ""}$${portfolio.total_pnl.toFixed(2)}` : "Loading..."}
          </div>
        </div>
        <div className="panel stat-card">
          <div className="panel-title">Win Rate</div>
          <div className="stat-value" style={{ color: (portfolio?.win_rate ?? 0) > 0.5 ? "#00ff88" : "#ff0044" }}>
            {portfolio ? `${(portfolio.win_rate * 100).toFixed(1)}%` : "—"}
          </div>
          <div className="stat-subtitle">
            {portfolio?.winning_trades ?? 0}W / {((portfolio?.total_trades ?? 0) - (portfolio?.winning_trades ?? 0))}L
          </div>
        </div>
        <div className="panel stat-card">
          <div className="panel-title">Drawdown</div>
          <div className="stat-value" style={{ color: (portfolio?.drawdown ?? 0) > 0.1 ? "#ff0044" : "#888" }}>
            {portfolio ? `${(portfolio.drawdown * 100).toFixed(1)}%` : "—"}
          </div>
          <div className="stat-subtitle">
            Equity: ${portfolio?.equity.toFixed(2) ?? "—"}
          </div>
        </div>
        <div className="panel stat-card">
          <div className="panel-title">Positions</div>
          <div className="stat-value" style={{ color: (portfolio?.open_positions ?? 0) > 0 ? "#00aaff" : "#888" }}>
            {portfolio?.open_positions ?? 0}
          </div>
          <div className="stat-subtitle">
            Cons. Losses: <span style={{ color: (portfolio?.consecutive_losses ?? 0) >= 3 ? "#ff0044" : "#888" }}>
              {portfolio?.consecutive_losses ?? 0}
            </span>
          </div>
        </div>
      </div>

      {/* ═══ Tab Navigation ═══ */}
      <div style={{ display: "flex", gap: 4, gridColumn: "1 / -1" }}>
        {["overview", "trading", "risk", "microstructure", "inference"].map((tab) => (
          <button key={tab} onClick={() => handleTabClick(tab)}
            style={{
              padding: "8px 20px", borderRadius: 8, fontSize: 11, fontWeight: 500,
              background: activeTab === tab ? "#1a1a2e" : "transparent",
              color: activeTab === tab ? "#ccc" : "#555",
              border: `1px solid ${activeTab === tab ? "#333" : "transparent"}`,
              cursor: "pointer", textTransform: "uppercase", letterSpacing: 1,
              transition: "all 0.2s ease",
            }}>
            {tab}
          </button>
        ))}
      </div>

      {/* ═══ OVERVIEW TAB ═══ */}
      {activeTab === "overview" && (
        <>
          {/* Mid Row: 3D Swarm Viz + Latest Signal */}
          <div className="mid-row">
            <div className="panel" style={{ padding: 0, overflow: "hidden" }}>
              <div style={{ padding: "16px 20px 0", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div className="panel-title" style={{ margin: 0 }}>Agent Swarm Intelligence</div>
                <span style={{ fontSize: 9, color: "#555" }}>
                  {agents.filter((a) => a.active).length}/{agents.length} active
                </span>
              </div>
              <AgentSwarmVisor agents={agents} />
            </div>

            <div className="panel">
              <div className="panel-title">Latest Signal</div>
              {signal ? (
                <div style={{ animation: "slideIn 0.3s ease" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16 }}>
                    <span className={`badge ${signal.signal === "long" ? "badge-green" : signal.signal === "short" ? "badge-red" : "badge-gray"}`}
                          style={{ fontSize: 14, fontWeight: 700, padding: "6px 16px" }}>
                      {signal.signal.toUpperCase()}
                    </span>
                    <span style={{ fontSize: 13, color: "#aaa" }}>
                      {(signal.confidence * 100).toFixed(0)}% confidence
                    </span>
                    <span style={{ fontSize: 10, color: "#555", marginLeft: "auto" }}>
                      via {signal.source}
                    </span>
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6, fontSize: 12, color: "#888", lineHeight: 2 }}>
                    <div>Entry: <strong style={{ color: "#ccc" }}>${signal.entry_price?.toFixed(2) ?? "—"}</strong></div>
                    <div>Stop Loss: <strong style={{ color: "#ff0044" }}>${signal.stop_loss?.toFixed(2) ?? "—"}</strong></div>
                    {signal.take_profits?.map((tp) => (
                      <div key={tp.level}>TP{tp.level}: <strong style={{ color: "#00ff88" }}>${tp.price.toFixed(2)}</strong></div>
                    ))}
                  </div>
                  <button onClick={() => setShowReasoning(!showReasoning)}
                    style={{
                      marginTop: 10, background: "none", border: "1px solid #222",
                      borderRadius: 6, padding: "5px 12px", color: "#666",
                      cursor: "pointer", fontSize: 10,
                    }}>
                    {showReasoning ? "Hide" : "Show"} Reasoning
                  </button>
                  {showReasoning && (
                    <p style={{ marginTop: 8, fontSize: 11, color: "#555", fontFamily: "var(--font-mono)", padding: 8, background: "#0a0a0f", borderRadius: 6, lineHeight: 1.5 }}>
                      {signal.reason}
                    </p>
                  )}

                  {/* Agent opinions mini list */}
                  <div style={{ marginTop: 14, borderTop: "1px solid #111", paddingTop: 12 }}>
                    <div style={{ fontSize: 9, color: "#555", textTransform: "uppercase", letterSpacing: 1, marginBottom: 8 }}>
                      Agent Consensus
                    </div>
                    {agents.filter((a) => a.active).slice(0, 5).map((agent) => (
                      <div key={agent.id} style={{
                        display: "flex", justifyContent: "space-between", alignItems: "center",
                        padding: "4px 6px", marginBottom: 3, background: "#0a0a0f", borderRadius: 4, fontSize: 10,
                      }}>
                        <span style={{ color: agent.color, fontWeight: 600 }}>{agent.id}</span>
                        <span className={`badge ${agent.signal === "long" ? "badge-green" : agent.signal === "short" ? "badge-red" : "badge-gray"}`}
                              style={{ fontSize: 9, padding: "2px 8px" }}>
                          {agent.signal.toUpperCase()} {(agent.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <p style={{ color: "#444" }}>Waiting for signal...</p>
              )}
            </div>
          </div>

          {/* Bottom: Price Chart + Risk Gauge + Event Log */}
          <div className="bottom-row">
            <div className="panel" style={{ gridColumn: "span 2", padding: 0 }}>
              <div style={{ padding: "16px 20px 0" }}>
                <div className="panel-title" style={{ margin: 0 }}>BTC/USDT — Live Chart</div>
              </div>
              <PriceChart data={candles} />
            </div>

            <div className="panel">
              <div className="panel-title">Risk Dashboard</div>
              <div style={{ textAlign: "center", padding: "8px 0" }}>
                <div style={{
                  width: 90, height: 90, borderRadius: "50%",
                  border: `5px solid ${riskColor}`,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  margin: "0 auto 10px",
                  background: `radial-gradient(circle, ${riskColor}11, transparent)`,
                  transition: "border-color 0.5s ease",
                }}>
                  <div>
                    <p style={{ fontSize: 22, fontWeight: 700, color: riskColor }}>{riskScore}</p>
                    <p style={{ fontSize: 8, color: "#555" }}>/100</p>
                  </div>
                </div>
                <p style={{ fontSize: 11, color: riskColor, fontWeight: 600, letterSpacing: 2 }}>
                  {riskLabel.toUpperCase()} RISK
                </p>
                <div style={{ marginTop: 10, fontSize: 10, color: "#888", textAlign: "left" }}>
                  <div className="risk-row">
                    <span>Drawdown</span>
                    <span style={{ color: (portfolio?.drawdown ?? 0) > 0.1 ? "#ff0044" : "#888" }}>
                      {((portfolio?.drawdown ?? 0) * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="risk-row">
                    <span>Consec. Losses</span>
                    <span style={{ color: (portfolio?.consecutive_losses ?? 0) >= 3 ? "#ff0044" : "#888" }}>
                      {portfolio?.consecutive_losses ?? 0}
                    </span>
                  </div>
                  <div className="risk-row">
                    <span>Open Positions</span>
                    <span>{portfolio?.open_positions ?? 0}</span>
                  </div>
                  <div className="risk-row">
                    <span>Win Rate</span>
                    <span style={{ color: (portfolio?.win_rate ?? 0) > 0.5 ? "#00ff88" : "#888" }}>
                      {((portfolio?.win_rate ?? 0) * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      {/* ═══ TRADING TAB ═══ */}
      {activeTab === "trading" && (
        <div className="mid-row">
          <MarketStructurePanel data={structureData} />
          <div className="panel">
            <div className="panel-title">Event Log</div>
            <div style={{ height: 240, overflowY: "auto", fontSize: 10, fontFamily: "var(--font-mono)" }}>
              {events.length === 0 ? (
                <p style={{ color: "#444" }}>Waiting for events...</p>
              ) : (
                events.slice(0, 30).map((ev, i) => (
                  <div key={i} style={{
                    padding: "3px 0", color: i === 0 ? "#ccc" : "#444",
                    borderBottom: i === 0 ? "1px solid #1a1a2e" : "none",
                  }}>
                    <span style={{ color: "#333" }}>[{events.length - i}]</span> {ev}
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      )}

      {/* ═══ RISK TAB ═══ */}
      {activeTab === "risk" && (
        <div className="mid-row">
          <div className="panel">
            <div className="panel-title">Risk Parameters</div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, fontSize: 12, color: "#888" }}>
              {[
                ["Max Risk/Trade", "1.0%"],
                ["Max Daily Loss", "5.0%"],
                ["Max Drawdown", "20.0%"],
                ["Max Leverage", "10x"],
                ["Max Positions", "3"],
                ["Kelly Fraction", "25%"],
                ["Stop Loss ATR", "1.5x"],
                ["Kill Switch", "Active"],
              ].map(([k, v]) => (
                <div key={k} style={{ display: "flex", justifyContent: "space-between", padding: "6px 0", borderBottom: "1px solid #111" }}>
                  <span>{k}</span>
                  <strong style={{ color: v === "Active" ? "#00ff88" : "#ccc" }}>{v}</strong>
                </div>
              ))}
            </div>
          </div>
          <div className="panel">
            <div className="panel-title">Kill Switch Status</div>
            <div style={{ textAlign: "center", padding: "20px 0" }}>
              <div style={{
                width: 60, height: 60, borderRadius: "50%",
                border: "4px solid #00ff88",
                display: "flex", alignItems: "center", justifyContent: "center",
                margin: "0 auto 12px",
              }}>
                <span style={{ color: "#00ff88", fontSize: 11, fontWeight: 700 }}>OK</span>
              </div>
              <p style={{ fontSize: 12, color: "#888" }}>All systems nominal</p>
              <div style={{ marginTop: 16, fontSize: 10, color: "#555" }}>
                <p>Consecutive Wins: <strong style={{ color: "#00ff88" }}>3</strong></p>
                <p>Size Multiplier: <strong>1.15x</strong></p>
                <p>Time Decay: <strong style={{ color: "#00ff88" }}>None</strong></p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ═══ INFERENCE ROUTING TAB ═══ */}
      {activeTab === "inference" && (
        <InferenceRoutingPanel data={inferenceRouting} />
      )}

      {/* ═══ MICROSTRUCTURE TAB ═══ */}
      {activeTab === "microstructure" && (
        <div style={{ display: "grid", gap: 16, gridColumn: "1 / -1" }}>
          <MicrostructurePanel data={microData} />
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
            <MarketStructurePanel data={structureData} />
            <div className="panel">
              <div className="panel-title">Trading Signals</div>
              <div style={{ display: "grid", gap: 6 }}>
                {[
                  { name: "EMA Crossover", signal: "hold", conf: 0.4 },
                  { name: "ML Random Forest", signal: signal?.signal || "hold", conf: signal?.confidence || 0.5 },
                  { name: "Market Structure", signal: structureData.signal.direction, conf: structureData.signal.confidence },
                  { name: "Delta Divergence", signal: microData.delta.divergence === "none" ? "hold" : microData.delta.divergence, conf: microData.delta.strength },
                  { name: "Orderbook Imbalance", signal: microData.orderbook.signal === "neutral" ? "hold" : microData.orderbook.signal, conf: Math.abs(microData.orderbook.imbalance) },
                ].map((s) => (
                  <div key={s.name} style={{
                    display: "flex", justifyContent: "space-between", alignItems: "center",
                    padding: "8px 10px", background: "#0a0a0f", borderRadius: 6,
                    borderLeft: `3px solid ${
                      s.signal === "long" ? "#00ff88" : s.signal === "short" ? "#ff0044" : "#333"
                    }`,
                  }}>
                    <span style={{ fontSize: 11, color: "#888" }}>{s.name}</span>
                    <span className={`badge ${s.signal === "long" ? "badge-green" : s.signal === "short" ? "badge-red" : "badge-gray"}`}
                          style={{ fontSize: 9 }}>
                      {s.signal.toUpperCase()} {(s.conf * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ═══ Footer ═══ */}
      <footer style={{
        gridColumn: "1 / -1", textAlign: "center",
        padding: "20px 0 10px", fontSize: 9, color: "#333",
        borderTop: "1px solid #0d0d1a", letterSpacing: 1,
      }}>
        QUANTEX v1.0 — Autonomous AI Trading System — Research & Educational Use Only
      </footer>
    </div>
  );
}
