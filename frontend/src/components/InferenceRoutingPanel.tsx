"use client";

// ── Types matching the /api/v2/inference/routing response ──────────────────

interface ProviderHealth {
  available: boolean;
  avg_latency_ms?: number;
  error_rate?: number;
  last_check?: number;
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

// ── Colors & helpers ──────────────────────────────────────────────────────

const PROVIDER_COLORS: Record<string, string> = {
  groq: "#00ff88",
  nvidia_nim: "#00aaff",
  openrouter: "#aa00ff",
  default: "#888",
};

const PROVIDER_LABELS: Record<string, string> = {
  groq: "Groq LPU",
  nvidia_nim: "NVIDIA NIM",
  openrouter: "OpenRouter",
};

const PROVIDER_DESCRIPTIONS: Record<string, string> = {
  groq: "Ultra-fast LPU inference, 800+ tok/s",
  nvidia_nim: "High-quality reasoning with DeepSeek V4 Flash",
  openrouter: "200+ models, universal fallback",
};

const TASK_COLORS: Record<string, string> = {
  urgent: "#ff0044",
  fast: "#ffaa00",
  reasoning: "#00aaff",
  analysis: "#00ff88",
  classification: "#aa00ff",
};

function pct(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

function round1(v: number): string {
  return v.toFixed(1);
}

function providerIcon(provider: string, available: boolean): string {
  const mapping: Record<string, string> = {
    groq: "\u26A1",        // lightning bolt
    nvidia_nim: "\u25B3",  // triangle
    openrouter: "\u2194",  // arrows
  };
  return mapping[provider] || "\u25CF";
}

function providerDescription(provider: string): string {
  return PROVIDER_DESCRIPTIONS[provider] || "";
}

// ── Sub-components ────────────────────────────────────────────────────────

function ProviderHealthCard({
  provider,
  health,
}: {
  provider: string;
  health: ProviderHealth;
}) {
  const available = health.available ?? false;
  const color = available ? PROVIDER_COLORS[provider] || "#888" : "#555";
  const label = PROVIDER_LABELS[provider] || provider;

  return (
    <div
      style={{
        background: "#0a0a0f",
        borderRadius: 10,
        padding: 16,
        border: `1px solid ${available ? `${color}33` : "#1a1a2e"}`,
        transition: "all 0.3s ease",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
        <div
          style={{
            width: 36,
            height: 36,
            borderRadius: 8,
            background: available ? `${color}18` : "#111",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 18,
            color: available ? color : "#444",
          }}
        >
          {providerIcon(provider, available)}
        </div>
        <div>
          <div style={{ fontSize: 13, fontWeight: 600, color: available ? "#ccc" : "#555" }}>
            {label}
          </div>
          <div style={{ fontSize: 10, color: "#555", marginTop: 2 }}>
            {providerDescription(provider)}
          </div>
        </div>
        <div style={{ marginLeft: "auto" }}>
          <div
            className={`badge ${available ? "badge-green" : "badge-gray"}`}
            style={{ fontSize: 9 }}
          >
            {available ? "\u25CF ONLINE" : "\u25CB OFFLINE"}
          </div>
        </div>
      </div>

      {available && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: 6,
            fontSize: 10,
            color: "#888",
          }}
        >
          <div>
            Avg Latency:{" "}
            <strong style={{ color }}>
              {health.avg_latency_ms ? `${round1(health.avg_latency_ms)}ms` : "—"}
            </strong>
          </div>
          <div>
            Error Rate:{" "}
            <strong
              style={{
                color:
                  (health.error_rate ?? 0) > 0.1
                    ? "#ff0044"
                    : (health.error_rate ?? 0) > 0.05
                      ? "#ffaa00"
                      : "#00ff88",
              }}
            >
              {health.error_rate != null ? pct(health.error_rate) : "—"}
            </strong>
          </div>
        </div>
      )}

      {/* Latency bar */}
      {available && health.avg_latency_ms != null && (
        <div style={{ marginTop: 10 }}>
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
                width: `${Math.min((health.avg_latency_ms / 1000) * 100, 100)}%`,
                height: "100%",
                background: color,
                borderRadius: 2,
                transition: "width 0.5s ease",
              }}
            />
          </div>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              fontSize: 8,
              color: "#444",
              marginTop: 3,
            }}
          >
            <span>0ms</span>
            <span>500ms</span>
            <span>1000ms</span>
          </div>
        </div>
      )}
    </div>
  );
}

function AgentChainRow({
  agentId,
  chain,
}: {
  agentId: string;
  chain: AgentChain;
}) {
  const agentLabel = agentId
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 10,
        padding: "8px 10px",
        background: "#0a0a0f",
        borderRadius: 8,
        fontSize: 11,
        borderLeft: `3px solid ${chain.task_override && TASK_COLORS[chain.task_override] ? TASK_COLORS[chain.task_override] : "#333"}`,
      }}
    >
      {/* Agent name */}
      <div style={{ width: 140, flexShrink: 0 }}>
        <div style={{ fontWeight: 600, color: "#ccc", fontSize: 11 }}>
          {agentLabel}
        </div>
        <div style={{ fontSize: 8, color: "#555", marginTop: 1 }}>
          {chain.task_override ? (
            <span
              className={`badge ${
                chain.task_override === "urgent"
                  ? "badge-red"
                  : chain.task_override === "fast"
                    ? "badge-yellow"
                    : chain.task_override === "reasoning"
                      ? "badge-blue"
                      : chain.task_override === "analysis"
                        ? "badge-green"
                        : "badge-gray"
              }`}
              style={{ fontSize: 8, padding: "1px 6px" }}
            >
              {chain.task_override.toUpperCase()}
            </span>
          ) : (
            <span style={{ color: "#444" }}>no override</span>
          )}
        </div>
      </div>

      {/* Provider chain arrows */}
      <div
        style={{
          flex: 1,
          display: "flex",
          alignItems: "center",
          gap: 4,
          flexWrap: "wrap",
        }}
      >
        {(chain.adapted_chain || chain.base_chain).map((p, i) => (
          <span key={p}>
            <span
              className={`badge ${
                PROVIDER_COLORS[p] ? "badge-blue" : "badge-gray"
              }`}
              style={{
                fontSize: 9,
                padding: "2px 8px",
                borderColor: `${PROVIDER_COLORS[p] || "#333"}33`,
                color: PROVIDER_COLORS[p] || "#888",
                background: `${PROVIDER_COLORS[p] || "#333"}11`,
              }}
            >
              {PROVIDER_LABELS[p] || p}
            </span>
            {i < (chain.adapted_chain || chain.base_chain).length - 1 && (
              <span style={{ color: "#333", fontSize: 10 }}>\u2192</span>
            )}
          </span>
        ))}
      </div>

      {/* Adaptation indicator */}
      {chain.adapted && chain.adapted_chain && (
        <div
          className="badge badge-yellow"
          style={{ fontSize: 8, padding: "2px 8px", flexShrink: 0 }}
        >
          ADAPTED
        </div>
      )}
    </div>
  );
}

function AdaptivePerfCard({
  agentId,
  providers,
}: {
  agentId: string;
  providers: Record<string, ProviderPerf | boolean | string[]>;
}) {
  const agentLabel = agentId
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());

  // Filter out internal keys
  const perfEntries = Object.entries(providers).filter(
    ([k]) => !k.startsWith("_")
  ) as [string, ProviderPerf][];

  if (perfEntries.length === 0) return null;

  return (
    <div
      style={{
        background: "#0a0a0f",
        borderRadius: 10,
        padding: 14,
        border: "1px solid #1a1a2e",
      }}
    >
      {/* Header */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 10,
        }}
      >
        <div style={{ fontSize: 12, fontWeight: 600, color: "#ccc" }}>
          {agentLabel}
        </div>
        {providers._adaptive_ready === true && (
          <div
            className="badge badge-green"
            style={{ fontSize: 8, padding: "2px 8px" }}
          >
            ADAPTIVE READY
          </div>
        )}
        {providers._adaptive_ready === false && (
          <div
            className="badge badge-gray"
            style={{ fontSize: 8, padding: "2px 8px" }}
          >
            COLLECTING...
          </div>
        )}
      </div>

      {/* Provider metrics */}
      <div style={{ display: "grid", gap: 8 }}>
        {perfEntries.map(([provider, perf]) => {
          const color = PROVIDER_COLORS[provider] || "#888";
          const sr = typeof perf.success_rate === "number" ? perf.success_rate : 0;
          const successColor =
            sr > 0.95 ? "#00ff88" : sr > 0.8 ? "#ffaa00" : "#ff0044";

          return (
            <div key={provider} style={{ fontSize: 10, color: "#888" }}>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  marginBottom: 4,
                }}
              >
                <span style={{ fontWeight: 600, color, fontSize: 10 }}>
                  {PROVIDER_LABELS[provider] || provider}
                </span>
                <span style={{ color: successColor }}>
                  {pct(sr)} success
                </span>
              </div>

              {/* Latency bar */}
              <div
                style={{
                  height: 4,
                  background: "#1a1a2e",
                  borderRadius: 2,
                  overflow: "hidden",
                  marginBottom: 3,
                }}
              >
                <div
                  style={{
                    width: `${Math.min(
                      ((typeof perf.ema_latency_ms === "number"
                        ? perf.ema_latency_ms
                        : 0) /
                        2000) *
                        100,
                      100
                    )}%`,
                    height: "100%",
                    background: color,
                    borderRadius: 2,
                    transition: "width 0.5s ease",
                  }}
                />
              </div>

              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  fontSize: 9,
                  color: "#555",
                }}
              >
                <span>
                  EMA:{" "}
                  <strong style={{ color }}>
                    {typeof perf.ema_latency_ms === "number"
                      ? `${round1(perf.ema_latency_ms)}ms`
                      : "—"}
                  </strong>
                </span>
                <span>
                  P50:{" "}
                  <strong style={{ color: "#888" }}>
                    {typeof perf.p50_latency_ms === "number"
                      ? `${round1(perf.p50_latency_ms)}ms`
                      : "—"}
                  </strong>
                </span>
                <span>
                  Samples:{" "}
                  <strong style={{ color: "#aaa" }}>
                    {typeof perf.samples === "number" ? perf.samples : 0}
                  </strong>
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Adapted chain indicator */}
      {providers._adapted_chain && Array.isArray(providers._adapted_chain) && (
        <div
          style={{
            marginTop: 8,
            paddingTop: 8,
            borderTop: "1px solid #1a1a2e",
            display: "flex",
            alignItems: "center",
            gap: 6,
            fontSize: 9,
            color: "#555",
          }}
        >
          <span>Adapted chain:</span>
          {(providers._adapted_chain as string[]).map((p, i) => (
            <span key={p}>
              <span
                style={{
                  color: PROVIDER_COLORS[p] || "#888",
                  fontWeight: 600,
                }}
              >
                {PROVIDER_LABELS[p] || p}
              </span>
              {i < (providers._adapted_chain as string[]).length - 1 && (
                <span style={{ color: "#333", margin: "0 2px" }}>\u2192</span>
              )}
            </span>
          ))}
          {(providers as AdaptiveAgentData)._chain_adapted === true && (
            <span
              className="badge badge-yellow"
              style={{ fontSize: 8, padding: "1px 6px", marginLeft: "auto" }}
            >
              REORDERED
            </span>
          )}
        </div>
      )}
    </div>
  );
}

// ── Latency Heatmap Component ────────────────────────────────────────────

function LatencyHeatmap({
  adaptiveRouting,
}: {
  adaptiveRouting: Record<string, AdaptiveAgentData>;
}) {
  // Extract all unique providers across all agents
  const allProviders = new Set<string>();
  const agentEntries: [string, Record<string, ProviderPerf>][] = [];

  for (const [agentId, providers] of Object.entries(adaptiveRouting)) {
    const perfEntries: Record<string, ProviderPerf> = {};
    for (const [key, val] of Object.entries(providers)) {
      if (key.startsWith("_")) continue;
      allProviders.add(key);
      perfEntries[key] = val as ProviderPerf;
    }
    if (Object.keys(perfEntries).length > 0) {
      agentEntries.push([agentId, perfEntries]);
    }
  }

  const sortedProviders = Array.from(allProviders).sort();
  const maxLatency = Math.max(
    ...agentEntries.flatMap(([, p]) =>
      Object.values(p).map((v) => v.ema_latency_ms ?? 0)
    ),
    1
  );

  // Color interpolation: 0→green, mid→yellow, max→red
  function heatColor(value: number): string {
    const ratio = Math.min(value / Math.max(maxLatency, 1), 1);
    // green (0,255,136) → yellow (255,170,0) → red (255,0,68)
    if (ratio < 0.5) {
      const t = ratio / 0.5;
      const r = Math.round(0 + t * 255);
      const g = Math.round(255 - t * 85);
      const b = Math.round(136 - t * 136);
      return `rgb(${r}, ${g}, ${b})`;
    } else {
      const t = (ratio - 0.5) / 0.5;
      const r = 255;
      const g = Math.round(170 - t * 170);
      const b = Math.round(0 + t * 68);
      return `rgb(${r}, ${g}, ${b})`;
    }
  }

  function heatBg(value: number): string {
    const ratio = Math.min(value / Math.max(maxLatency, 1), 1);
    const color = heatColor(value);
    // Add alpha based on how much data we have
    const alpha = 0.15 + ratio * 0.55;
    return `rgba(${color.slice(4, -1)}, ${alpha})`;
  }

  if (agentEntries.length === 0) return null;

  return (
    <div>
      {/* Table grid */}
      <div style={{ overflowX: "auto" }}>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: `180px repeat(${sortedProviders.length}, minmax(110px, 1fr))`,
            gap: 0,
            fontSize: 10,
          }}
        >
          {/* Header row */}
          <div
            style={{
              padding: "8px 10px",
              color: "#555",
              fontSize: 9,
              textTransform: "uppercase",
              letterSpacing: 1,
              borderBottom: "1px solid #1a1a2e",
            }}
          >
            Agent
          </div>
          {sortedProviders.map((provider) => (
            <div
              key={provider}
              style={{
                padding: "8px 6px",
                color: PROVIDER_COLORS[provider] || "#888",
                fontWeight: 600,
                fontSize: 9,
                textAlign: "center",
                textTransform: "uppercase",
                letterSpacing: 0.5,
                borderBottom: "1px solid #1a1a2e",
                borderLeft: "1px solid #111",
              }}
            >
              {PROVIDER_LABELS[provider] || provider}
            </div>
          ))}

          {/* Data rows */}
          {agentEntries.map(([agentId, perf], rowIdx) => {
            const agentLabel = agentId
              .replace(/_/g, " ")
              .replace(/\b\w/g, (c) => c.toUpperCase());

            return (
              <div
                key={agentId}
                style={{ display: "contents" }}
              >
                {/* Agent label */}
                <div
                  style={{
                    padding: "10px 10px",
                    color: "#ccc",
                    fontWeight: 600,
                    fontSize: 10,
                    display: "flex",
                    alignItems: "center",
                    gap: 6,
                    borderBottom:
                      rowIdx < agentEntries.length - 1
                        ? "1px solid #111"
                        : "none",
                  }}
                >
                  <span>{agentLabel}</span>
                  {/* Mini adapted indicator */}
                  {(adaptiveRouting[agentId] as AdaptiveAgentData)
                    ._chain_adapted === true && (
                    <span
                      className="badge badge-yellow"
                      style={{ fontSize: 7, padding: "1px 5px" }}
                    >
                      A
                    </span>
                  )}
                </div>

                {/* Provider cells */}
                {sortedProviders.map((provider) => {
                  const p = perf[provider];
                  const ema = p?.ema_latency_ms;
                  const hasData = ema != null && ema > 0;
                  const sr = p?.success_rate ?? 0;
                  const samples = p?.samples ?? 0;

                  return (
                    <div
                      key={`${agentId}-${provider}`}
                      style={{
                        padding: "8px 6px",
                        textAlign: "center",
                        fontSize: 11,
                        fontWeight: 600,
                        fontFamily: "var(--font-mono)",
                        background: hasData
                          ? heatBg(ema)
                          : "#08080e",
                        color: hasData
                          ? ema > maxLatency * 0.6
                            ? "#fff"
                            : "#ccc"
                          : "#333",
                        borderBottom:
                          rowIdx < agentEntries.length - 1
                            ? "1px solid #0a0a0f"
                            : "none",
                        borderLeft: "1px solid #111",
                        cursor: hasData ? "pointer" : "default",
                        transition: "all 0.2s ease",
                        position: "relative",
                      }}
                      title={
                        hasData
                          ? `${agentLabel} → ${PROVIDER_LABELS[provider] || provider}\n` +
                            `EMA Latency: ${round1(ema)}ms\n` +
                            `P50 Latency: ${round1(p!.p50_latency_ms)}ms\n` +
                            `Success Rate: ${pct(sr)}\n` +
                            `Samples: ${samples}`
                          : "No data"
                      }
                      onMouseEnter={(e) => {
                        if (hasData) {
                          e.currentTarget.style.transform = "scale(1.05)";
                          e.currentTarget.style.zIndex = "10";
                          e.currentTarget.style.boxShadow =
                            "0 4px 12px rgba(0,0,0,0.5)";
                          e.currentTarget.style.borderRadius = "4px";
                        }
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.transform = "scale(1)";
                        e.currentTarget.style.zIndex = "0";
                        e.currentTarget.style.boxShadow = "none";
                        e.currentTarget.style.borderRadius = "0";
                      }}
                    >
                      {hasData ? `${Math.round(ema)}ms` : "—"}
                    </div>
                  );
                })}
              </div>
            );
          })}
        </div>
      </div>

      {/* Color scale legend */}
      <div
        style={{
          marginTop: 12,
          paddingTop: 10,
          borderTop: "1px solid #1a1a2e",
          display: "flex",
          alignItems: "center",
          gap: 12,
          fontSize: 9,
          color: "#555",
        }}
      >
        <span>Latency:</span>
        {/* Gradient bar */}
        <div
          style={{
            flex: 1,
            maxWidth: 300,
            height: 8,
            borderRadius: 4,
            background:
              "linear-gradient(to right, rgb(0,255,136), rgb(255,170,0), rgb(255,0,68))",
          }}
        />
        <span style={{ fontFamily: "var(--font-mono)" }}>0ms</span>
        <div
          style={{
            flex: 1,
            maxWidth: 50,
            height: 1,
            background: "#333",
          }}
        />
        <span style={{ fontFamily: "var(--font-mono)" }}>
          {maxLatency < 1000
            ? `${Math.round(maxLatency)}ms`
            : `${(maxLatency / 1000).toFixed(1)}s`}
        </span>

        <span style={{ marginLeft: "auto", display: "flex", gap: 12 }}>
          <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <span
              className="badge badge-yellow"
              style={{ fontSize: 7, padding: "1px 5px" }}
            >
              A
            </span>{` `}
            Chain adapted
          </span>
          <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <div
              style={{
                width: 10,
                height: 10,
                borderRadius: 2,
                background: "#08080e",
                border: "1px solid #1a1a2e",
              }}
            />
            No data
          </span>
          <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <div
              style={{
                width: 8,
                height: 8,
                borderRadius: "50%",
                background: PROVIDER_COLORS.groq || "#00ff88",
              }}
            />
            Groq
          </span>
          <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <div
              style={{
                width: 8,
                height: 8,
                borderRadius: "50%",
                background: PROVIDER_COLORS.nvidia_nim || "#00aaff",
              }}
            />
            NIM
          </span>
          <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <div
              style={{
                width: 8,
                height: 8,
                borderRadius: "50%",
                background: PROVIDER_COLORS.openrouter || "#aa00ff",
              }}
            />
            OpenRouter
          </span>
        </span>
      </div>
    </div>
  );
}

// ── Main Component ────────────────────────────────────────────────────────

export default function InferenceRoutingPanel({
  data,
}: {
  data: InferenceRoutingData | null;
}) {
  if (!data) {
    return (
      <div className="panel">
        <div className="panel-title">Inference Routing</div>
        <p style={{ color: "#444", fontSize: 12 }}>Waiting for routing data...</p>
      </div>
    );
  }

  const { provider_health, agent_chains, adaptive_routing, agent_mappings, cost_usage } = data;
  const totalAgentsWithData = Object.keys(adaptive_routing).length;
  const adaptedAgents = Object.values(agent_chains).filter((c) => c.adapted).length;
  const providersOnline = Object.values(provider_health).filter((h) => h.available).length;
  const totalProviders = Object.keys(provider_health).length;

  return (
    <div style={{ display: "grid", gap: 16, gridColumn: "1 / -1" }}>
      {/* ═══ Summary Bar ═══ */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr 1fr 1fr",
          gap: 14,
          animation: "slideUp 0.3s ease",
        }}
      >
        <div className="panel stat-card">
          <div className="panel-title">Providers Online</div>
          <div
            className="stat-value"
            style={{
              color:
                providersOnline === totalProviders
                  ? "#00ff88"
                  : providersOnline > 0
                    ? "#ffaa00"
                    : "#ff0044",
            }}
          >
            {providersOnline}/{totalProviders}
          </div>
          <div className="stat-subtitle">
            {providersOnline === 0
              ? "All providers unavailable"
              : providersOnline === totalProviders
                ? "Full redundancy"
                : `${totalProviders - providersOnline} provider(s) down`}
          </div>
        </div>
        <div className="panel stat-card">
          <div className="panel-title">Agents Tracked</div>
          <div className="stat-value green">{totalAgentsWithData}</div>
          <div className="stat-subtitle">
            with adaptive routing data
          </div>
        </div>
        <div className="panel stat-card">
          <div className="panel-title">Adapted Chains</div>
          <div
            className="stat-value"
            style={{ color: adaptedAgents > 0 ? "#ffaa00" : "#888" }}
          >
            {adaptedAgents}
          </div>
          <div className="stat-subtitle">
            {adaptedAgents === 0
              ? "No chains reordered yet"
              : "Chain(s) reordered by latency"}
          </div>
        </div>
        <div className="panel stat-card">
          <div className="panel-title">Budget Status</div>
          <div
            className="stat-value"
            style={{
              color: cost_usage.over_budget ? "#ff0044" : "#00ff88",
              fontSize: 20,
            }}
          >
            {cost_usage.over_budget ? "OVER" : "OK"}
          </div>
          <div className="stat-subtitle">
            {cost_usage.cache
              ? `Cache: ${JSON.stringify(cost_usage.cache).slice(0, 40)}`
              : "No cache data"}
          </div>
        </div>
      </div>

      {/* ═══ Provider Health Section ═══ */}
      <div className="panel" style={{ animation: "slideUp 0.35s ease" }}>
        <div className="panel-title">Provider Health</div>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr 1fr",
            gap: 12,
          }}
        >
          {Object.entries(provider_health).map(([provider, health]) => (
            <ProviderHealthCard
              key={provider}
              provider={provider}
              health={health}
            />
          ))}
        </div>
      </div>

      {/* ═══ Agent Chains Section ═══ */}
      <div className="panel" style={{ animation: "slideUp 0.4s ease" }}>
        <div className="panel-title">
          Agent Provider Chains
          <span style={{ marginLeft: 8, fontWeight: 400, color: "#555" }}>
            — {adaptedAgents > 0 ? `${adaptedAgents} adapted by latency` : "static configuration"}
          </span>
        </div>
        <div style={{ display: "grid", gap: 4 }}>
          {Object.entries(agent_chains).map(([agentId, chain]) => (
            <AgentChainRow
              key={agentId}
              agentId={agentId}
              chain={chain}
            />
          ))}
        </div>

        {/* Legend */}
        <div
          style={{
            marginTop: 12,
            paddingTop: 10,
            borderTop: "1px solid #1a1a2e",
            display: "flex",
            gap: 16,
            fontSize: 9,
            color: "#555",
          }}
        >
          <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <span style={{ color: "#00ff88" }}>\u25CF</span> Task Override
          </span>
          <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <span style={{ color: "#ffaa00" }}>\u25CF</span> Chain Adapted
          </span>
          <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <span style={{ color: "#555" }}>\u2192</span> Provider fallback order
          </span>
        </div>
      </div>

      {/* ═══ Adaptive Routing Details Section ═══ */}
      <div className="panel" style={{ animation: "slideUp 0.45s ease" }}>
        <div className="panel-title">
          Adaptive Routing Perf
          <span style={{ marginLeft: 8, fontWeight: 400, color: "#555" }}>
            — EMA latency, success rate, and provider chain per agent
          </span>
        </div>
        {Object.keys(adaptive_routing).length === 0 ? (
          <div
            style={{
              textAlign: "center",
              padding: "30px 0",
              color: "#555",
              fontSize: 12,
            }}
          >
            No adaptive routing data yet. Data appears after agents make inference requests.
            <div
              style={{
                marginTop: 10,
                fontSize: 10,
                color: "#444",
              }}
            >
              Each request feeds latency observations into the adaptive router.
            </div>
          </div>
        ) : (
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: 12,
            }}
          >
            {Object.entries(adaptive_routing).map(([agentId, providers]) => (
              <AdaptivePerfCard
                key={agentId}
                agentId={agentId}
                providers={providers as Record<string, ProviderPerf | boolean | string[]>}
              />
            ))}
          </div>
        )}
      </div>

      {/* ═══ Latency Heatmap Section ═══ */}
      {Object.keys(adaptive_routing).length > 0 && (
        <div className="panel" style={{ animation: "slideUp 0.475s ease" }}>
          <div className="panel-title">
            EMA Latency Heatmap
            <span style={{ marginLeft: 8, fontWeight: 400, color: "#555" }}>
              — Per-agent × per-provider EMA latency (hover for details)
            </span>
          </div>
          <LatencyHeatmap adaptiveRouting={adaptive_routing} />
        </div>
      )}

      {/* ═══ Agent Model Mappings Section ═══ */}
      {agent_mappings.length > 0 && (
        <div className="panel" style={{ animation: "slideUp 0.5s ease" }}>
          <div className="panel-title">Agent Model Assignments</div>
          <div style={{ overflowX: "auto" }}>
            <table
              style={{
                width: "100%",
                borderCollapse: "collapse",
                fontSize: 10,
                color: "#888",
              }}
            >
              <thead>
                <tr
                  style={{
                    borderBottom: "1px solid #1a1a2e",
                    color: "#555",
                    textTransform: "uppercase",
                    fontSize: 9,
                    letterSpacing: 1,
                  }}
                >
                  <th style={{ padding: "6px 10px", textAlign: "left" }}>Agent</th>
                  <th style={{ padding: "6px 10px", textAlign: "left" }}>Primary Model</th>
                  <th style={{ padding: "6px 10px", textAlign: "left" }}>Provider</th>
                  <th style={{ padding: "6px 10px", textAlign: "center" }}>Free</th>
                  <th style={{ padding: "6px 10px", textAlign: "center" }}>Reasoning</th>
                  <th style={{ padding: "6px 10px", textAlign: "center" }}>Speed</th>
                  <th style={{ padding: "6px 10px", textAlign: "left" }}>Fallback</th>
                </tr>
              </thead>
              <tbody>
                {agent_mappings.map((m) => {
                  const label = m.agent_name || m.agent_id
                    .replace(/_/g, " ")
                    .replace(/\b\w/g, (c) => c.toUpperCase());
                  const provColor = PROVIDER_COLORS[m.primary_provider] || "#888";
                  return (
                    <tr
                      key={m.agent_id}
                      style={{
                        borderBottom: "1px solid #111",
                        transition: "background 0.2s",
                      }}
                      onMouseEnter={(e) =>
                        (e.currentTarget.style.background = "#0a0a0f")
                      }
                      onMouseLeave={(e) =>
                        (e.currentTarget.style.background = "transparent")
                      }
                    >
                      <td
                        style={{
                          padding: "8px 10px",
                          fontWeight: 600,
                          color: "#ccc",
                        }}
                      >
                        {label}
                      </td>
                      <td style={{ padding: "8px 10px", color: "#aaa" }}>
                        {m.primary_model}
                      </td>
                      <td style={{ padding: "8px 10px" }}>
                        <span style={{ color: provColor, fontWeight: 500 }}>
                          {m.primary_provider}
                        </span>
                      </td>
                      <td style={{ padding: "8px 10px", textAlign: "center" }}>
                        {m.primary_free ? (
                          <span style={{ color: "#00ff88" }}>FREE</span>
                        ) : (
                          <span style={{ color: "#ffaa00" }}>PAID</span>
                        )}
                      </td>
                      <td style={{ padding: "8px 10px", textAlign: "center" }}>
                        <div
                          style={{
                            display: "inline-flex",
                            alignItems: "center",
                            gap: 4,
                          }}
                        >
                          <div
                            style={{
                              width: 30,
                              height: 4,
                              background: "#1a1a2e",
                              borderRadius: 2,
                              overflow: "hidden",
                            }}
                          >
                            <div
                              style={{
                                width: `${m.reasoning_score * 100}%`,
                                height: "100%",
                                background:
                                  m.reasoning_score > 0.8
                                    ? "#00aaff"
                                    : m.reasoning_score > 0.5
                                      ? "#888"
                                      : "#555",
                                borderRadius: 2,
                              }}
                            />
                          </div>
                          <span style={{ color: "#888", fontSize: 9 }}>
                            {(m.reasoning_score * 100).toFixed(0)}
                          </span>
                        </div>
                      </td>
                      <td style={{ padding: "8px 10px", textAlign: "center" }}>
                        <div
                          style={{
                            display: "inline-flex",
                            alignItems: "center",
                            gap: 4,
                          }}
                        >
                          <div
                            style={{
                              width: 30,
                              height: 4,
                              background: "#1a1a2e",
                              borderRadius: 2,
                              overflow: "hidden",
                            }}
                          >
                            <div
                              style={{
                                width: `${m.speed_score * 100}%`,
                                height: "100%",
                                background:
                                  m.speed_score > 0.8
                                    ? "#00ff88"
                                    : m.speed_score > 0.5
                                      ? "#888"
                                      : "#555",
                                borderRadius: 2,
                              }}
                            />
                          </div>
                          <span style={{ color: "#888", fontSize: 9 }}>
                            {(m.speed_score * 100).toFixed(0)}
                          </span>
                        </div>
                      </td>
                      <td
                        style={{
                          padding: "8px 10px",
                          color: m.fallback_model ? "#555" : "#333",
                        }}
                      >
                        {m.fallback_model || "—"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ═══ Cost & Cache Section ═══ */}
      {cost_usage && Object.keys(cost_usage).length > 0 && (
        <div className="panel" style={{ animation: "slideUp 0.55s ease" }}>
          <div className="panel-title">Cost & Cache Usage</div>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr 1fr",
              gap: 12,
            }}
          >
            <div style={{ background: "#0a0a0f", borderRadius: 8, padding: 12 }}>
              <div
                style={{
                  fontSize: 9,
                  color: "#555",
                  textTransform: "uppercase",
                  letterSpacing: 1,
                  marginBottom: 8,
                }}
              >
                Today&apos;s Usage
              </div>
              <pre
                style={{
                  fontSize: 9,
                  color: "#888",
                  fontFamily: "var(--font-mono)",
                  whiteSpace: "pre-wrap",
                  wordBreak: "break-all",
                }}
              >
                {JSON.stringify(cost_usage.today ?? {}, null, 2).slice(0, 500)}
              </pre>
            </div>
            <div style={{ background: "#0a0a0f", borderRadius: 8, padding: 12 }}>
              <div
                style={{
                  fontSize: 9,
                  color: "#555",
                  textTransform: "uppercase",
                  letterSpacing: 1,
                  marginBottom: 8,
                }}
              >
                Cache Stats
              </div>
              <pre
                style={{
                  fontSize: 9,
                  color: "#888",
                  fontFamily: "var(--font-mono)",
                  whiteSpace: "pre-wrap",
                  wordBreak: "break-all",
                }}
              >
                {JSON.stringify(cost_usage.cache ?? {}, null, 2).slice(0, 500)}
              </pre>
            </div>
            <div style={{ background: "#0a0a0f", borderRadius: 8, padding: 12 }}>
              <div
                style={{
                  fontSize: 9,
                  color: "#555",
                  textTransform: "uppercase",
                  letterSpacing: 1,
                  marginBottom: 8,
                }}
              >
                Budget Status
              </div>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                }}
              >
                <div
                  style={{
                    width: 12,
                    height: 12,
                    borderRadius: "50%",
                    background: cost_usage.over_budget ? "#ff0044" : "#00ff88",
                    animation: cost_usage.over_budget
                      ? "pulse-fast 1s infinite"
                      : "none",
                  }}
                />
                <span
                  style={{
                    fontSize: 14,
                    fontWeight: 700,
                    color: cost_usage.over_budget ? "#ff0044" : "#00ff88",
                  }}
                >
                  {cost_usage.over_budget ? "OVER BUDGET" : "WITHIN BUDGET"}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
