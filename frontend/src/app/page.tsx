"use client";

import { useEffect, useState } from "react";
import { useTradingSocket } from "@/lib/use-trading-socket";
import { fetchConfig } from "@/lib/api-client";
import type { AppConfig } from "@/lib/types";

import BackgroundEffects from "@/components/BackgroundEffects";
import Header from "@/components/Header";
import PortfolioSummary from "@/components/PortfolioSummary";
import MarketTicker from "@/components/MarketTicker";
import AgentSignalsGrid from "@/components/AgentSignalsGrid";
import CouncilDeliberation from "@/components/CouncilDeliberation";
import RiskMetrics from "@/components/RiskMetrics";
import TradeLogTable from "@/components/TradeLogTable";
import SystemLog from "@/components/SystemLog";
import PaperTradingPanel from "@/components/PaperTradingPanel";
import ModelDeployment from "@/components/ModelDeployment";
import SentinelXStatus from "@/components/SentinelXStatus";
import RayServeHealth from "@/components/RayServeHealth";
import ConfigPanel from "@/components/ConfigPanel";

export default function DashboardPage() {
  const { data, status } = useTradingSocket();

  const [config, setConfig] = useState<AppConfig | null>(null);

  // Fetch config once on mount
  useEffect(() => {
    fetchConfig()
      .then(setConfig)
      .catch(() => {}); // Config fetch is best-effort
  }, []);

  // Derive paper active from data
  const paperActive = data?.paper != null;

  return (
    <>
      <BackgroundEffects />

      <Header
        connectionStatus={status}
        equity={data?.portfolio?.equity}
        killSwitch={data?.portfolio?.kill_switch}
        orderCount={data?.orders?.length ?? 0}
        paperActive={paperActive}
      />

      <main className="relative z-10 max-w-[1440px] mx-auto px-4 sm:px-6 py-4 space-y-4">
        {/* Portfolio */}
        <PortfolioSummary
          portfolio={data?.portfolio}
          equityHistory={data?.equity_history}
        />

        {/* Paper Trading */}
        {paperActive && <PaperTradingPanel paper={data?.paper ?? null} />}

        {/* Two-column section: Market + Agents */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <MarketTicker prices={data?.prices} />
          <AgentSignalsGrid agents={data?.agents} />
        </div>

        {/* Council */}
        <CouncilDeliberation council={data?.council} />

        {/* Two-column: Risk + Deployment + Serve */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div className="lg:col-span-2">
            <RiskMetrics risk={data?.risk} />
          </div>
          <div className="space-y-4">
            <ModelDeployment
              deployment={data?.deployment}
              ppoLatency={data?.ppo_latency}
            />
            <RayServeHealth serveHealth={data?.serve_health} />
          </div>
        </div>

        {/* Trade Log */}
        <TradeLogTable orders={data?.orders} />

        {/* Sentinel-X */}
        <SentinelXStatus sentinelx={data?.sentinelx} />

        {/* System Log */}
        <SystemLog logs={data?.logs} />

        {/* Config */}
        <ConfigPanel config={config} />
      </main>

      {/* Footer */}
      <footer className="relative z-10 flex items-center justify-between px-6 py-3 border-t border-hud-border bg-hud-card/60 backdrop-blur-md text-[11px] text-hud-text-muted">
        <span className="font-mono text-[10px]">
          ÆGIS Trader v0.2.0
        </span>
        <span className="font-mono text-[10px]">
          {data?.updated_at
            ? `Updated: ${new Date(data.updated_at).toLocaleTimeString("en-US", { hour12: false })}`
            : "Waiting for data..."}
        </span>
      </footer>
    </>
  );
}
