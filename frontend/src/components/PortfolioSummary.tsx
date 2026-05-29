"use client";

import { cn } from "@/lib/cn";
import type { PortfolioData } from "@/lib/types";
import dynamic from "next/dynamic";

const EquityChart = dynamic(() => import("@/components/EquityChart"), {
  ssr: false,
  loading: () => <div className="h-full flex items-center justify-center text-hud-text-muted text-xs">Loading chart...</div>,
});

/* ─── Formatting helpers ────────────────────────────────────────────────── */
const fmtCurrency = (v?: number | null) => {
  if (v == null) return "$—";
  const abs = Math.abs(v);
  const sign = v < 0 ? "-" : "";
  if (abs >= 1_000_000) return sign + "$" + (abs / 1_000_000).toFixed(2) + "M";
  if (abs >= 1_000) return sign + "$" + (abs / 1_000).toFixed(1) + "K";
  return sign + "$" + abs.toFixed(2);
};

const fmtPct = (v?: number | null) => {
  if (v == null) return "—%";
  return (v >= 0 ? "+" : "") + v.toFixed(2) + "%";
};

interface PortfolioSummaryProps {
  portfolio?: PortfolioData;
  equityHistory?: { t: string; v: number }[];
}

export default function PortfolioSummary({
  portfolio,
  equityHistory,
}: PortfolioSummaryProps) {
  if (!portfolio) {
    return (
      <GlassCard title="Portfolio">
        <div className="flex items-center justify-center h-32 text-hud-text-muted text-sm">
          Waiting for portfolio data...
        </div>
      </GlassCard>
    );
  }

  const { equity, cash, daily_pnl, drawdown_pct, peak_equity, kill_switch } =
    portfolio;

  return (
    <GlassCard title="Portfolio">
      {/* Metric grid */}
      <div className="grid grid-cols-3 lg:grid-cols-6 gap-3 mb-4">
        <MetricBox
          label="Equity"
          value={fmtCurrency(equity)}
          positive={equity >= 0}
          neon
        />
        <MetricBox label="Cash" value={fmtCurrency(cash)} />
        <MetricBox
          label="Daily P&L"
          value={fmtCurrency(daily_pnl)}
          positive={daily_pnl >= 0}
        />
        <MetricBox
          label="Drawdown"
          value={fmtPct(-drawdown_pct)}
          positive={drawdown_pct <= 0}
        />
        <MetricBox label="Peak" value={fmtCurrency(peak_equity)} />
        <MetricBox
          label="Kill Switch"
          value={kill_switch ? "● ON" : "● OFF"}
          positive={!kill_switch}
        />
      </div>

      {/* Equity chart */}
      <div className="h-44 border-t border-hud-border pt-3">
        <EquityChart data={equityHistory || []} />
      </div>
    </GlassCard>
  );
}

/* ─── Sub-components ────────────────────────────────────────────────────── */
function GlassCard({
  title,
  children,
  className,
}: {
  title: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section
      className={cn(
        "glass-card p-4 animate-hud-fade-in",
        "col-span-full",
        className
      )}
    >
      <h2 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-hud-text-muted mb-4 flex items-center gap-2">
        <span className="w-1.5 h-1.5 rounded-full bg-hud-indigo shadow-[0_0_6px_rgba(99,102,241,0.5)]" />
        {title}
      </h2>
      {children}
    </section>
  );
}

function MetricBox({
  label,
  value,
  positive,
  neon,
}: {
  label: string;
  value: string;
  positive?: boolean;
  neon?: boolean;
}) {
  return (
    <div className="flex flex-col gap-1 p-3 rounded-lg border border-hud-border bg-hud-bg/50">
      <span className="text-[10px] uppercase tracking-widest text-hud-text-muted font-semibold">
        {label}
      </span>
      <span
        className={cn(
          "font-mono text-sm font-bold tabular-nums tracking-tight",
          positive === undefined && "text-hud-text",
          positive === true && "text-hud-green",
          positive === false && "text-hud-red",
          neon && positive !== false && "neon-cyan",
          neon && positive === false && "neon-red"
        )}
      >
        {value}
      </span>
    </div>
  );
}

export { GlassCard, MetricBox, fmtCurrency, fmtPct };
