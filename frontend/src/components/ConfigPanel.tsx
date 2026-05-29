"use client";

import type { AppConfig } from "@/lib/types";

interface ConfigPanelProps {
  config?: AppConfig | null;
}

const fmtCurrency = (v?: number | null) => {
  if (v == null) return "$—";
  const abs = Math.abs(v);
  if (abs >= 1_000_000) return "$" + (abs / 1_000_000).toFixed(2) + "M";
  if (abs >= 1_000) return "$" + (abs / 1_000).toFixed(1) + "K";
  return "$" + abs.toFixed(2);
};

export default function ConfigPanel({ config }: ConfigPanelProps) {
  if (!config) return null;

  const items = [
    { label: "Exchange", value: config.exchange || "—" },
    { label: "Symbols", value: (config.symbols || []).join(", ") },
    {
      label: "Paper Trading",
      value: config.paper_trading ? "Enabled" : "Disabled",
    },
    { label: "Capital", value: fmtCurrency(config.initial_capital) },
    {
      label: "Min Consensus",
      value:
        config.min_consensus_score != null
          ? (config.min_consensus_score * 100).toFixed(0) + "%"
          : "—",
    },
    {
      label: "Max Drawdown",
      value:
        config.max_daily_drawdown_pct != null
          ? (config.max_daily_drawdown_pct * 100).toFixed(1) + "%"
          : "—",
    },
    {
      label: "Kelly Fraction",
      value:
        config.kelly_fraction != null
          ? (config.kelly_fraction * 100).toFixed(0) + "%"
          : "—",
    },
  ];

  return (
    <section className="glass-card p-4 animate-hud-fade-in">
      <h2 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-hud-text-muted mb-3 flex items-center gap-2">
        <span className="w-1.5 h-1.5 rounded-full bg-hud-text-muted" />
        Configuration
      </h2>
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-7 gap-2">
        {items.map((item) => (
          <div
            key={item.label}
            className="flex flex-col gap-0.5 p-2 rounded-lg border border-hud-border bg-hud-bg/50"
          >
            <span className="text-[9px] uppercase tracking-wider text-hud-text-muted font-semibold">
              {item.label}
            </span>
            <span className="font-mono text-[11px] font-medium text-hud-text truncate">
              {item.value}
            </span>
          </div>
        ))}
      </div>
    </section>
  );
}
