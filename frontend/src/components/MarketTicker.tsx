"use client";

import { cn } from "@/lib/cn";

interface MarketTickerProps {
  prices?: Record<string, number>;
}

const fmtPrice = (v?: number | null) => {
  if (v == null) return "—";
  if (v >= 1000) return v.toFixed(2);
  if (v >= 1) return v.toFixed(4);
  return v.toFixed(6);
};

const fmtPct = (v?: number | null) => {
  if (v == null) return "—";
  return (v >= 0 ? "+" : "") + v.toFixed(2) + "%";
};

const symbolColors: Record<string, string> = {
  BTC: "text-hud-yellow",
  ETH: "text-hud-purple",
  SOL: "text-hud-cyan",
  AVAX: "text-hud-red",
  LINK: "text-hud-blue",
};

export default function MarketTicker({ prices }: MarketTickerProps) {
  if (!prices || Object.keys(prices).length === 0) {
    return (
      <MarketCard title="Market">
        <div className="flex items-center justify-center h-20 text-hud-text-muted text-sm">
          Waiting for prices...
        </div>
      </MarketCard>
    );
  }

  return (
    <MarketCard title="Market">
      <div className="space-y-1.5">
        {Object.entries(prices).map(([sym, price]) => {
          const color =
            symbolColors[sym.replace(/USD.*$/, "")] || "text-hud-text";
          return (
            <div
              key={sym}
              className="flex items-center justify-between px-3 py-2 rounded-lg border border-hud-border bg-hud-bg/50 transition-colors hover:border-hud-border-hover"
            >
              <span className={cn("text-xs font-bold tracking-wide", color)}>
                {sym}
              </span>
              <span className="font-mono text-sm font-medium text-hud-text tabular-nums">
                {fmtPrice(price)}
              </span>
            </div>
          );
        })}
      </div>
    </MarketCard>
  );
}

function MarketCard({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="glass-card p-4 animate-hud-fade-in">
      <h2 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-hud-text-muted mb-3 flex items-center gap-2">
        <span className="w-1.5 h-1.5 rounded-full bg-hud-cyan shadow-[0_0_6px_rgba(6,182,212,0.5)]" />
        {title}
      </h2>
      {children}
    </section>
  );
}
