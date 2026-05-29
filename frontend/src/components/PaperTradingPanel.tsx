"use client";

import { cn } from "@/lib/cn";
import type { PaperState } from "@/lib/types";

interface PaperTradingPanelProps {
  paper?: PaperState | null;
}

const fmtCurrency = (v?: number | null) => {
  if (v == null) return "$—";
  const abs = Math.abs(v);
  const sign = v < 0 ? "-" : "";
  if (abs >= 1_000_000) return sign + "$" + (abs / 1_000_000).toFixed(2) + "M";
  if (abs >= 1_000) return sign + "$" + (abs / 1_000).toFixed(1) + "K";
  return sign + "$" + abs.toFixed(2);
};

const fmtPrice = (v?: number | null) => {
  if (v == null) return "—";
  if (v >= 1000) return v.toFixed(2);
  if (v >= 1) return v.toFixed(4);
  return v.toFixed(6);
};

export default function PaperTradingPanel({
  paper,
}: PaperTradingPanelProps) {
  if (!paper) return null;

  return (
    <section className="glass-card p-4 animate-hud-fade-in col-span-full">
      <h2 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-hud-text-muted mb-3 flex items-center gap-2">
        <span className="w-1.5 h-1.5 rounded-full bg-hud-yellow shadow-[0_0_6px_rgba(234,179,8,0.5)]" />
        Paper Trading
        <span className="ml-1 text-[10px] font-semibold uppercase tracking-wider px-1.5 py-0.5 rounded bg-hud-yellow/10 text-hud-yellow border border-hud-yellow/20">
          PAPER
        </span>
      </h2>

      {/* Metrics */}
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-2 mb-4">
        <MetricBox label="Equity" value={fmtCurrency(paper.equity)} />
        <MetricBox label="Cash" value={fmtCurrency(paper.cash)} />
        <MetricBox
          label="Total P&L"
          value={fmtCurrency(paper.total_pnl)}
          positive={(paper.total_pnl || 0) >= 0}
        />
        <MetricBox
          label="Daily P&L"
          value={fmtCurrency(paper.daily_pnl)}
          positive={(paper.daily_pnl || 0) >= 0}
        />
        <MetricBox
          label="Win Rate"
          value={
            paper.win_rate != null
              ? (paper.win_rate * 100).toFixed(1) + "%"
              : "—"
          }
        />
        <MetricBox
          label="Sharpe"
          value={paper.sharpe != null ? paper.sharpe.toFixed(2) : "—"}
        />
        <MetricBox
          label="Max DD"
          value={
            paper.max_drawdown != null
              ? (paper.max_drawdown * 100).toFixed(2) + "%"
              : "—"
          }
        />
        <MetricBox
          label="Trades"
          value={String(paper.total_trades ?? "—")}
        />
      </div>

      {/* Positions */}
      {paper.positions && paper.positions.length > 0 && (
        <>
          <div className="flex items-center gap-2 mb-2">
            <span className="text-[10px] uppercase tracking-wider text-hud-text-muted font-semibold">
              Open Positions
            </span>
            <span className="font-mono text-[10px] text-hud-text-muted bg-hud-bg/50 px-1.5 py-0.5 rounded">
              {paper.positions.length}
            </span>
          </div>
          <div className="max-h-32 overflow-y-auto space-y-1">
            {paper.positions.map((pos, i) => (
              <div
                key={i}
                className="flex items-center gap-3 px-3 py-1.5 rounded-lg border border-hud-border bg-hud-bg/50 font-mono text-[11px]"
              >
                <span className="font-bold text-hud-text flex-1">
                  {pos.symbol}
                </span>
                <span className="text-hud-text-secondary">{pos.quantity}</span>
                <span className="text-hud-text-muted">
                  @ {fmtPrice(pos.entry_price)}
                </span>
                <span className="text-hud-text-secondary">
                  {fmtCurrency(pos.current_value)}
                </span>
              </div>
            ))}
          </div>
        </>
      )}
    </section>
  );
}

function MetricBox({
  label,
  value,
  positive,
}: {
  label: string;
  value: string;
  positive?: boolean;
}) {
  return (
    <div className="flex flex-col gap-0.5 p-2.5 rounded-lg border border-hud-border bg-hud-bg/50">
      <span className="text-[9px] uppercase tracking-wider text-hud-text-muted font-semibold">
        {label}
      </span>
      <span
        className={cn(
          "font-mono text-xs font-bold tabular-nums",
          positive === undefined && "text-hud-text",
          positive === true && "text-hud-green",
          positive === false && "text-hud-red"
        )}
      >
        {value}
      </span>
    </div>
  );
}
