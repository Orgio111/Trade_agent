"use client";

import type { RiskMetric } from "@/lib/types";

interface RiskMetricsProps {
  risk?: Record<string, RiskMetric>;
}

const fmtPct = (v?: number | null) => {
  if (v == null) return "—";
  return (v * 100).toFixed(2) + "%";
};

const fmtCurrency = (v?: number | null) => {
  if (v == null) return "$—";
  const abs = Math.abs(v);
  if (abs >= 1_000) return "$" + (abs / 1_000).toFixed(1) + "K";
  return "$" + abs.toFixed(2);
};

const fmtPrice = (v?: number | null) => {
  if (v == null) return "—";
  if (v >= 1000) return v.toFixed(2);
  if (v >= 1) return v.toFixed(4);
  return v.toFixed(6);
};

export default function RiskMetrics({ risk }: RiskMetricsProps) {
  if (!risk || Object.keys(risk).length === 0) {
    return (
      <section className="glass-card p-4 animate-hud-fade-in">
        <h2 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-hud-text-muted mb-3 flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-hud-green shadow-[0_0_6px_rgba(34,197,94,0.5)]" />
          Risk Metrics
        </h2>
        <div className="flex items-center justify-center h-20 text-hud-text-muted text-sm">
          No risk data yet
        </div>
      </section>
    );
  }

  // Calculate aggregate portfolio heat
  const maxPosition = Math.max(
    ...Object.values(risk).map((r) => r.position_size_usd || 0)
  );

  return (
    <section className="glass-card p-4 animate-hud-fade-in">
      <h2 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-hud-text-muted mb-3 flex items-center gap-2">
        <span className="w-1.5 h-1.5 rounded-full bg-hud-green shadow-[0_0_6px_rgba(34,197,94,0.5)]" />
        Risk Metrics
      </h2>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
        {Object.entries(risk).map(([sym, r]) => (
          <div
            key={sym}
            className="p-3 rounded-lg border border-hud-border bg-hud-bg/50"
          >
            <div className="text-[11px] font-bold text-hud-text-muted mb-2 uppercase tracking-wide">
              {sym}
            </div>
            <div className="space-y-1">
              <RiskRow label="VaR 95%" value={fmtPct(r.var_95)} />
              <RiskRow label="VaR 99%" value={fmtPct(r.var_99)} />
              <RiskRow label="CVaR 99%" value={fmtPct(r.cvar_99)} />
              <RiskRow label="Kelly" value={fmtPct(r.kelly_fractional)} />
              <RiskRow
                label="Position"
                value={fmtCurrency(r.position_size_usd)}
              />
              <RiskRow label="Stop Loss" value={fmtPrice(r.stop_loss_price)} />
              <RiskRow
                label="Take Profit"
                value={fmtPrice(r.take_profit_price)}
              />
            </div>
          </div>
        ))}
      </div>

      {/* Portfolio heat bar */}
      {maxPosition > 0 && (
        <div className="mt-3 pt-3 border-t border-hud-border">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-[10px] uppercase tracking-wider text-hud-text-muted font-semibold">
              Portfolio Heat
            </span>
            <span className="font-mono text-[10px] text-hud-text-secondary">
              {fmtCurrency(maxPosition)}
            </span>
          </div>
          <div className="hud-progress-track">
            <div
              className="hud-progress-fill"
              style={{
                width: `${Math.min((maxPosition / 100_000) * 100, 100)}%`,
              }}
            />
          </div>
        </div>
      )}
    </section>
  );
}

function RiskRow({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-[10px] text-hud-text-muted">{label}</span>
      <span className="font-mono text-[11px] font-medium text-hud-text tabular-nums">
        {value}
      </span>
    </div>
  );
}
