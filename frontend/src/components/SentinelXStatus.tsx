"use client";

import { cn } from "@/lib/cn";
import type { SentinelXState } from "@/lib/types";

interface SentinelXStatusProps {
  sentinelx?: SentinelXState;
}

export default function SentinelXStatus({
  sentinelx,
}: SentinelXStatusProps) {
  if (!sentinelx) return null;

  const cb = sentinelx.circuit_breaker || "CLOSED";
  const ksActive = sentinelx.rust_ks_active;
  const heat = sentinelx.rust_portfolio_heat;
  const heatPct = heat != null ? Math.min(heat * 100, 100) : 0;

  return (
    <section className="glass-card p-4 animate-hud-fade-in col-span-full">
      <h2 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-hud-text-muted mb-3 flex items-center gap-2">
        <span className="w-1.5 h-1.5 rounded-full bg-hud-indigo shadow-[0_0_6px_rgba(99,102,241,0.5)]" />
        Sentinel-X Status
      </h2>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        {/* GPU Circuit Breaker */}
        <div className="p-3 rounded-lg border border-hud-border bg-hud-bg/50">
          <span className="text-[10px] uppercase tracking-wider text-hud-text-muted font-semibold mb-2 block">
            GPU Circuit Breaker
          </span>
          <div className="flex items-center gap-2">
            <span
              className={cn(
                "w-2 h-2 rounded-full",
                cb === "CLOSED"
                  ? "bg-hud-green shadow-[0_0_8px_rgba(34,197,94,0.5)]"
                  : "bg-hud-red shadow-[0_0_8px_rgba(239,68,68,0.5)]"
              )}
            />
            <span className="font-mono text-sm font-bold text-hud-text">
              {cb}
            </span>
          </div>
        </div>

        {/* Rust Kill Switch */}
        <div className="p-3 rounded-lg border border-hud-border bg-hud-bg/50">
          <span className="text-[10px] uppercase tracking-wider text-hud-text-muted font-semibold mb-2 block">
            Rust Kill Switch
          </span>
          <div className="flex items-center gap-2">
            <span
              className={cn(
                "w-2 h-2 rounded-full",
                ksActive
                  ? "bg-hud-red shadow-[0_0_8px_rgba(239,68,68,0.5)]"
                  : "bg-hud-green shadow-[0_0_8px_rgba(34,197,94,0.5)]"
              )}
            />
            <span
              className={cn(
                "font-mono text-sm font-bold",
                ksActive ? "text-hud-red" : "text-hud-green"
              )}
            >
              {ksActive ? "ACTIVE" : "INACTIVE"}
            </span>
          </div>
        </div>

        {/* Portfolio Heat */}
        <div className="p-3 rounded-lg border border-hud-border bg-hud-bg/50">
          <span className="text-[10px] uppercase tracking-wider text-hud-text-muted font-semibold mb-2 block">
            Portfolio Heat
          </span>
          <div className="font-mono text-sm font-bold text-hud-text mb-2">
            {heat != null ? (heat * 100).toFixed(1) + "%" : "—"}
          </div>
          <div className="hud-progress-track">
            <div
              className={cn(
                "hud-progress-fill",
                heatPct > 70 ? "red" : heatPct > 40 ? "yellow" : "cyan"
              )}
              style={{ width: `${heatPct}%` }}
            />
          </div>
        </div>
      </div>
    </section>
  );
}
