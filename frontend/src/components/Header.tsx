"use client";

import React from "react";
import { cn } from "@/lib/cn";
import type { ConnectionStatus } from "@/lib/use-trading-socket";

/* ─── Formatting helpers ────────────────────────────────────────────────── */
const fmtCurrency = (v?: number | null) => {
  if (v == null) return "$—";
  const abs = Math.abs(v);
  const sign = v < 0 ? "-" : "";
  if (abs >= 1_000_000) return sign + "$" + (abs / 1_000_000).toFixed(2) + "M";
  if (abs >= 1_000) return sign + "$" + (abs / 1_000).toFixed(1) + "K";
  return sign + "$" + abs.toFixed(2);
};

/* ─── Props ─────────────────────────────────────────────────────────────── */
interface HeaderProps {
  connectionStatus: ConnectionStatus;
  equity?: number;
  killSwitch?: boolean;
  orderCount?: number;
  paperActive?: boolean;
}

/* ─── Component ─────────────────────────────────────────────────────────── */
export default function Header({
  connectionStatus,
  equity,
  killSwitch,
  orderCount = 0,
  paperActive,
}: HeaderProps) {
  const statusColor = {
    connecting: "text-hud-yellow animate-hud-pulse",
    connected: "text-hud-green",
    disconnected: "text-hud-red animate-hud-pulse",
    error: "text-hud-red",
  }[connectionStatus];

  const statusLabel = {
    connecting: "CONNECTING",
    connected: "LIVE",
    disconnected: "DISCONNECTED",
    error: "ERROR",
  }[connectionStatus];

  const [time, setTime] = React.useState("--:--:--");

  React.useEffect(() => {
    const tick = () =>
      setTime(
        new Date().toLocaleTimeString("en-US", { hour12: false })
      );
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);

  return (
    <header className="relative z-10 flex items-center justify-between px-6 py-3 border-b border-hud-border bg-hud-card/80 backdrop-blur-xl">
      {/* Left: Logo */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-3">
          <svg width={28} height={28} viewBox="0 0 28 28" fill="none">
            <rect
              x={2}
              y={2}
              width={24}
              height={24}
              rx={6}
              stroke="url(#logo-grad)"
              strokeWidth={2}
            />
            <path
              d="M8 18L12 10L16 14L20 8"
              stroke="url(#logo-grad)"
              strokeWidth={2}
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            <defs>
              <linearGradient id="logo-grad" x1="0" y1="0" x2="28" y2="28">
                <stop stopColor="#6366f1" />
                <stop offset="1" stopColor="#06b6d4" />
              </linearGradient>
            </defs>
          </svg>
          <div>
            <h1 className="text-sm font-bold tracking-tight text-hud-text">
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-hud-indigo to-hud-cyan">
                ÆGIS
              </span>{" "}
              <span className="text-hud-text-muted font-normal">Trader</span>
            </h1>
            <p className="text-[10px] text-hud-text-muted font-mono uppercase tracking-widest">
              Multi-Agent Trading System
            </p>
          </div>
        </div>
        {/* Status badge */}
        <div className={cn("flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-semibold uppercase tracking-wider border", 
          connectionStatus === "connected" ? "bg-hud-green/10 border-hud-green/20 text-hud-green" :
          connectionStatus === "connecting" ? "bg-hud-yellow/10 border-hud-yellow/20 text-hud-yellow" :
          "bg-hud-red/10 border-hud-red/20 text-hud-red"
        )}>
          <span className={cn("w-1.5 h-1.5 rounded-full",
            connectionStatus === "connected" ? "bg-hud-green shadow-[0_0_6px_rgba(34,197,94,0.5)]" :
            connectionStatus === "connecting" ? "bg-hud-yellow shadow-[0_0_6px_rgba(234,179,8,0.5)]" :
            "bg-hud-red shadow-[0_0_6px_rgba(239,68,68,0.5)]"
          )} />
          {statusLabel}
        </div>
      </div>

      {/* Right: Metrics + Time */}
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-4">
          <HeaderMetric label="Equity" value={fmtCurrency(equity)} className={equity && equity >= 0 ? "text-hud-green" : "text-hud-text"} />
          <HeaderMetric label="Orders" value={String(orderCount)} />
          {paperActive && (
            <div className="flex items-center gap-1.5 px-2 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wider bg-hud-yellow/10 border border-hud-yellow/20 text-hud-yellow">
              <span className="w-1.5 h-1.5 rounded-full bg-hud-yellow shadow-[0_0_6px_rgba(234,179,8,0.5)]" />
              Paper
            </div>
          )}
          {killSwitch && (
            <div className="flex items-center gap-1.5 px-2 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wider bg-hud-red/10 border border-hud-red/20 text-hud-red">
              <span className="w-1.5 h-1.5 rounded-full bg-hud-red shadow-[0_0_6px_rgba(239,68,68,0.5)] animate-hud-pulse" />
              HALT
            </div>
          )}
        </div>
        <div className="text-xs font-mono text-hud-text-muted tracking-widest">
          {time} <span className="text-hud-text-muted/50">UTC</span>
        </div>
      </div>
    </header>
  );
}

/* ─── Sub-component ─────────────────────────────────────────────────────── */
function HeaderMetric({
  label,
  value,
  className,
}: {
  label: string;
  value: string;
  className?: string;
}) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-[10px] uppercase tracking-widest text-hud-text-muted font-semibold">
        {label}
      </span>
      <span className={cn("text-xs font-mono font-semibold", className || "text-hud-text")}>
        {value}
      </span>
    </div>
  );
}

