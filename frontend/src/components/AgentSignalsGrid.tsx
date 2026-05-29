"use client";

import { cn } from "@/lib/cn";
import type { AgentSignal } from "@/lib/types";

interface AgentSignalsGridProps {
  agents?: Record<string, AgentSignal>;
}

export default function AgentSignalsGrid({
  agents,
}: AgentSignalsGridProps) {
  if (!agents || Object.keys(agents).length === 0) {
    return (
      <AgentCard title="Agent Signals">
        <div className="flex items-center justify-center h-20 text-hud-text-muted text-sm">
          Awaiting agent signals...
        </div>
      </AgentCard>
    );
  }

  return (
    <AgentCard title="Agent Signals">
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
        {Object.entries(agents).map(([name, signal]) => {
          const side = (signal.side || "HOLD").toUpperCase();
          return (
            <div
              key={name}
              className="flex flex-col gap-1.5 p-3 rounded-lg border border-hud-border bg-hud-bg/50 transition-colors hover:border-hud-border-hover"
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-hud-text capitalize">
                  {name.replace(/_/g, " ")}
                </span>
                {signal.confidence != null && (
                  <span className="font-mono text-[10px] text-hud-text-muted">
                    {(signal.confidence * 100).toFixed(0)}%
                  </span>
                )}
              </div>
              <span
                className={cn(
                  "text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded w-fit",
                  side === "BUY" && "bg-hud-green/15 text-hud-green",
                  side === "SELL" && "bg-hud-red/15 text-hud-red",
                  side === "HOLD" && "bg-hud-yellow/15 text-hud-yellow"
                )}
              >
                {side}
              </span>
              {signal.status && (
                <span className="text-[10px] text-hud-text-muted leading-tight">
                  {signal.status}
                </span>
              )}
            </div>
          );
        })}
      </div>
    </AgentCard>
  );
}

function AgentCard({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="glass-card p-4 animate-hud-fade-in">
      <h2 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-hud-text-muted mb-3 flex items-center gap-2">
        <span className="w-1.5 h-1.5 rounded-full bg-hud-purple shadow-[0_0_6px_rgba(168,85,247,0.5)]" />
        {title}
      </h2>
      {children}
    </section>
  );
}
