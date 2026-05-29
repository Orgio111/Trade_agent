"use client";

import { cn } from "@/lib/cn";
import type { LogEntry } from "@/lib/types";

interface SystemLogProps {
  logs?: LogEntry[];
}

const fmtTime = (iso?: string) => {
  if (!iso) return "";
  try {
    return new Date(iso).toLocaleTimeString("en-US", { hour12: false });
  } catch {
    return "";
  }
};

export default function SystemLog({ logs }: SystemLogProps) {
  const arr = logs || [];

  return (
    <section className="glass-card p-4 animate-hud-fade-in col-span-full">
      <h2 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-hud-text-muted mb-3 flex items-center gap-2">
        <span className="w-1.5 h-1.5 rounded-full bg-hud-text-muted shadow-[0_0_6px_rgba(100,116,139,0.5)]" />
        System Log
        <span className="ml-1 font-mono text-[10px] text-hud-text-muted bg-hud-bg/50 px-1.5 py-0.5 rounded">
          {arr.length}
        </span>
      </h2>

      {arr.length === 0 ? (
        <div className="flex items-center justify-center h-16 text-hud-text-muted text-sm">
          No log entries yet
        </div>
      ) : (
        <div className="max-h-60 overflow-y-auto font-mono text-[11px] space-y-0.5">
          {[...arr].reverse().slice(0, 100).map((entry, i) => {
            const level = (entry.level || "INFO").toLowerCase();
            return (
              <div
                key={i}
                className={cn(
                  "flex items-center gap-2 px-2 py-1 rounded transition-colors hover:bg-hud-card/50",
                  level === "error" && "bg-hud-red/5"
                )}
              >
                <span className="text-hud-text-muted flex-shrink-0 w-16">
                  {fmtTime(entry.time)}
                </span>
                <span
                  className={cn(
                    "flex-shrink-0 text-[9px] font-bold uppercase px-1 py-0.5 rounded w-12 text-center",
                    level === "info" && "bg-hud-blue/10 text-hud-blue",
                    level === "warning" && "bg-hud-yellow/10 text-hud-yellow",
                    level === "error" && "bg-hud-red/10 text-hud-red",
                    level === "debug" && "bg-hud-purple/10 text-hud-purple"
                  )}
                >
                  {entry.level || "INFO"}
                </span>
                <span className="text-hud-text-secondary truncate">
                  {entry.msg}
                </span>
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}
