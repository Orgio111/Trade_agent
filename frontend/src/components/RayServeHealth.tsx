"use client";

import { cn } from "@/lib/cn";
import type { ServeHealth } from "@/lib/types";

interface RayServeHealthProps {
  serveHealth?: ServeHealth | null;
}

export default function RayServeHealth({
  serveHealth,
}: RayServeHealthProps) {
  if (!serveHealth) {
    return (
      <section className="glass-card p-4 animate-hud-fade-in">
        <h2 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-hud-text-muted mb-3 flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-hud-text-muted" />
          Ray Serve Health
        </h2>
        <div className="flex items-center justify-center h-16 text-hud-text-muted text-sm">
          Not configured
        </div>
      </section>
    );
  }

  return (
    <section className="glass-card p-4 animate-hud-fade-in">
      <h2 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-hud-text-muted mb-3 flex items-center gap-2">
        <span className={cn(
          "w-1.5 h-1.5 rounded-full",
          serveHealth.reachable ? "bg-hud-green shadow-[0_0_6px_rgba(34,197,94,0.5)]" : "bg-hud-red shadow-[0_0_6px_rgba(239,68,68,0.5)]"
        )} />
        Ray Serve Health
      </h2>

      {/* URL & status */}
      <div className="flex items-center gap-3 mb-3 flex-wrap">
        {serveHealth.serve_url && (
          <div className="flex items-center gap-2">
            <span className="text-[10px] uppercase tracking-wider text-hud-text-muted font-semibold">
              Endpoint:
            </span>
            <span className="font-mono text-[11px] text-hud-text-secondary truncate max-w-[200px]">
              {serveHealth.serve_url}
            </span>
          </div>
        )}
        <div className="flex items-center gap-1.5">
          <span className={cn(
            "w-2 h-2 rounded-full",
            serveHealth.reachable ? "bg-hud-green shadow-[0_0_6px_rgba(34,197,94,0.5)]" : "bg-hud-red shadow-[0_0_6px_rgba(239,68,68,0.5)]"
          )} />
          <span className={cn(
            "text-[11px] font-semibold",
            serveHealth.reachable ? "text-hud-green" : "text-hud-red"
          )}>
            {serveHealth.reachable ? "Reachable" : "Unreachable"}
          </span>
        </div>
      </div>

      {/* Deployments */}
      {serveHealth.deployments && Object.keys(serveHealth.deployments).length > 0 ? (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          {Object.entries(serveHealth.deployments).map(([name, dep]) => {
            const status = (dep.status || "unknown").toLowerCase();
            const isOk =
              status === "ok" ||
              status === "ready" ||
              status === "healthy";
            return (
              <div
                key={name}
                className="p-2.5 rounded-lg border border-hud-border bg-hud-bg/50"
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-[11px] font-bold text-hud-text">
                    {name}
                  </span>
                  <span
                    className={cn(
                      "font-mono text-[10px] font-semibold",
                      isOk ? "text-hud-green" : "text-hud-red"
                    )}
                  >
                    {status}
                  </span>
                </div>
                {dep.version && (
                  <div className="text-[10px] font-mono text-hud-text-muted">
                    v{dep.version}
                  </div>
                )}
                {dep.error && (
                  <div className="text-[9px] text-hud-red mt-1 truncate">
                    {dep.error}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      ) : (
        <div className="text-[11px] text-hud-text-muted">
          No deployments found
        </div>
      )}
    </section>
  );
}
