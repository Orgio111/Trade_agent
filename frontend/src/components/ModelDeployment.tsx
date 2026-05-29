"use client";

import type { ModelDeployment } from "@/lib/types";

interface ModelDeploymentProps {
  deployment?: ModelDeployment;
  ppoLatency?: { t: string; v: number }[];
}

export default function ModelDeployment({
  deployment,
  ppoLatency,
}: ModelDeploymentProps) {
  if (!deployment) return null;

  const latencyAvg =
    ppoLatency && ppoLatency.length > 0
      ? ppoLatency.reduce((s, p) => s + p.v, 0) / ppoLatency.length
      : 0;

  const latencyMax = ppoLatency
    ? Math.max(...ppoLatency.map((p) => p.v), 0)
    : 0;

  return (
    <section className="glass-card p-4 animate-hud-fade-in">
      <h2 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-hud-text-muted mb-3 flex items-center gap-2">
        <span className="w-1.5 h-1.5 rounded-full bg-hud-cyan shadow-[0_0_6px_rgba(6,182,212,0.5)]" />
        Model Deployment
      </h2>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-3">
        <DeploymentMetric label="Version" value={String(deployment.version ?? "—")} />
        <DeploymentMetric label="Source" value={deployment.source || "—"} />
        <DeploymentMetric
          label="Loaded"
          value={deployment.model_loaded ? "YES" : "NO"}
          positive={deployment.model_loaded}
        />
        <DeploymentMetric
          label="Registry"
          value={
            deployment.latest_registry_version
              ? "v" + deployment.latest_registry_version
              : "v" + (deployment.local_version || "?")
          }
        />
      </div>

      {/* PPO Latency */}
      <div className="border-t border-hud-border pt-3">
        <div className="text-[10px] uppercase tracking-wider text-hud-text-muted font-semibold mb-2">
          PPO Inference Latency
        </div>
        {ppoLatency && ppoLatency.length > 0 ? (
          <div className="flex gap-4">
            <div className="flex flex-col gap-0.5">
              <span className="text-[9px] text-hud-text-muted">Avg</span>
              <span className="font-mono text-xs font-bold text-hud-text">
                {latencyAvg.toFixed(1)}ms
              </span>
            </div>
            <div className="flex flex-col gap-0.5">
              <span className="text-[9px] text-hud-text-muted">Max</span>
              <span className="font-mono text-xs font-bold text-hud-text">
                {latencyMax.toFixed(1)}ms
              </span>
            </div>
            <div className="flex flex-col gap-0.5">
              <span className="text-[9px] text-hud-text-muted">Samples</span>
              <span className="font-mono text-xs font-bold text-hud-text">
                {ppoLatency.length}
              </span>
            </div>
          </div>
        ) : (
          <div className="text-[11px] text-hud-text-muted">
            No latency data yet
          </div>
        )}
      </div>
    </section>
  );
}

function DeploymentMetric({
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
        className={`font-mono text-sm font-bold tabular-nums ${
          positive === undefined
            ? "text-hud-text"
            : positive
            ? "text-hud-green"
            : "text-hud-red"
        }`}
      >
        {value}
      </span>
    </div>
  );
}
