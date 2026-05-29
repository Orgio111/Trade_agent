"use client";

import { cn } from "@/lib/cn";
import type { CouncilDecision } from "@/lib/types";

interface CouncilDeliberationProps {
  council?: Record<string, CouncilDecision>;
}

export default function CouncilDeliberation({
  council,
}: CouncilDeliberationProps) {
  if (!council || Object.keys(council).length === 0) {
    return (
      <section className="glass-card p-4 animate-hud-fade-in">
        <h2 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-hud-text-muted mb-3 flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-hud-orange shadow-[0_0_6px_rgba(249,115,22,0.5)]" />
          Council Deliberation
        </h2>
        <div className="flex items-center justify-center h-20 text-hud-text-muted text-sm">
          Awaiting council session...
        </div>
      </section>
    );
  }

  return (
    <section className="glass-card p-4 animate-hud-fade-in">
      <h2 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-hud-text-muted mb-3 flex items-center gap-2">
        <span className="w-1.5 h-1.5 rounded-full bg-hud-orange shadow-[0_0_6px_rgba(249,115,22,0.5)]" />
        Council Deliberation
      </h2>
      <div className="space-y-3">
        {Object.entries(council).map(([sym, decision]) => (
          <CouncilSymbolGroup key={sym} symbol={sym} decision={decision} />
        ))}
      </div>
    </section>
  );
}

/* ─── Sub-component ────────────────────────────────────────────────────── */
function CouncilSymbolGroup({
  symbol,
  decision,
}: {
  symbol: string;
  decision: CouncilDecision;
}) {
  const verdict = (decision.final_side || "HOLD").toUpperCase();
  const bullPct = ((decision.bull_score || 0) * 100).toFixed(0);
  const bearPct = ((decision.bear_score || 0) * 100).toFixed(0);
  const consensus = ((decision.consensus_score || 0) * 100).toFixed(0);

  return (
    <div className="p-3 rounded-lg border border-hud-border bg-hud-bg/50">
      {/* Header */}
      <div className="flex items-center gap-3 mb-3">
        <span className="text-sm font-bold text-hud-text">{symbol}</span>
        <span
          className={cn(
            "text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded",
            verdict === "BUY" && "bg-hud-green/15 text-hud-green",
            verdict === "SELL" && "bg-hud-red/15 text-hud-red",
            verdict === "HOLD" && "bg-hud-yellow/15 text-hud-yellow"
          )}
        >
          {verdict}
        </span>
        <span className="ml-auto font-mono text-[10px] text-hud-text-muted">
          Consensus: {consensus}%
        </span>
      </div>

      {/* Score bars */}
      <div className="space-y-1.5 mb-3">
        <ScoreBar label="Bull" value={bullPct} color="green" />
        <ScoreBar label="Bear" value={bearPct} color="red" />
      </div>

      {/* Rationale */}
      {decision.rationale && (
        <div className="text-[11px] text-hud-text-secondary leading-relaxed p-2 rounded bg-hud-bg/50 border border-hud-border mb-3">
          <span className="text-hud-blue font-semibold">Rationale:</span>{" "}
          {decision.rationale}
        </div>
      )}

      {/* Debate log */}
      {decision.debate_log && decision.debate_log.length > 0 && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          {decision.debate_log.map((entry, i) => {
            const pos = (entry.position || "HOLD").toUpperCase();
            return (
              <div
                key={i}
                className={cn(
                  "p-2.5 rounded-lg border border-hud-border bg-hud-bg/30",
                  pos === "BUY" && "border-l-2 border-l-hud-green",
                  pos === "SELL" && "border-l-2 border-l-hud-red",
                  pos === "HOLD" && "border-l-2 border-l-hud-yellow"
                )}
              >
                <div className="flex items-center gap-2 mb-1">
                  <span
                    className={cn(
                      "text-[10px] font-bold uppercase px-1.5 py-0.5 rounded",
                      pos === "BUY" && "text-hud-green bg-hud-green/10",
                      pos === "SELL" && "text-hud-red bg-hud-red/10",
                      pos === "HOLD" && "text-hud-yellow bg-hud-yellow/10"
                    )}
                  >
                    {pos}
                  </span>
                  <span className="text-[11px] font-semibold text-hud-text">
                    {entry.agent}
                  </span>
                  {entry.score != null && (
                    <span className="ml-auto font-mono text-[10px] text-hud-text-muted">
                      {entry.score.toFixed(2)}
                    </span>
                  )}
                </div>
                {entry.argument && (
                  <p className="text-[10px] text-hud-text-secondary leading-relaxed mb-1.5">
                    {entry.argument}
                  </p>
                )}
                {entry.supporting && entry.supporting.length > 0 && (
                  <div className="flex flex-wrap gap-1 mb-1">
                    {entry.supporting.slice(0, 3).map((f: string, j: number) => (
                      <span
                        key={j}
                        className="text-[9px] px-1.5 py-0.5 rounded bg-hud-green/10 border border-hud-green/20 text-hud-green"
                      >
                        ↑ {f}
                      </span>
                    ))}
                  </div>
                )}
                {entry.risks && entry.risks.length > 0 && (
                  <div className="flex flex-wrap gap-1">
                    {entry.risks.slice(0, 3).map((r: string, j: number) => (
                      <span
                        key={j}
                        className="text-[9px] px-1.5 py-0.5 rounded bg-hud-red/10 border border-hud-red/20 text-hud-red"
                      >
                        ↓ {r}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function ScoreBar({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color: "green" | "red";
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-[10px] font-semibold text-hud-text-muted w-10 text-right">
        {label}
      </span>
      <div className="flex-1 hud-progress-track">
        <div
          className={cn("hud-progress-fill", color === "green" ? "green" : "red")}
          style={{ width: `${value}%` }}
        />
      </div>
      <span className="font-mono text-[10px] text-hud-text-muted w-8 text-right">
        {value}%
      </span>
    </div>
  );
}
