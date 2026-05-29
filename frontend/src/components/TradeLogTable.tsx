"use client";

import { cn } from "@/lib/cn";
import type { OrderData } from "@/lib/types";

interface TradeLogTableProps {
  orders?: OrderData[];
}

const fmtTime = (iso?: string) => {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleTimeString("en-US", { hour12: false });
  } catch {
    return iso;
  }
};

const fmtPrice = (v?: number | null) => {
  if (v == null) return "—";
  if (v >= 1000) return v.toFixed(2);
  if (v >= 1) return v.toFixed(4);
  return v.toFixed(6);
};

export default function TradeLogTable({ orders }: TradeLogTableProps) {
  const arr = orders || [];

  return (
    <section className="glass-card p-4 animate-hud-fade-in col-span-full">
      <h2 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-hud-text-muted mb-3 flex items-center gap-2">
        <span className="w-1.5 h-1.5 rounded-full bg-hud-blue shadow-[0_0_6px_rgba(59,130,246,0.5)]" />
        Trade Log
        <span className="ml-1 font-mono text-[10px] text-hud-text-muted bg-hud-bg/50 px-1.5 py-0.5 rounded">
          {arr.length}
        </span>
      </h2>

      {arr.length === 0 ? (
        <div className="flex items-center justify-center h-16 text-hud-text-muted text-sm">
          No trades yet
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-hud-border">
                <Th>Time</Th>
                <Th>Symbol</Th>
                <Th>Side</Th>
                <Th>Qty</Th>
                <Th>Price</Th>
                <Th>Type</Th>
                <Th>Status</Th>
              </tr>
            </thead>
            <tbody>
              {arr.slice(0, 50).map((o, i) => {
                const side = (o.side || "").toUpperCase();
                const status = (o.status || "").toUpperCase();
                return (
                  <tr
                    key={o.order_id || i}
                    className="border-b border-hud-border/50 transition-colors hover:bg-hud-card/50"
                  >
                    <Td>{fmtTime(o.created_at)}</Td>
                    <Td className="font-semibold">{o.symbol || ""}</Td>
                    <Td>
                      <span
                        className={cn(
                          "font-bold uppercase text-[10px]",
                          side === "BUY" && "trade-buy",
                          side === "SELL" && "trade-sell",
                          side === "HOLD" && "trade-hold"
                        )}
                      >
                        {side}
                      </span>
                    </Td>
                    <Td className="font-mono">{o.quantity || 0}</Td>
                    <Td className="font-mono">{fmtPrice(o.price)}</Td>
                    <Td className="text-hud-text-muted">{o.type || ""}</Td>
                    <Td>
                      <span
                        className={cn(
                          "text-[10px] font-semibold",
                          status === "FILLED" && "text-hud-green",
                          status === "REJECTED" && "text-hud-red",
                          status === "PENDING" && "text-hud-yellow",
                          !["FILLED", "REJECTED", "PENDING"].includes(status) &&
                            "text-hud-text-muted"
                        )}
                      >
                        {status}
                      </span>
                    </Td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function Th({ children }: { children: React.ReactNode }) {
  return (
    <th className="text-left py-2 px-3 font-semibold text-[10px] uppercase tracking-wider text-hud-text-muted whitespace-nowrap">
      {children}
    </th>
  );
}

function Td({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <td className={cn("py-2 px-3 whitespace-nowrap text-hud-text", className)}>
      {children}
    </td>
  );
}
