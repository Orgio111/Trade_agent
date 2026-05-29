"use client";

import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
} from "chart.js";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip
);

interface EquityChartProps {
  data: { t: string; v: number }[];
}

export default function EquityChart({ data }: EquityChartProps) {
  if (!data || data.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-hud-text-muted text-xs">
        No data yet
      </div>
    );
  }

  const chartData = {
    labels: data.map((d) => {
      try {
        return new Date(d.t).toLocaleTimeString("en-US", { hour12: false });
      } catch {
        return "";
      }
    }),
    datasets: [
      {
        data: data.map((d) => d.v),
        borderColor: "#6366f1",
        backgroundColor: (ctx: { chart: { ctx: CanvasRenderingContext2D } }) => {
          const gradient = ctx.chart.ctx.createLinearGradient(0, 0, 0, 180);
          gradient.addColorStop(0, "rgba(99, 102, 241, 0.2)");
          gradient.addColorStop(1, "rgba(99, 102, 241, 0.01)");
          return gradient;
        },
        fill: true,
        tension: 0.4,
        pointRadius: 0,
        borderWidth: 2,
      },
    ],
  };

  return (
    <Line
      data={chartData}
      options={{
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 300 },
        plugins: { legend: { display: false }, tooltip: { enabled: true, mode: "index", intersect: false } },
        scales: {
          x: { display: false, grid: { display: false } },
          y: {
            display: true,
            grid: { color: "rgba(99, 102, 241, 0.06)" },
            ticks: {
              color: "#64748b",
              font: { size: 9, family: "'JetBrains Mono', monospace" },
              callback: (v: unknown) => "$" + Number(v).toFixed(0),
            },
          },
        },
        interaction: { intersect: false, mode: "index" },
      }}
      height={180}
    />
  );
}
