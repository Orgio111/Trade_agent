/* ─── Dashboard API Client ──────────────────────────────────────────────── */

import type { DashboardSnapshot, ServeHealth, AppConfig } from "@/lib/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

async function fetchJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`API ${path}: ${res.status}`);
  return res.json();
}

export async function fetchSnapshot(): Promise<DashboardSnapshot> {
  return fetchJson<DashboardSnapshot>("/api/snapshot");
}

export async function fetchServeHealth(): Promise<ServeHealth> {
  return fetchJson<ServeHealth>("/api/serve-status");
}

export async function fetchConfig(): Promise<AppConfig> {
  return fetchJson<AppConfig>("/api/config");
}
