export const runtime = "nodejs";

export const NEXT_PUBLIC_BACKEND_URL =
  (process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8001").replace(/\/+$/, "");

export function apiUrl(path: string) {
  const p = path.startsWith("/") ? path : `/${path}`;
  return `${NEXT_PUBLIC_BACKEND_URL}${p}`;
}

export function wsBase() {
  try {
    const url = new URL(NEXT_PUBLIC_BACKEND_URL);
    const proto = url.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${url.host}/ws`;
  } catch {
    return "ws://localhost:8001/ws";
  }
}
