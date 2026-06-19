import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

async function getBackendBase(): Promise<string> {
  return (process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8001").replace(/\/+$/, "");
}

export async function GET() {
  try {
    const base = await getBackendBase();
    const res = await fetch(`${base}/health`);
    const body = await res.text();
    return new NextResponse(body, {
      status: res.status,
      headers: { "content-type": res.headers.get("content-type") || "application/json" },
    });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : "unknown_error";
    return NextResponse.json({ status: "backend_unreachable", error: message }, { status: 502 });
  }
}
