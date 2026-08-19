import { NextResponse } from "next/server";

import { orchestratorUpstreamBase } from "@/lib/orchestrator-url";

const API_BASE = orchestratorUpstreamBase();

/** Allowed orchestrator preview paths (no open proxy). */
const PREVIEW_PATH = /^\/utils\/prescan\/[a-zA-Z0-9_-]+\/preview\.jpg$/;

export async function GET(request: Request): Promise<NextResponse> {
  const { searchParams } = new URL(request.url);
  const path = searchParams.get("path");

  if (!path || !PREVIEW_PATH.test(path)) {
    return NextResponse.json({ detail: "invalid preview path" }, { status: 400 });
  }

  const upstream = `${API_BASE}${path}`;
  try {
    const response = await fetch(upstream, { cache: "no-store" });
    if (!response.ok) {
      return NextResponse.json(
        { detail: `preview upstream ${response.status}` },
        { status: response.status },
      );
    }
    const bytes = await response.arrayBuffer();
    return new NextResponse(bytes, {
      status: 200,
      headers: {
        "Content-Type": response.headers.get("Content-Type") ?? "image/jpeg",
        "Cache-Control": "no-store",
      },
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json({ detail: message }, { status: 502 });
  }
}
