import { NextResponse } from "next/server";

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

/** Allowed orchestrator source video paths (no open proxy). */
const SOURCE_PATH = /^\/artifacts\/[a-zA-Z0-9_-]+\/source\.mp4$/;

export async function GET(request: Request): Promise<NextResponse> {
  const { searchParams } = new URL(request.url);
  const path = searchParams.get("path");

  if (!path || !SOURCE_PATH.test(path)) {
    return NextResponse.json({ detail: "invalid source path" }, { status: 400 });
  }

  const upstream = `${API_BASE}${path}`;
  const range = request.headers.get("range");
  const upstreamHeaders: HeadersInit = {};
  if (range) {
    upstreamHeaders.Range = range;
  }

  try {
    const response = await fetch(upstream, {
      cache: "no-store",
      headers: upstreamHeaders,
    });
    if (!response.ok && response.status !== 206) {
      return NextResponse.json(
        { detail: `source upstream ${response.status}` },
        { status: response.status },
      );
    }

    const headers = new Headers();
    const contentType = response.headers.get("Content-Type");
    if (contentType) {
      headers.set("Content-Type", contentType);
    } else {
      headers.set("Content-Type", "video/mp4");
    }
    headers.set("Cache-Control", "no-store");
    const contentLength = response.headers.get("Content-Length");
    if (contentLength) {
      headers.set("Content-Length", contentLength);
    }
    const contentRange = response.headers.get("Content-Range");
    if (contentRange) {
      headers.set("Content-Range", contentRange);
    }
    const acceptRanges = response.headers.get("Accept-Ranges");
    if (acceptRanges) {
      headers.set("Accept-Ranges", acceptRanges);
    } else {
      headers.set("Accept-Ranges", "bytes");
    }

    return new NextResponse(response.body, {
      status: response.status,
      headers,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json({ detail: message }, { status: 502 });
  }
}
