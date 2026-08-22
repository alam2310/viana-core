import { NextResponse } from "next/server";

import { orchestratorUpstreamBase } from "@/lib/orchestrator-url";

type RouteContext = { params: Promise<{ path?: string[] }> };

async function proxy(request: Request, context: RouteContext): Promise<NextResponse> {
  const { path: segments } = await context.params;
  const path = (segments ?? []).join("/");
  const upstream = new URL(
    `${orchestratorUpstreamBase()}/${path}${new URL(request.url).search}`,
  );

  const headers = new Headers();
  const accept = request.headers.get("accept");
  if (accept) {
    headers.set("Accept", accept);
  }
  const contentType = request.headers.get("content-type");
  if (contentType) {
    headers.set("Content-Type", contentType);
  }
  const range = request.headers.get("range");
  if (range) {
    headers.set("Range", range);
  }

  const init: RequestInit = {
    method: request.method,
    headers,
    cache: "no-store",
  };
  if (request.method !== "GET" && request.method !== "HEAD") {
    init.body = await request.arrayBuffer();
  }

  try {
    const response = await fetchUpstream(upstream, init, request.method);
    const outHeaders = new Headers();
    const respType = response.headers.get("Content-Type");
    if (respType) {
      outHeaders.set("Content-Type", respType);
    }
    outHeaders.set("Cache-Control", "no-store");
    if (
      request.method === "HEAD" ||
      response.status === 204 ||
      response.status === 304
    ) {
      return new NextResponse(null, { status: response.status, headers: outHeaders });
    }
    const body = await response.arrayBuffer();
    return new NextResponse(body, { status: response.status, headers: outHeaders });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json({ detail: message }, { status: 502 });
  }
}

/** One short retry on GET/HEAD when the orchestrator is briefly unreachable (restart blip). */
async function fetchUpstream(
  upstream: URL,
  init: RequestInit,
  method: string,
): Promise<Response> {
  try {
    return await fetch(upstream, init);
  } catch (err) {
    const idempotent = method === "GET" || method === "HEAD";
    if (!idempotent) {
      throw err;
    }
    await new Promise<void>((resolve) => {
      setTimeout(resolve, 200);
    });
    return await fetch(upstream, init);
  }
}

export async function GET(request: Request, context: RouteContext) {
  return proxy(request, context);
}

export async function POST(request: Request, context: RouteContext) {
  return proxy(request, context);
}

export async function PATCH(request: Request, context: RouteContext) {
  return proxy(request, context);
}

export async function DELETE(request: Request, context: RouteContext) {
  return proxy(request, context);
}

export async function PUT(request: Request, context: RouteContext) {
  return proxy(request, context);
}
