/**
 * Resolve orchestrator REST base URL for browser vs server.
 * Browser uses same-origin Next.js proxy so remote UI hosts work without hardcoded localhost.
 */

const SERVER_ORCHESTRATOR =
  process.env.ORCHESTRATOR_API_URL ??
  process.env.NEXT_PUBLIC_API_URL ??
  "http://localhost:8000";

/** Upstream URL for Next.js route handlers (server-side). */
export function orchestratorUpstreamBase(): string {
  return SERVER_ORCHESTRATOR.replace(/\/$/, "");
}

/** Browser REST base — proxied through Next.js. */
export function browserApiBase(): string {
  if (typeof window === "undefined") {
    return orchestratorUpstreamBase();
  }
  return "/api/orchestrator";
}

/** WebSocket URL — host must expose orchestrator port (default 8000). */
export function browserWsJobsUrl(): string {
  if (typeof window === "undefined") {
    const base = orchestratorUpstreamBase();
    return `${base.replace(/^http/, "ws")}/ws/jobs`;
  }
  const { protocol, hostname } = window.location;
  const wsProto = protocol === "https:" ? "wss:" : "ws:";
  const port = process.env.NEXT_PUBLIC_API_PORT ?? "8000";
  return `${wsProto}//${hostname}:${port}/ws/jobs`;
}
