const PROJECT_KEY = "viana.project_id";
const TELEMETRY_KEY = "viana.telemetry_detail";
const PENDING_KEY = "viana.pending_paths";

export function readProjectId(): string {
  if (typeof window === "undefined") {
    return "nh48";
  }
  return window.localStorage.getItem(PROJECT_KEY) ?? "nh48";
}

export function writeProjectId(projectId: string): void {
  window.localStorage.setItem(PROJECT_KEY, projectId);
}

export function readTelemetryDetail(): boolean {
  if (typeof window === "undefined") {
    return false;
  }
  return window.localStorage.getItem(TELEMETRY_KEY) === "true";
}

export function writeTelemetryDetail(enabled: boolean): void {
  window.localStorage.setItem(TELEMETRY_KEY, enabled ? "true" : "false");
}

export function readPendingPaths(): string[] {
  if (typeof window === "undefined") {
    return [];
  }
  try {
    const raw = window.localStorage.getItem(PENDING_KEY);
    const parsed = raw ? (JSON.parse(raw) as unknown) : [];
    return Array.isArray(parsed)
      ? parsed.filter((item): item is string => typeof item === "string")
      : [];
  } catch {
    return [];
  }
}

export function writePendingPaths(paths: string[]): void {
  window.localStorage.setItem(PENDING_KEY, JSON.stringify(paths));
}
