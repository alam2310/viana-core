"use client";

import type { TelemetryMessage } from "@viana/contracts";

import { formatEta } from "@/lib/validation";

export function mapLogMessage(message: unknown): string {
  if (typeof message !== "string") {
    return "Activity event";
  }
  switch (message) {
    case "process_start":
      return "Processing started";
    case "process_complete":
      return "Processing finished";
    case "interrupted":
      return "Processing paused";
    default:
      return message;
  }
}

export function progressFromTelemetry(
  messages: TelemetryMessage[],
  jobId: string,
): {
  current: number;
  total: number;
  fps?: number;
  etaSec?: number;
  crossingCount?: number;
  pct: number | null;
} | null {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const msg = messages[i];
    if (msg.job_id !== jobId || msg.telemetry_type !== "PROGRESS") {
      continue;
    }
    const data = msg.data;
    const current =
      typeof data.current_frame === "number" ? data.current_frame : undefined;
    const total =
      typeof data.total_frames === "number" ? data.total_frames : undefined;
    if (current === undefined || total === undefined || total <= 0) {
      continue;
    }
    const fps =
      typeof data.processing_fps === "number" ? data.processing_fps : undefined;
    const etaSec =
      typeof data.eta_sec === "number"
        ? data.eta_sec
        : fps && fps > 0
          ? (total - current) / fps
          : undefined;
    const crossingCount =
      typeof data.crossing_count === "number" ? data.crossing_count : undefined;
    return {
      current,
      total,
      fps,
      etaSec,
      crossingCount,
      pct: Math.min(100, Math.round((current / total) * 100)),
    };
  }
  return null;
}

export function formatProgressLine(progress: {
  pct: number | null;
  fps?: number;
  etaSec?: number;
}): string {
  const parts: string[] = [];
  if (progress.pct !== null) {
    parts.push(`${progress.pct}%`);
  }
  if (progress.fps !== undefined) {
    parts.push(`${progress.fps.toFixed(1)} fps`);
  }
  parts.push(formatEta(progress.etaSec));
  return parts.join(" · ");
}

export interface CrossingRow {
  id: string;
  time: string;
  vehicle: string;
  direction: string;
  trackId: string;
}

export function crossingsFromTelemetry(
  messages: TelemetryMessage[],
  jobId: string,
  limit = 500,
): CrossingRow[] {
  const rows: CrossingRow[] = [];
  for (const msg of messages) {
    if (msg.job_id !== jobId || msg.telemetry_type !== "MOVING_EVENT") {
      continue;
    }
    const data = msg.data;
    const frame =
      typeof data.frame_index === "number" ? data.frame_index : undefined;
    rows.push({
      id: `${msg.job_id}-${rows.length}`,
      time: frame !== undefined ? `frame ${frame}` : "—",
      vehicle:
        typeof data.class_name === "string" ? data.class_name : "Unknown",
      direction:
        typeof data.direction === "string"
          ? data.direction.charAt(0).toUpperCase() + data.direction.slice(1)
          : "—",
      trackId: typeof data.track_id === "number" ? String(data.track_id) : "—",
    });
  }
  return rows.slice(-limit);
}

export interface ActivityRow {
  id: string;
  text: string;
}

export function activityFromTelemetry(
  messages: TelemetryMessage[],
  jobId: string,
  limit = 50,
): ActivityRow[] {
  const rows: ActivityRow[] = [];
  for (const msg of messages) {
    if (msg.job_id !== jobId || msg.telemetry_type !== "LOG") {
      continue;
    }
    rows.push({
      id: `${msg.job_id}-log-${rows.length}`,
      text: mapLogMessage(msg.data.message),
    });
  }
  return rows.slice(-limit);
}
