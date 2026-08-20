"use client";

import type { TelemetryMessage } from "@viana/contracts";

import { DATE_PATTERN, TIME_PATTERN, formatEta } from "@/lib/validation";

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
  arrow: string;
  trackId: string;
  /** Media timeline position from MOVING_EVENT `video_pts_ms` when present. */
  videoPtsMs?: number;
  /** From MOVING_EVENT `frame_index` — fallback sync key with `fps`. */
  frameIndex?: number;
  /** From MOVING_EVENT `fps` (or job hint) — pairs with `frameIndex`. */
  fps?: number;
}

function directionArrow(direction: string): string {
  const d = direction.toLowerCase();
  if (d === "in") {
    return "↑";
  }
  if (d === "out") {
    return "↓";
  }
  return "·";
}

export function crossingArrowClass(direction: string): string {
  const d = direction.toLowerCase();
  if (d === "in") {
    return "font-bold text-lime-600 dark:text-lime-400";
  }
  if (d === "out") {
    return "font-bold text-fuchsia-600 dark:text-fuchsia-400";
  }
  return "font-bold text-muted";
}

function frameToClock(frame: number, fps: number): string {
  const totalSec = Math.max(0, Math.floor(frame / fps));
  const h = Math.floor(totalSec / 3600);
  const m = Math.floor((totalSec % 3600) / 60);
  const s = totalSec % 60;
  return [h, m, s].map((v) => String(v).padStart(2, "0")).join(":");
}

function parseStartDateTime(dateStr: string, timeStr: string): Date | null {
  const date = dateStr.trim();
  const time = timeStr.trim();
  if (!DATE_PATTERN.test(date) || !TIME_PATTERN.test(time)) {
    return null;
  }
  const [day, month, year] = date.split("-").map(Number);
  const [hours, minutes, seconds] = time.split(":").map(Number);
  return new Date(year, month - 1, day, hours, minutes, seconds);
}

function formatWallClockTime(date: Date): string {
  const h = String(date.getHours()).padStart(2, "0");
  const m = String(date.getMinutes()).padStart(2, "0");
  const s = String(date.getSeconds()).padStart(2, "0");
  return `${h}:${m}:${s}`;
}

function formatEventTimestamp(value: string): string {
  const trimmed = value.trim();
  if (!trimmed) {
    return "—";
  }

  // API can emit ISO timestamps or HH:MM:SS strings; normalize display.
  const hmsOnly = trimmed.match(/^(\d{2}):(\d{2}):(\d{2})$/);
  if (hmsOnly) {
    return trimmed;
  }

  // Engine encodes OSD / user start clock as naive UTC ISO (`...Z`).
  // Extract that clock directly — do not convert through the browser timezone.
  const isoClock = trimmed.match(
    /^\d{4}-\d{2}-\d{2}[T ](\d{2}):(\d{2}):(\d{2})/,
  );
  if (isoClock) {
    return `${isoClock[1]}:${isoClock[2]}:${isoClock[3]}`;
  }

  const inlineHms = trimmed.match(/(\d{2}):(\d{2}):(\d{2})/);
  if (inlineHms) {
    return `${inlineHms[1]}:${inlineHms[2]}:${inlineHms[3]}`;
  }

  return "—";
}

function eventTimeFromElapsedMs(
  elapsedMs: number,
  startTime?: string,
  startDate?: string,
): string | null {
  if (!Number.isFinite(elapsedMs) || elapsedMs < 0) {
    return null;
  }
  if (!startTime?.trim() || !startDate?.trim()) {
    return null;
  }
  const base = parseStartDateTime(startDate, startTime);
  if (!base) {
    return null;
  }
  return formatWallClockTime(new Date(base.getTime() + elapsedMs));
}

function eventTimeFromFrame(
  frame: number,
  fps: number,
  startTime?: string,
  startDate?: string,
): string {
  if (!fps || fps <= 0) {
    return "—";
  }
  const fromStart = eventTimeFromElapsedMs(
    (frame / fps) * 1000,
    startTime,
    startDate,
  );
  if (fromStart) {
    return fromStart;
  }
  return frameToClock(frame, fps);
}

export function crossingsFromTelemetry(
  messages: TelemetryMessage[],
  jobId: string,
  fpsHint?: number,
  options?: {
    startTime?: string;
    startDate?: string;
    limit?: number;
  },
): CrossingRow[] {
  const limit = options?.limit ?? 500;
  const rows: CrossingRow[] = [];
  for (const msg of messages) {
    if (msg.job_id !== jobId || msg.telemetry_type !== "MOVING_EVENT") {
      continue;
    }
    const data = msg.data;
    const frame =
      typeof data.frame_index === "number" ? data.frame_index : undefined;
    const fps = typeof data.fps === "number" ? data.fps : fpsHint;
    const direction =
      typeof data.direction === "string"
        ? data.direction.charAt(0).toUpperCase() + data.direction.slice(1)
        : "—";
    const rawDirection = typeof data.direction === "string" ? data.direction : "";
    const videoPtsMs =
      typeof data.video_pts_ms === "number" ? data.video_pts_ms : undefined;
    const elapsedMs =
      typeof videoPtsMs === "number"
        ? videoPtsMs
        : frame !== undefined && fps && fps > 0
          ? (frame / fps) * 1000
          : undefined;
    const fromVideoStart =
      elapsedMs !== undefined
        ? eventTimeFromElapsedMs(
            elapsedMs,
            options?.startTime,
            options?.startDate,
          )
        : null;
    rows.push({
      id: `${msg.job_id}-${rows.length}-${frame ?? 0}`,
      time: fromVideoStart
        ? fromVideoStart
        : typeof data.event_timestamp === "string" && data.event_timestamp.trim()
          ? formatEventTimestamp(data.event_timestamp)
          : frame !== undefined && fps
            ? eventTimeFromFrame(
                frame,
                fps,
                options?.startTime,
                options?.startDate,
              )
            : "—",
      vehicle:
        typeof data.class_name === "string" ? data.class_name : "Unknown",
      direction,
      arrow: directionArrow(rawDirection),
      trackId: typeof data.track_id === "number" ? String(data.track_id) : "—",
      videoPtsMs,
      frameIndex: frame,
      fps: typeof fps === "number" && fps > 0 ? fps : undefined,
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
