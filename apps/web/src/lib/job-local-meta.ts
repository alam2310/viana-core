import type { JobStatusResponse } from "@viana/contracts";

const STORAGE_KEY = "viana.job_local_meta";

export interface JobLocalMeta {
  submittedAt?: string;
  processingStartedAt?: string;
  processingEndedAt?: string;
  videoDurationSec?: number;
}

type MetaMap = Record<string, JobLocalMeta>;

function readMap(): MetaMap {
  if (typeof window === "undefined") {
    return {};
  }
  try {
    return JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? "{}") as MetaMap;
  } catch {
    return {};
  }
}

function writeMap(map: MetaMap): void {
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(map));
}

export function getJobLocalMeta(jobId: string): JobLocalMeta {
  return readMap()[jobId] ?? {};
}

/** Track submit/processing times until API exposes `created_at` / durations (backend S11–S12). */
export function syncJobLocalMeta(jobs: JobStatusResponse[]): void {
  if (typeof window === "undefined") {
    return;
  }
  const map = readMap();
  const now = new Date().toISOString();
  for (const job of jobs) {
    const prev = map[job.job_id] ?? {};
    if (!prev.submittedAt) {
      prev.submittedAt = now;
    }
    if (job.status === "PROCESSING" && !prev.processingStartedAt) {
      prev.processingStartedAt = now;
    }
    if (
      (job.status === "COMPLETED" || job.status === "FAILED" || job.status === "CANCELLED") &&
      !prev.processingEndedAt
    ) {
      prev.processingEndedAt = now;
    }
    if (
      job.progress?.total_frames &&
      job.progress?.processing_fps &&
      job.progress.processing_fps > 0 &&
      !prev.videoDurationSec
    ) {
      prev.videoDurationSec = job.progress.total_frames / job.progress.processing_fps;
    }
    map[job.job_id] = prev;
  }
  writeMap(map);
}

export function formatSubmittedAt(iso: string | undefined): string {
  if (!iso) {
    return "—";
  }
  try {
    return new Date(iso).toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return "—";
  }
}

export function formatDurationSec(sec: number | undefined): string {
  if (sec === undefined || !Number.isFinite(sec) || sec < 0) {
    return "—";
  }
  if (sec < 60) {
    return `${Math.round(sec)}s`;
  }
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  return `${m}m ${s}s`;
}

export function processingDurationSec(meta: JobLocalMeta): number | undefined {
  if (!meta.processingStartedAt) {
    return undefined;
  }
  const end = meta.processingEndedAt
    ? new Date(meta.processingEndedAt).getTime()
    : Date.now();
  const start = new Date(meta.processingStartedAt).getTime();
  const sec = (end - start) / 1000;
  return sec > 0 ? sec : undefined;
}

export function sortJobsBySubmitted(
  jobs: JobStatusResponse[],
): JobStatusResponse[] {
  const map = readMap();
  return [...jobs].sort((a, b) => {
    const ta = map[a.job_id]?.submittedAt ?? "";
    const tb = map[b.job_id]?.submittedAt ?? "";
    if (ta !== tb) {
      return tb.localeCompare(ta);
    }
    return b.job_id.localeCompare(a.job_id);
  });
}
