import type { JobStatusResponse } from "@viana/contracts";

export function formatSubmittedAt(iso: string | null | undefined): string {
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

/** Format a duration that is already in **seconds** (not frames or ms). */
export function formatVideoLengthHms(
  sec: number | null | undefined,
): string {
  if (sec === undefined || sec === null || !Number.isFinite(sec) || sec < 0) {
    return "—";
  }
  const total = Math.round(sec);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  return [h, m, s].map((v) => String(v).padStart(2, "0")).join(":");
}

export function formatDurationSec(sec: number | null | undefined): string {
  if (sec === undefined || sec === null || !Number.isFinite(sec) || sec < 0) {
    return "—";
  }
  if (sec < 60) {
    return `${Math.round(sec)}s`;
  }
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  return `${m}m ${s}s`;
}

/** Wall-clock GPU run time from `JobStatus.processing_duration_sec`. */
export function runTimeSec(job: JobStatusResponse): number | undefined {
  if (
    typeof job.processing_duration_sec === "number" &&
    Number.isFinite(job.processing_duration_sec)
  ) {
    return job.processing_duration_sec;
  }
  return undefined;
}

/** Newest `created_at` first. API field is required on JobStatus. */
export function sortJobsBySubmitted(
  jobs: JobStatusResponse[],
): JobStatusResponse[] {
  return [...jobs].sort((a, b) => {
    if (a.created_at !== b.created_at) {
      return b.created_at.localeCompare(a.created_at);
    }
    return b.job_id.localeCompare(a.job_id);
  });
}
