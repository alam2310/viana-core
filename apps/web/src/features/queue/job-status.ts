import type { JobStatus, JobStatusResponse } from "@viana/contracts";

/**
 * Operator-facing copy only. API `JobStatus` enum values are unchanged (S25 / F020).
 * Waiting-for-capacity states share “Queued” and name the resource.
 */
export const STATUS_LABELS: Record<JobStatus, string> = {
  PRESCAN_PENDING: "Queued (PS)",
  PRESCAN_RUNNING: "Pre-scanning",
  PRESCAN_FAILED: "Prescan failed",
  AWAITING_REVIEW: "Needs review",
  READY: "Queued (GPU)",
  PROCESSING: "Processing",
  PAUSED: "Paused",
  COMPLETED: "Completed",
  FAILED: "Failed",
  CANCELLED: "Cancelled",
};

export const STATUS_HINTS: Record<JobStatus, string> = {
  PRESCAN_PENDING: "Waiting for a prescan worker",
  PRESCAN_RUNNING: "Prescan is sampling the video",
  PRESCAN_FAILED: "Prescan error — retry or inspect the message",
  AWAITING_REVIEW: "Confirm geometry and metadata before processing",
  READY: "Confirmed — waiting for a GPU slot",
  PROCESSING: "Engine running on GPU",
  PAUSED: "Processing paused",
  COMPLETED: "Finished successfully",
  FAILED: "Processing failed",
  CANCELLED: "Cancelled",
};

/** One distinct hue per status (light + dark). */
export const STATUS_BADGE_CLASS: Record<JobStatus, string> = {
  PRESCAN_PENDING:
    "bg-slate-200 text-slate-800 dark:bg-slate-700 dark:text-slate-100",
  PRESCAN_RUNNING:
    "bg-violet-200 text-violet-900 dark:bg-violet-900 dark:text-violet-100",
  PRESCAN_FAILED:
    "bg-rose-200 text-rose-900 dark:bg-rose-950 dark:text-rose-100",
  AWAITING_REVIEW:
    "bg-amber-200 text-amber-950 dark:bg-amber-900 dark:text-amber-100",
  READY: "bg-sky-200 text-sky-950 dark:bg-sky-900 dark:text-sky-100",
  PROCESSING:
    "bg-orange-200 text-orange-950 dark:bg-orange-900 dark:text-orange-100",
  PAUSED:
    "bg-yellow-200 text-yellow-950 dark:bg-yellow-900 dark:text-yellow-100",
  COMPLETED:
    "bg-emerald-200 text-emerald-950 dark:bg-emerald-900 dark:text-emerald-100",
  FAILED: "bg-red-200 text-red-950 dark:bg-red-900 dark:text-red-100",
  CANCELLED:
    "bg-zinc-300 text-zinc-800 dark:bg-zinc-700 dark:text-zinc-200",
};

export function statusLabel(status: JobStatus): string {
  return STATUS_LABELS[status] ?? status;
}

export function statusHint(status: JobStatus): string {
  return STATUS_HINTS[status] ?? status;
}

export function statusBadgeClass(status: JobStatus): string {
  return STATUS_BADGE_CLASS[status] ?? "bg-accent text-foreground";
}

export function isReviewable(status: JobStatus): boolean {
  return status === "AWAITING_REVIEW" || status === "READY" || status === "PRESCAN_FAILED";
}

export function canRetryPrescan(status: JobStatus): boolean {
  return status === "PRESCAN_FAILED";
}

export function canStartFresh(status: JobStatus): boolean {
  return status === "PAUSED" || status === "FAILED";
}

export function canOpenOutput(status: JobStatus): boolean {
  return status === "COMPLETED";
}

export function isCancellable(status: JobStatus): boolean {
  return status !== "COMPLETED" && status !== "CANCELLED";
}

/** Pause from operator stop is resumable; engine failure parked as PAUSED is not. */
export function isResumablePause(job: JobStatusResponse): boolean {
  if (job.status !== "PAUSED") {
    return false;
  }
  const message = job.error_message?.trim() ?? "";
  if (!message) {
    return true;
  }
  return /^(worker cancelled|interrupted)$/i.test(message);
}

export function isActiveStatus(status: JobStatus): boolean {
  return (
    status === "PRESCAN_PENDING" ||
    status === "PRESCAN_RUNNING" ||
    status === "AWAITING_REVIEW" ||
    status === "READY" ||
    status === "PROCESSING" ||
    status === "PAUSED"
  );
}

export function shouldPollJobs(jobs: JobStatusResponse[]): boolean {
  return jobs.some((job) => isActiveStatus(job.status));
}

export function sortJobsFifo<T extends { queue_position?: number; job_id: string }>(
  jobs: T[],
): T[] {
  return [...jobs].sort((a, b) => {
    const posA = a.queue_position ?? Number.MAX_SAFE_INTEGER;
    const posB = b.queue_position ?? Number.MAX_SAFE_INTEGER;
    if (posA !== posB) {
      return posA - posB;
    }
    return a.job_id.localeCompare(b.job_id);
  });
}
