import type { JobStatus, JobStatusResponse } from "@viana/contracts";

export const STATUS_LABELS: Record<JobStatus, string> = {
  PRESCAN_PENDING: "Queued",
  PRESCAN_RUNNING: "Pre-scan",
  PRESCAN_FAILED: "Pre-scan failed",
  AWAITING_REVIEW: "Review",
  READY: "Ready",
  PROCESSING: "Processing",
  PAUSED: "Paused",
  COMPLETED: "Completed",
  FAILED: "Failed",
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

export function statusBadgeClass(status: JobStatus): string {
  return STATUS_BADGE_CLASS[status] ?? "bg-accent text-foreground";
}

export function isReviewable(status: JobStatus): boolean {
  return status === "AWAITING_REVIEW" || status === "READY" || status === "PRESCAN_FAILED";
}

export function isCancellable(status: JobStatus): boolean {
  return status !== "COMPLETED" && status !== "CANCELLED";
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
