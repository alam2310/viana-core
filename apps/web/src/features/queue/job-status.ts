import type { JobStatus, JobStatusResponse } from "@viana/contracts";

export const STATUS_LABELS: Record<JobStatus, string> = {
  PRESCAN_PENDING: "Waiting for prescan",
  PRESCAN_RUNNING: "Pre-scanning video",
  PRESCAN_FAILED: "Prescan failed",
  AWAITING_REVIEW: "Needs review",
  READY: "Ready",
  PROCESSING: "Processing",
  PAUSED: "Paused",
  COMPLETED: "Completed",
  FAILED: "Failed",
  CANCELLED: "Cancelled",
};

export function statusLabel(status: JobStatus): string {
  return STATUS_LABELS[status] ?? status;
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
