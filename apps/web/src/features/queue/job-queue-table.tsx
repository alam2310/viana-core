"use client";

import type { JobStatusResponse } from "@viana/contracts";

import { Button } from "@/components/ui/button";
import {
  isCancellable,
  isReviewable,
  statusLabel,
} from "@/features/queue/job-status";
import { videoStem } from "@/lib/geometry";
import {
  formatDurationSec,
  formatSubmittedAt,
  getJobLocalMeta,
  processingDurationSec,
  sortJobsBySubmitted,
} from "@/lib/job-local-meta";
import { formatJobErrorMessage, gpuIdFromDevice } from "@/lib/job-errors";
import { formatEta } from "@/lib/validation";
import { cn } from "@/lib/utils";

function CancelJobButton({
  jobId,
  busy,
  onCancel,
}: {
  jobId: string;
  busy: boolean;
  onCancel: (jobId: string) => void;
}) {
  return (
    <button
      type="button"
      title="Cancel job"
      aria-label="Cancel job"
      disabled={busy}
      className="inline-flex h-7 w-7 shrink-0 items-center justify-center rounded text-red-600 hover:bg-red-50 disabled:opacity-40"
      onClick={(event) => {
        event.stopPropagation();
        onCancel(jobId);
      }}
    >
      <span className="text-base leading-none" aria-hidden>
        ×
      </span>
    </button>
  );
}

function progressPct(job: JobStatusResponse): number | null {
  if (job.status === "COMPLETED") {
    return 100;
  }
  const total = job.progress?.total_frames;
  const current = job.progress?.current_frame;
  if (!total || total <= 0 || current === undefined) {
    return null;
  }
  return Math.min(100, Math.round((current / total) * 100));
}

export function JobQueueTable({
  jobs,
  busyId,
  selectedJobId,
  monitorJobId,
  onSelectJob,
  onReview,
  onMonitor,
  onRetryPrescan,
  onResume,
  onStartFresh,
  onCancel,
}: {
  jobs: JobStatusResponse[];
  busyId: string | null;
  selectedJobId: string | null;
  monitorJobId: string | null;
  onSelectJob: (job: JobStatusResponse) => void;
  onReview: (job: JobStatusResponse) => void;
  onMonitor: (job: JobStatusResponse) => void;
  onRetryPrescan: (jobId: string) => void;
  onResume: (jobId: string) => void;
  onStartFresh: (jobId: string) => void;
  onCancel: (jobId: string) => void;
}) {
  const sorted = sortJobsBySubmitted(jobs);

  return (
    <section className="rounded-lg border border-neutral-200 bg-white p-4">
      <h2 className="text-sm font-semibold tracking-wide text-neutral-500 uppercase">
        Job queue
      </h2>
      <p className="mt-1 text-xs text-neutral-500">
        Videos are processed in order after you confirm the prescan review.
      </p>

      <div className="mt-3 overflow-x-auto">
        <table className="w-full table-fixed text-left text-sm">
          <colgroup>
            <col className="w-[14%]" />
            <col className="w-[11%]" />
            <col className="w-[9%]" />
            <col className="w-[6%]" />
            <col className="w-[8%]" />
            <col className="w-[9%]" />
            <col className="w-[14%]" />
            <col className="w-[29%]" />
          </colgroup>
          <thead>
            <tr className="border-b border-neutral-200 text-xs text-neutral-500">
              <th className="py-2 pr-2 font-medium">Video</th>
              <th className="py-2 pr-2 font-medium">Submitted</th>
              <th className="py-2 pr-2 font-medium">Status</th>
              <th className="py-2 pr-2 font-medium whitespace-nowrap">GPU</th>
              <th className="py-2 pr-2 font-medium whitespace-nowrap">Vid</th>
              <th className="py-2 pr-2 font-medium whitespace-nowrap">Run</th>
              <th className="py-2 pr-2 font-medium">Progress</th>
              <th className="py-2 font-medium">Actions</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((job) => {
              const pct = progressPct(job);
              const local = getJobLocalMeta(job.job_id);
              const errorText = formatJobErrorMessage(job.error_message);
              const paused = job.status === "PAUSED";
              return (
                <tr
                  key={job.job_id}
                  className={cn(
                    "cursor-pointer border-b border-neutral-100 align-top hover:bg-neutral-50",
                    paused && "bg-amber-50",
                    selectedJobId === job.job_id && "bg-sky-50",
                    monitorJobId === job.job_id && "bg-emerald-50",
                  )}
                  onClick={() => onSelectJob(job)}
                >
                  <td className="py-2 pr-2">
                    <p
                      className="truncate font-mono text-xs"
                      title={job.source_video_path}
                    >
                      {videoStem(job.source_video_path)}
                    </p>
                  </td>
                  <td className="py-2 pr-2 text-xs text-neutral-600 whitespace-nowrap">
                    {formatSubmittedAt(local.submittedAt)}
                  </td>
                  <td className="py-2 pr-2">
                    <span
                      className={cn(
                        "inline-block whitespace-nowrap rounded px-1.5 py-0.5 text-[11px] font-medium",
                        job.status === "AWAITING_REVIEW" && "bg-amber-100 text-amber-900",
                        job.status === "PROCESSING" && "bg-emerald-100 text-emerald-900",
                        job.status === "PRESCAN_FAILED" && "bg-red-100 text-red-900",
                        job.status === "READY" && "bg-blue-100 text-blue-900",
                      )}
                    >
                      {statusLabel(job.status)}
                    </span>
                    {errorText ? (
                      <p className="mt-1 line-clamp-2 text-[10px] text-red-700">
                        {errorText}
                      </p>
                    ) : null}
                  </td>
                  <td className="py-2 pr-2 text-center font-mono text-xs text-neutral-600">
                    {gpuIdFromDevice(job.gpu_device)}
                  </td>
                  <td className="py-2 pr-2 text-xs text-neutral-600 whitespace-nowrap">
                    {formatDurationSec(local.videoDurationSec)}
                  </td>
                  <td className="py-2 pr-2 text-xs text-neutral-600 whitespace-nowrap">
                    {formatDurationSec(processingDurationSec(local))}
                  </td>
                  <td className="py-2 pr-2 text-xs text-neutral-600">
                    {pct !== null ? (
                      <>
                        <div className="h-1.5 w-full max-w-[6rem] overflow-hidden rounded bg-neutral-200">
                          <div
                            className="h-full bg-neutral-900"
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                        <p className="mt-0.5 whitespace-nowrap">
                          {pct}%
                          {job.status === "PROCESSING" && job.progress?.processing_fps
                            ? ` · ${job.progress.processing_fps.toFixed(1)} fps`
                            : ""}
                          {job.status === "PROCESSING" &&
                          job.progress?.eta_sec !== undefined
                            ? ` · ${formatEta(job.progress.eta_sec)}`
                            : ""}
                        </p>
                      </>
                    ) : job.queue_position ? (
                      <span className="whitespace-nowrap">#{job.queue_position}</span>
                    ) : (
                      "—"
                    )}
                  </td>
                  <td className="py-2">
                    <div
                      className="flex flex-wrap items-center gap-1"
                      onClick={(event) => event.stopPropagation()}
                    >
                      {isReviewable(job.status) ? (
                        <Button
                          type="button"
                          size="sm"
                          variant="outline"
                          onClick={() => onReview(job)}
                        >
                          Review
                        </Button>
                      ) : null}
                      {job.status === "PRESCAN_FAILED" ? (
                        <Button
                          type="button"
                          size="sm"
                          disabled={busyId === job.job_id}
                          onClick={() => onRetryPrescan(job.job_id)}
                        >
                          Retry
                        </Button>
                      ) : null}
                      {job.status === "PROCESSING" ? (
                        <Button
                          type="button"
                          size="sm"
                          variant="outline"
                          onClick={() => onMonitor(job)}
                        >
                          Monitor
                        </Button>
                      ) : null}
                      {job.status === "PAUSED" ? (
                        <Button
                          type="button"
                          size="sm"
                          disabled={busyId === job.job_id}
                          onClick={() => onResume(job.job_id)}
                        >
                          Resume
                        </Button>
                      ) : null}
                      {(job.status === "PAUSED" ||
                        (job.status === "FAILED" && job.checkpoint_exists)) && (
                        <Button
                          type="button"
                          size="sm"
                          variant="danger"
                          disabled={busyId === job.job_id}
                          onClick={() => onStartFresh(job.job_id)}
                        >
                          Fresh
                        </Button>
                      )}
                      {isCancellable(job.status) ? (
                        <CancelJobButton
                          jobId={job.job_id}
                          busy={busyId === job.job_id}
                          onCancel={onCancel}
                        />
                      ) : null}
                    </div>
                  </td>
                </tr>
              );
            })}
            {sorted.length === 0 ? (
              <tr>
                <td colSpan={8} className="py-6 text-center text-sm text-neutral-500">
                  No jobs yet — add videos above.
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
    </section>
  );
}
