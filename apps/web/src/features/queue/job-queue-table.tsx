"use client";

import type { JobStatusResponse } from "@viana/contracts";

import {
  IconCancel,
  IconFolder,
  IconMonitor,
  IconPlay,
  IconRestart,
  IconRetry,
  IconReview,
  RoundIconButton,
} from "@/components/ui/icon-button";
import {
  isCancellable,
  isReviewable,
  statusBadgeClass,
  statusLabel,
} from "@/features/queue/job-status";
import { videoStem } from "@/lib/geometry";
import {
  formatSubmittedAt,
  formatVideoLengthHms,
  getJobLocalMeta,
  runTimeSec,
  sortJobsBySubmitted,
} from "@/lib/job-local-meta";
import { progressBarColor } from "@/lib/progress-bar";
import { gpuIdFromDevice } from "@/lib/job-errors";
import { formatEta } from "@/lib/validation";
import { cn } from "@/lib/utils";

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

function timeRemaining(job: JobStatusResponse): string {
  if (job.status !== "PROCESSING" || job.progress?.eta_sec === undefined) {
    return "—";
  }
  return formatEta(job.progress.eta_sec);
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
  onOpenOutput,
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
  onOpenOutput: (job: JobStatusResponse) => void;
}) {
  const sorted = sortJobsBySubmitted(jobs);

  return (
    <section className="rounded-lg border border-border bg-card p-4">
      <h2 className="text-sm font-semibold tracking-wide text-muted uppercase">
        Job Queue
      </h2>
      <p className="mt-1 text-xs text-muted">
        Confirm prescan review to process videos in order.
      </p>

      <div className="mt-3 overflow-x-auto">
        <table className="w-full table-fixed text-left text-sm leading-tight">
          <colgroup>
            <col style={{ width: "15%" }} />
            <col style={{ width: "10%" }} />
            <col style={{ width: "9%" }} />
            <col style={{ width: "5%" }} />
            <col style={{ width: "8%" }} />
            <col style={{ width: "8%" }} />
            <col style={{ width: "11%" }} />
            <col style={{ width: "9%" }} />
            <col style={{ width: "12%" }} />
          </colgroup>
          <thead>
            <tr className="border-b border-border text-xs text-muted">
              <th className="px-2 py-1 font-medium">Video</th>
              <th className="px-2 py-1 font-medium">Submitted</th>
              <th className="px-2 py-1 font-medium">Status</th>
              <th className="px-2 py-1 text-left font-medium">GPU</th>
              <th className="px-2 py-1 font-medium whitespace-nowrap">Video Length</th>
              <th className="px-2 py-1 font-medium whitespace-nowrap">Run Time</th>
              <th className="px-2 py-1 font-medium">Progress</th>
              <th className="px-2 py-1 font-medium whitespace-nowrap">Time Remaining</th>
              <th className="px-2 py-1 font-medium">Actions</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((job) => {
              const pct = progressPct(job);
              const local = getJobLocalMeta(job.job_id);
              const paused = job.status === "PAUSED";
              return (
                <tr
                  key={job.job_id}
                  className={cn(
                    "cursor-pointer border-b border-border align-middle hover:bg-card-hover",
                    paused && "bg-row-paused",
                    selectedJobId === job.job_id && "bg-row-selected",
                    monitorJobId === job.job_id && "bg-row-monitor",
                  )}
                  onClick={() => onSelectJob(job)}
                >
                  <td className="px-2 py-1">
                    <p
                      className="truncate font-mono text-xs leading-tight"
                      title={job.source_video_path}
                    >
                      {videoStem(job.source_video_path)}
                    </p>
                  </td>
                  <td className="px-2 py-1 text-xs text-muted whitespace-nowrap">
                    {formatSubmittedAt(job.created_at ?? local.submittedAt)}
                  </td>
                  <td className="px-2 py-1">
                    <span
                      className={cn(
                        "inline-block whitespace-nowrap rounded px-1.5 py-0.5 text-[11px] font-medium leading-none",
                        statusBadgeClass(job.status),
                      )}
                    >
                      {statusLabel(job.status)}
                    </span>
                  </td>
                  <td className="px-2 py-1 text-left font-mono text-xs text-muted">
                    {gpuIdFromDevice(job.gpu_device)}
                  </td>
                  <td className="px-2 py-1 font-mono text-xs text-muted whitespace-nowrap">
                    {formatVideoLengthHms(
                      job.video_duration_sec ?? local.videoDurationSec,
                    )}
                  </td>
                  <td className="px-2 py-1 font-mono text-xs text-muted whitespace-nowrap">
                    {formatVideoLengthHms(runTimeSec(job, local))}
                  </td>
                  <td className="px-2 py-1">
                    {pct !== null ? (
                      <div className="flex items-center gap-1.5">
                        <div className="h-1.5 w-[4.5rem] shrink-0 overflow-hidden rounded bg-accent">
                          <div
                            className={cn(
                              "h-full transition-colors",
                              progressBarColor(pct, job.status),
                            )}
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                        <span className="w-10 shrink-0 text-right text-xs tabular-nums">
                          {pct}%
                        </span>
                      </div>
                    ) : job.queue_position ? (
                      <span className="text-xs whitespace-nowrap text-muted">
                        #{job.queue_position}
                      </span>
                    ) : (
                      <span className="text-xs text-muted">—</span>
                    )}
                  </td>
                  <td className="px-2 py-1 text-xs text-muted whitespace-nowrap">
                    {timeRemaining(job)}
                  </td>
                  <td className="px-2 py-1">
                    <div
                      className="flex items-center justify-start gap-1"
                      onClick={(event) => event.stopPropagation()}
                    >
                      {isReviewable(job.status) ? (
                        <RoundIconButton
                          label="Review prescan"
                          variant="info"
                          size="sm"
                          onClick={() => onReview(job)}
                        >
                          <IconReview size={16} />
                        </RoundIconButton>
                      ) : null}
                      {job.status === "PRESCAN_FAILED" ? (
                        <RoundIconButton
                          label="Retry pre-scan"
                          variant="warning"
                          size="sm"
                          disabled={busyId === job.job_id}
                          onClick={() => onRetryPrescan(job.job_id)}
                        >
                          <IconRetry size={16} />
                        </RoundIconButton>
                      ) : null}
                      {job.status === "PROCESSING" ? (
                        <RoundIconButton
                          label="Monitor live"
                          variant="success"
                          size="sm"
                          onClick={() => onMonitor(job)}
                        >
                          <IconMonitor size={16} />
                        </RoundIconButton>
                      ) : null}
                      {job.status === "PAUSED" ? (
                        <RoundIconButton
                          label="Resume job"
                          variant="success"
                          size="sm"
                          disabled={busyId === job.job_id}
                          onClick={() => onResume(job.job_id)}
                        >
                          <IconPlay size={16} />
                        </RoundIconButton>
                      ) : null}
                      {(job.status === "PAUSED" ||
                        job.status === "FAILED") && (
                        <RoundIconButton
                          label="Resubmit — overwrite existing output"
                          variant="accent"
                          size="sm"
                          disabled={busyId === job.job_id}
                          onClick={() => onStartFresh(job.job_id)}
                        >
                          <IconRestart size={16} />
                        </RoundIconButton>
                      )}
                      {job.status === "COMPLETED" ? (
                        <RoundIconButton
                          label="Open output directory"
                          variant="info"
                          size="sm"
                          onClick={() => onOpenOutput(job)}
                        >
                          <IconFolder size={16} />
                        </RoundIconButton>
                      ) : null}
                      {isCancellable(job.status) ? (
                        <RoundIconButton
                          label="Cancel job"
                          size="sm"
                          variant="danger"
                          disabled={busyId === job.job_id}
                          onClick={() => onCancel(job.job_id)}
                        >
                          <IconCancel size={16} />
                        </RoundIconButton>
                      ) : null}
                    </div>
                  </td>
                </tr>
              );
            })}
            {sorted.length === 0 ? (
              <tr>
                <td colSpan={9} className="py-6 text-center text-sm text-muted">
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
