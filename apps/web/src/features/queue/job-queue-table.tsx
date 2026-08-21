"use client";

import { useMemo } from "react";
import type { JobStatusResponse } from "@viana/contracts";

import {
  IconCancel,
  IconFolder,
  IconRestart,
  IconReview,
  RoundIconButton,
} from "@/components/ui/icon-button";
import {
  canOpenOutput,
  canStartFresh,
  isCancellable,
  isReviewable,
  statusBadgeClass,
  statusHint,
  statusLabel,
} from "@/features/queue/job-status";
import { videoStem } from "@/lib/geometry";
import {
  formatSubmittedAt,
  formatVideoLengthHms,
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
  // API `eta_sec` = (total_frames - current_frame) / processing_fps (seconds).
  if (job.status !== "PROCESSING" || job.progress?.eta_sec === undefined) {
    return "—";
  }
  return formatEta(job.progress.eta_sec);
}

export function JobQueueTable({
  jobs,
  busyId,
  selectedJobId,
  onSelectJob,
  onReview,
  onStartFresh,
  onStop,
  onOpenOutput,
}: {
  jobs: JobStatusResponse[];
  busyId: string | null;
  selectedJobId: string | null;
  onSelectJob: (job: JobStatusResponse) => void;
  onReview: (job: JobStatusResponse) => void;
  onStartFresh: (jobId: string) => void;
  onStop: (jobId: string) => void;
  onOpenOutput: (job: JobStatusResponse) => void;
}) {
  // ⚡ Bolt: Memoize the sorted list to prevent expensive O(N log N) sorting on every dashboard re-render (which happens frequently due to polling/telemetry).
  const sorted = useMemo(() => sortJobsBySubmitted(jobs), [jobs]);

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
            <col style={{ width: "16%" }} />
            <col style={{ width: "13%" }} />
            <col style={{ width: "10%" }} />
            <col style={{ width: "5%" }} />
            <col style={{ width: "9%" }} />
            <col style={{ width: "9%" }} />
            <col style={{ width: "14%" }} />
            <col style={{ width: "12%" }} />
            <col style={{ width: "12%" }} />
          </colgroup>
          <thead>
            <tr className="border-b border-border text-xs text-muted">
              <th className="px-3 py-1 font-medium">Video</th>
              <th className="px-3 py-1 font-medium">Submitted</th>
              <th className="px-3 py-1 font-medium">Status</th>
              <th className="px-3 py-1 font-medium">GPU</th>
              <th className="px-3 py-1 font-medium whitespace-nowrap">Video Length</th>
              <th className="px-3 py-1 font-medium whitespace-nowrap">Run Time</th>
              <th className="px-3 py-1 font-medium">Progress</th>
              <th className="px-3 py-1 font-medium whitespace-nowrap">Time Remaining</th>
              <th className="px-3 py-1 font-medium">Actions</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((job) => {
              const pct = progressPct(job);
              const paused = job.status === "PAUSED";
              const busy = busyId === job.job_id;
              const review = isReviewable(job.status);
              const startFresh = canStartFresh(job.status);
              const openOutput = canOpenOutput(job.status);
              const stop = isCancellable(job.status);
              return (
                <tr
                  key={job.job_id}
                  className={cn(
                    "cursor-pointer border-b border-border align-middle hover:bg-card-hover",
                    paused && "bg-row-paused",
                    selectedJobId === job.job_id && "bg-row-selected",
                  )}
                  onClick={() => onSelectJob(job)}
                >
                  <td className="px-3 py-1">
                    <p
                      className="truncate font-mono text-xs leading-tight"
                      title={job.source_video_path}
                    >
                      {videoStem(job.source_video_path)}
                    </p>
                  </td>
                  <td className="px-3 py-1 text-xs text-muted whitespace-nowrap">
                    {formatSubmittedAt(job.created_at)}
                  </td>
                  <td className="px-3 py-1">
                    <span
                      title={statusHint(job.status)}
                      className={cn(
                        "inline-block max-w-full truncate whitespace-nowrap rounded px-1.5 py-0.5 text-[11px] font-medium leading-none",
                        statusBadgeClass(job.status),
                      )}
                    >
                      {statusLabel(job.status)}
                    </span>
                  </td>
                  <td className="px-3 py-1 font-mono text-xs text-muted">
                    {gpuIdFromDevice(job.gpu_device)}
                  </td>
                  <td className="px-3 py-1 font-mono text-xs text-muted whitespace-nowrap">
                    {formatVideoLengthHms(job.video_duration_sec)}
                  </td>
                  <td className="px-3 py-1 font-mono text-xs text-muted whitespace-nowrap">
                    {formatVideoLengthHms(runTimeSec(job))}
                  </td>
                  <td className="px-3 py-1">
                    {pct !== null ? (
                      <div className="flex min-w-0 items-center gap-2">
                        <div className="h-1.5 min-w-0 flex-1 overflow-hidden rounded bg-accent">
                          <div
                            className={cn(
                              "h-full transition-colors",
                              progressBarColor(pct, job.status),
                            )}
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                        <span className="w-8 shrink-0 text-right text-xs tabular-nums">
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
                  <td className="px-3 py-1 text-xs tabular-nums text-muted whitespace-nowrap">
                    {timeRemaining(job)}
                  </td>
                  <td className="px-3 py-1">
                    <div
                      className="flex flex-nowrap items-center justify-start gap-1"
                      onClick={(event) => event.stopPropagation()}
                    >
                      {openOutput ? (
                        <RoundIconButton
                          label="Open Output Directory"
                          variant="info"
                          size="xs"
                          onClick={() => onOpenOutput(job)}
                        >
                          <IconFolder size={14} />
                        </RoundIconButton>
                      ) : (
                        <>
                          <RoundIconButton
                            label="Review Prescan"
                            variant="info"
                            size="xs"
                            disabled={!review}
                            onClick={() => onReview(job)}
                          >
                            <IconReview size={14} />
                          </RoundIconButton>
                          <RoundIconButton
                            label="Restart (Overwrite)"
                            variant="accent"
                            size="xs"
                            disabled={!startFresh || busy}
                            onClick={() => onStartFresh(job.job_id)}
                          >
                            <IconRestart size={14} />
                          </RoundIconButton>
                          <RoundIconButton
                            label="Stop Job"
                            size="xs"
                            variant="danger"
                            disabled={!stop || busy}
                            onClick={() => onStop(job.job_id)}
                          >
                            <IconCancel size={14} />
                          </RoundIconButton>
                        </>
                      )}
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
