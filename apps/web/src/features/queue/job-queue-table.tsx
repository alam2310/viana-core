"use client";

import type { JobStatusResponse } from "@viana/contracts";

import { Button } from "@/components/ui/button";
import {
  isCancellable,
  isReviewable,
  sortJobsFifo,
  statusLabel,
} from "@/features/queue/job-status";
import { videoStem } from "@/lib/geometry";
import { formatEta } from "@/lib/validation";
import { cn } from "@/lib/utils";

const EMPTY_15MIN_COPY =
  "Start time was not set — 15-minute report is unavailable.";

function progressPct(job: JobStatusResponse): number | null {
  const total = job.progress?.total_frames;
  const current = job.progress?.current_frame;
  if (!total || total <= 0 || current === undefined) {
    return null;
  }
  return Math.min(100, Math.round((current / total) * 100));
}

function metadataSummary(job: JobStatusResponse): string {
  const meta = job.confirmed_metadata ?? job.proposed_metadata;
  if (!meta?.user_start_time && !meta?.user_start_date && !meta?.location) {
    return "—";
  }
  return [meta.user_start_time, meta.user_start_date, meta.location]
    .filter(Boolean)
    .join(" · ");
}

function artifactPaths(job: JobStatusResponse): {
  events: string;
  aggregate: string;
  processed: string;
} {
  const stem = videoStem(job.source_video_path);
  const dir = job.output_dir;
  return {
    events: `${dir}/${stem}_events.csv`,
    aggregate: `${dir}/${stem}_15min.csv`,
    processed: `${dir}/${stem}_processed.mp4`,
  };
}

export function JobQueueTable({
  jobs,
  busyId,
  monitorJobId,
  onReview,
  onMonitor,
  onRetryPrescan,
  onResume,
  onStartFresh,
  onCancel,
}: {
  jobs: JobStatusResponse[];
  busyId: string | null;
  monitorJobId: string | null;
  onReview: (job: JobStatusResponse) => void;
  onMonitor: (job: JobStatusResponse) => void;
  onRetryPrescan: (jobId: string) => void;
  onResume: (jobId: string) => void;
  onStartFresh: (jobId: string) => void;
  onCancel: (jobId: string) => void;
}) {
  const sorted = sortJobsFifo(jobs);

  return (
    <section className="rounded-lg border border-neutral-200 bg-white p-4">
      <h2 className="text-sm font-semibold tracking-wide text-neutral-500 uppercase">
        Job queue
      </h2>
      <p className="mt-1 text-xs text-neutral-500">
        FIFO execution after prescan review. Synced from GET /jobs.
      </p>

      <div className="mt-3 overflow-x-auto">
        <table className="w-full min-w-[720px] text-left text-sm">
          <thead>
            <tr className="border-b border-neutral-200 text-xs text-neutral-500">
              <th className="py-2 pr-3 font-medium">Video</th>
              <th className="py-2 pr-3 font-medium">Status</th>
              <th className="py-2 pr-3 font-medium">Time · Date · Location</th>
              <th className="py-2 pr-3 font-medium">Progress</th>
              <th className="py-2 font-medium">Actions</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((job) => {
              const pct = progressPct(job);
              const paused = job.status === "PAUSED";
              const artifacts = artifactPaths(job);
              const showEmpty15min =
                job.status === "COMPLETED" && !job.confirmed_metadata?.user_start_time;
              return (
                <tr
                  key={job.job_id}
                  className={cn(
                    "border-b border-neutral-100 align-top",
                    paused && "bg-amber-50",
                    monitorJobId === job.job_id && "bg-emerald-50",
                  )}
                >
                  <td className="py-3 pr-3">
                    <p className="font-mono text-xs" title={job.source_video_path}>
                      {videoStem(job.source_video_path)}
                    </p>
                    <p className="mt-0.5 truncate font-mono text-[10px] text-neutral-400">
                      {job.source_video_path}
                    </p>
                  </td>
                  <td className="py-3 pr-3">
                    <span
                      className={cn(
                        "inline-block rounded px-2 py-0.5 text-xs font-medium",
                        job.status === "AWAITING_REVIEW" && "bg-amber-100 text-amber-900",
                        job.status === "PROCESSING" && "bg-emerald-100 text-emerald-900",
                        job.status === "PRESCAN_FAILED" && "bg-red-100 text-red-900",
                        job.status === "READY" && "bg-blue-100 text-blue-900",
                      )}
                    >
                      {statusLabel(job.status)}
                    </span>
                    {job.gpu_device ? (
                      <p className="mt-1 text-[10px] text-neutral-500">{job.gpu_device}</p>
                    ) : null}
                    {job.error_message ? (
                      <p className="mt-1 text-xs text-red-700">{job.error_message}</p>
                    ) : null}
                  </td>
                  <td className="py-3 pr-3 text-xs text-neutral-600">
                    {metadataSummary(job)}
                  </td>
                  <td className="py-3 pr-3 text-xs text-neutral-600">
                    {job.status === "PROCESSING" && pct !== null ? (
                      <>
                        <div className="h-1.5 w-24 overflow-hidden rounded bg-neutral-200">
                          <div
                            className="h-full bg-neutral-900"
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                        <p className="mt-1">
                          {pct}%
                          {job.progress?.processing_fps
                            ? ` · ${job.progress.processing_fps.toFixed(1)} fps`
                            : ""}
                          {job.progress?.eta_sec !== undefined
                            ? ` · ${formatEta(job.progress.eta_sec)}`
                            : ""}
                          {job.progress?.crossing_count !== undefined
                            ? ` · ${job.progress.crossing_count} crossings`
                            : ""}
                        </p>
                      </>
                    ) : job.queue_position ? (
                      <span>Queue #{job.queue_position}</span>
                    ) : (
                      "—"
                    )}
                  </td>
                  <td className="py-3">
                    <div className="flex flex-wrap gap-1">
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
                          Retry prescan
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
                          Start fresh
                        </Button>
                      )}
                      {isCancellable(job.status) ? (
                        <Button
                          type="button"
                          size="sm"
                          variant="ghost"
                          disabled={busyId === job.job_id}
                          onClick={() => onCancel(job.job_id)}
                        >
                          Cancel
                        </Button>
                      ) : null}
                    </div>
                    {job.status === "COMPLETED" ? (
                      <div className="mt-2 space-y-1 text-[10px] font-mono text-neutral-600">
                        <p>{artifacts.events}</p>
                        <p>{artifacts.aggregate}</p>
                        <p>{artifacts.processed}</p>
                        {showEmpty15min ? (
                          <p className="mt-1 font-sans text-xs text-amber-800">
                            {EMPTY_15MIN_COPY}
                          </p>
                        ) : null}
                      </div>
                    ) : null}
                  </td>
                </tr>
              );
            })}
            {sorted.length === 0 ? (
              <tr>
                <td colSpan={5} className="py-6 text-center text-sm text-neutral-500">
                  No jobs — add videos via intake above.
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
    </section>
  );
}
