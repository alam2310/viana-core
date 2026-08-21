"use client";

import type { JobStatusResponse, TelemetryMessage } from "@viana/contracts";

import { CrossingsTable } from "@/features/telemetry/crossings-table";
import {
  crossingsFromTelemetry,
  formatProgressLine,
  progressFromTelemetry,
} from "@/features/telemetry/telemetry-formatters";

/**
 * Authoritative total is `JobStatus.progress.crossing_count` from GET /jobs
 * plus WS PROGRESS (I002 / 6.10). Do not use the in-session MOVING_EVENT list
 * length — that list resets when the page or details view is opened late.
 */
export function liveCrossingCount(
  job: JobStatusResponse,
  messages: TelemetryMessage[],
): number | undefined {
  const fromWs = progressFromTelemetry(messages, job.job_id)?.crossingCount;
  if (typeof fromWs === "number") {
    return fromWs;
  }
  const fromStatus = job.progress?.crossing_count;
  return typeof fromStatus === "number" ? fromStatus : undefined;
}

export function LiveCrossings({
  job,
  messages,
}: {
  job: JobStatusResponse;
  messages: TelemetryMessage[];
}) {
  const progress = progressFromTelemetry(messages, job.job_id);
  const meta = job.confirmed_metadata ?? job.proposed_metadata;
  const crossings = crossingsFromTelemetry(messages, job.job_id, undefined, {
    startTime: meta?.user_start_time,
    startDate: meta?.user_start_date,
  });
  const crossingCount = liveCrossingCount(job, messages);

  if (job.status !== "PROCESSING") {
    return null;
  }

  const progressLine = progress
    ? formatProgressLine(progress)
    : job.progress
      ? `${job.progress.current_frame} / ${job.progress.total_frames} frames`
      : null;

  return (
    <div className="space-y-2">
      {progressLine ? (
        <p className="text-sm font-medium">{progressLine}</p>
      ) : null}
      <details className="rounded border border-border" open>
        <summary className="cursor-pointer px-3 py-1.5 text-xs font-semibold tracking-wide text-muted">
            Live Crossings
        </summary>
        <div className="border-t border-border px-1 pb-1">
          {crossings.length > 0 ? (
            <CrossingsTable rows={crossings} maxRows={30} />
          ) : (
            <p className="px-2 py-2 text-xs text-muted">
              {typeof crossingCount === "number" && crossingCount > 0
                ? "Recent rows appear here as new crossings arrive. The total above is from the job API."
                : "Crossings appear here when vehicles pass the counting line."}
            </p>
          )}
        </div>
      </details>
    </div>
  );
}
