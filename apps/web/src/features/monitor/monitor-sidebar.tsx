"use client";

import type { JobStatusResponse, TelemetryMessage } from "@viana/contracts";

import { Button } from "@/components/ui/button";
import { CrossingsTable } from "@/features/telemetry/crossings-table";
import { LiveProcessedVideo } from "@/features/monitor/live-processed-video";
import {
  crossingsFromTelemetry,
  formatProgressLine,
  progressFromTelemetry,
} from "@/features/telemetry/telemetry-formatters";
import { videoStem } from "@/lib/geometry";

export function MonitorSidebar({
  job,
  messages,
  onClose,
}: {
  job: JobStatusResponse;
  messages: TelemetryMessage[];
  onClose: () => void;
}) {
  const progress = progressFromTelemetry(messages, job.job_id);
  const fps = job.progress?.processing_fps ?? progress?.fps;
  const meta = job.confirmed_metadata ?? job.proposed_metadata;
  const crossings = crossingsFromTelemetry(messages, job.job_id, fps, {
    startTime: meta?.user_start_time,
    startDate: meta?.user_start_date,
  });
  const refreshKey =
    progress?.current ?? job.progress?.current_frame ?? 0;

  return (
    <aside className="flex h-full flex-col rounded-lg border border-border bg-card">
      <div className="flex items-center justify-between border-b border-border px-4 py-3">
        <div>
          <h2 className="text-sm font-semibold">Live monitor</h2>
          <p className="font-mono text-xs text-muted">
            {videoStem(job.source_video_path)}
          </p>
        </div>
        <Button
          type="button"
          size="sm"
          variant="ghost"
          className="dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-200"
          onClick={onClose}
        >
          Close
        </Button>
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        <LiveProcessedVideo jobId={job.job_id} refreshKey={refreshKey} />

        <p className="mt-3 text-sm font-medium">
          {progress
            ? formatProgressLine(progress)
            : job.progress
              ? `${job.progress.current_frame ?? 0} / ${job.progress.total_frames ?? "?"} frames`
              : "Waiting for progress…"}
        </p>

        <details className="mt-4 rounded border border-border">
          <summary className="cursor-pointer px-3 py-1.5 text-xs font-semibold tracking-wide text-muted uppercase">
            Live Crossings ({crossings.length})
          </summary>
          <div className="border-t border-border px-1 pb-1">
            {crossings.length > 0 ? (
              <CrossingsTable rows={crossings} maxRows={30} />
            ) : (
              <p className="px-2 py-2 text-xs text-muted">
                Crossings appear here when vehicles pass the counting line.
              </p>
            )}
          </div>
        </details>
      </div>
    </aside>
  );
}
