"use client";

import type { JobStatusResponse, TelemetryMessage } from "@viana/contracts";

import { Button } from "@/components/ui/button";
import { CrossingsTable } from "@/features/telemetry/crossings-table";
import {
  crossingsFromTelemetry,
  formatProgressLine,
  progressFromTelemetry,
} from "@/features/telemetry/telemetry-formatters";
import { videoStem } from "@/lib/geometry";

/*
 * PARKED (2026-08-20): Live processed-MP4 preview is disabled in this view.
 * Do not import or mount `LiveProcessedVideo` / `crossing-media-sync` until
 * that work is un-parked — see docs/steps/STABILIZATION_BACKLOG.md (S20 / S24)
 * and file headers on:
 *   - apps/web/src/features/monitor/live-processed-video.tsx
 *   - apps/web/src/features/monitor/crossing-media-sync.ts
 * Live Crossings render WS telemetry immediately (no UI delay / frame gate).
 */

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
  const meta = job.confirmed_metadata ?? job.proposed_metadata;
  const crossings = crossingsFromTelemetry(messages, job.job_id, undefined, {
    startTime: meta?.user_start_time,
    startDate: meta?.user_start_date,
  });

  return (
    <aside className="flex h-full flex-col rounded-lg border border-border bg-card">
      <div className="flex items-center justify-between border-b border-border px-4 py-3">
        <div>
          <h2 className="text-sm font-semibold">Live Monitor</h2>
          <p className="font-mono text-xs text-muted">
            {videoStem(job.source_video_path)}
          </p>
        </div>
        <Button type="button" size="sm" variant="ghost" onClick={onClose}>
          Close
        </Button>
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        <p className="text-sm font-medium">
          {progress
            ? formatProgressLine(progress)
            : job.progress
              ? `${job.progress.current_frame ?? 0} / ${job.progress.total_frames ?? "?"} frames`
              : "Waiting for progress…"}
        </p>

        <details className="mt-4 rounded border border-border" open>
          <summary className="cursor-pointer px-3 py-1.5 text-xs font-semibold tracking-wide text-muted">
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
