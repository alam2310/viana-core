"use client";

import type { JobStatusResponse, TelemetryMessage } from "@viana/contracts";

import { Button } from "@/components/ui/button";
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
  const crossings = crossingsFromTelemetry(messages, job.job_id, fps);
  const refreshKey =
    progress?.current ?? job.progress?.current_frame ?? 0;

  return (
    <aside className="flex h-full flex-col rounded-lg border border-neutral-200 bg-white">
      <div className="flex items-center justify-between border-b border-neutral-200 px-4 py-3">
        <div>
          <h2 className="text-sm font-semibold">Live monitor</h2>
          <p className="font-mono text-xs text-neutral-500">
            {videoStem(job.source_video_path)}
          </p>
        </div>
        <Button type="button" size="sm" variant="ghost" onClick={onClose}>
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

        <details className="mt-4 rounded border border-neutral-200">
          <summary className="cursor-pointer px-3 py-2 text-xs font-semibold tracking-wide text-neutral-600 uppercase">
            Live crossings ({crossings.length})
          </summary>
          <div className="border-t border-neutral-200 px-1 pb-2">
            {crossings.length > 0 ? (
              <div className="max-h-48 overflow-y-auto">
                <table className="w-full text-left text-xs">
                  <thead className="sticky top-0 bg-neutral-50">
                    <tr>
                      <th className="px-2 py-1">Time</th>
                      <th className="px-2 py-1">Class</th>
                      <th className="px-2 py-1 text-center">Dir</th>
                    </tr>
                  </thead>
                  <tbody>
                    {crossings.slice(-30).reverse().map((row) => (
                      <tr key={row.id} className="border-t border-neutral-100">
                        <td className="px-2 py-1 font-mono">{row.time}</td>
                        <td className="px-2 py-1">{row.vehicle}</td>
                        <td className="px-2 py-1 text-center text-base">
                          {row.arrow}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="px-2 py-3 text-xs text-neutral-500">
                Crossings appear here when vehicles pass the counting line. If none
                show, the engine may not be emitting crossing events yet (see backend
                tracker).
              </p>
            )}
          </div>
        </details>
      </div>
    </aside>
  );
}
