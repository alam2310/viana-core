"use client";

import { useState } from "react";
import type { JobStatusResponse, TelemetryMessage } from "@viana/contracts";

import { Button } from "@/components/ui/button";
import {
  activityFromTelemetry,
  crossingsFromTelemetry,
  formatProgressLine,
  progressFromTelemetry,
} from "@/features/telemetry/telemetry-formatters";
import { partialVideoUrl } from "@/lib/api-client";
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
  const [showRaw, setShowRaw] = useState(false);
  const progress = progressFromTelemetry(messages, job.job_id);
  const crossings = crossingsFromTelemetry(messages, job.job_id);
  const activity = activityFromTelemetry(messages, job.job_id);
  const jobMessages = messages.filter((msg) => msg.job_id === job.job_id);

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
        <video
          key={job.job_id}
          className="w-full rounded border border-neutral-300 bg-black"
          controls
          src={partialVideoUrl(job.job_id)}
        >
          <track kind="captions" />
        </video>

        <p className="mt-3 text-sm font-medium">
          {progress
            ? formatProgressLine(progress)
            : job.progress
              ? `${job.progress.current_frame ?? 0} / ${job.progress.total_frames ?? "?"} frames`
              : "Waiting for progress…"}
        </p>

        {crossings.length > 0 ? (
          <div className="mt-4">
            <h3 className="text-xs font-semibold tracking-wide text-neutral-500 uppercase">
              Crossing feed
            </h3>
            <div className="mt-2 max-h-40 overflow-y-auto rounded border border-neutral-200">
              <table className="w-full text-left text-xs">
                <thead className="sticky top-0 bg-neutral-50">
                  <tr>
                    <th className="px-2 py-1">Time</th>
                    <th className="px-2 py-1">Vehicle</th>
                    <th className="px-2 py-1">Direction</th>
                    <th className="px-2 py-1">Track</th>
                  </tr>
                </thead>
                <tbody>
                  {crossings.slice(-20).reverse().map((row) => (
                    <tr key={row.id} className="border-t border-neutral-100">
                      <td className="px-2 py-1 font-mono">{row.time}</td>
                      <td className="px-2 py-1">{row.vehicle}</td>
                      <td className="px-2 py-1">{row.direction}</td>
                      <td className="px-2 py-1 text-neutral-400">{row.trackId}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ) : null}

        {activity.length > 0 ? (
          <div className="mt-4">
            <h3 className="text-xs font-semibold tracking-wide text-neutral-500 uppercase">
              Activity log
            </h3>
            <ul className="mt-2 max-h-32 space-y-1 overflow-y-auto text-xs text-neutral-700">
              {activity.slice(-10).reverse().map((row) => (
                <li key={row.id}>{row.text}</li>
              ))}
            </ul>
          </div>
        ) : null}

        <details className="mt-4">
          <summary
            className="cursor-pointer text-xs text-neutral-500"
            onClick={() => setShowRaw((prev) => !prev)}
          >
            Raw JSON (collapsed)
          </summary>
          {showRaw ? (
            <pre className="mt-2 max-h-40 overflow-auto rounded bg-neutral-50 p-2 text-[10px]">
              {JSON.stringify(jobMessages.slice(-8), null, 2)}
            </pre>
          ) : null}
        </details>
      </div>
    </aside>
  );
}
