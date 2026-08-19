"use client";

import type { TelemetryMessage } from "@viana/contracts";

import {
  activityFromTelemetry,
  crossingsFromTelemetry,
  formatProgressLine,
  progressFromTelemetry,
} from "@/features/telemetry/telemetry-formatters";

export function TelemetryPanel({
  messages,
  focusedJobId,
  telemetryDetail,
  onTelemetryDetail,
}: {
  messages: TelemetryMessage[];
  focusedJobId: string | null;
  telemetryDetail: boolean;
  onTelemetryDetail: (value: boolean) => void;
}) {
  const jobId = focusedJobId;
  const progress = jobId ? progressFromTelemetry(messages, jobId) : null;
  const crossings = jobId ? crossingsFromTelemetry(messages, jobId) : [];
  const activity = jobId ? activityFromTelemetry(messages, jobId) : [];

  return (
    <section className="rounded-lg border border-border bg-card p-4">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-sm font-semibold tracking-wide text-muted uppercase">
          Telemetry
        </h2>
        <label className="flex items-center gap-2 text-xs">
          <input
            type="checkbox"
            checked={telemetryDetail}
            onChange={(event) => onTelemetryDetail(event.target.checked)}
          />
          telemetry_detail on next confirm
        </label>
      </div>

      {!jobId ? (
        <p className="mt-3 text-sm text-muted">
          Open Monitor on a processing job to view structured telemetry.
        </p>
      ) : (
        <div className="mt-3 space-y-3">
          <p className="text-sm font-medium">
            {progress ? formatProgressLine(progress) : "No progress yet"}
          </p>
          {crossings.length > 0 ? (
            <div>
              <h3 className="text-xs font-medium text-muted">Recent crossings</h3>
              <ul className="mt-1 max-h-24 overflow-y-auto text-xs">
                {crossings.slice(-5).reverse().map((row) => (
                  <li key={row.id}>
                    {row.vehicle} · {row.direction} · {row.time}
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
          {activity.length > 0 ? (
            <div>
              <h3 className="text-xs font-medium text-muted">Activity</h3>
              <ul className="mt-1 text-xs text-muted">
                {activity.slice(-3).reverse().map((row) => (
                  <li key={row.id}>{row.text}</li>
                ))}
              </ul>
            </div>
          ) : null}
        </div>
      )}
    </section>
  );
}
