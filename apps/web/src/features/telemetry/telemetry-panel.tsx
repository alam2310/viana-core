"use client";

import type { TelemetryMessage } from "@viana/contracts";

export function TelemetryPanel({
  messages,
  telemetryDetail,
  onTelemetryDetail,
}: {
  messages: TelemetryMessage[];
  telemetryDetail: boolean;
  onTelemetryDetail: (value: boolean) => void;
}) {
  return (
    <section className="rounded-lg border border-neutral-200 bg-white p-4">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-sm font-semibold tracking-wide text-neutral-500 uppercase">
          Telemetry
        </h2>
        <label className="flex items-center gap-2 text-xs">
          <input
            type="checkbox"
            checked={telemetryDetail}
            onChange={(event) => onTelemetryDetail(event.target.checked)}
          />
          telemetry_detail on next submit
        </label>
      </div>
      <pre className="mt-3 max-h-64 overflow-auto rounded bg-neutral-50 p-3 text-xs">
        {messages.length > 0
          ? JSON.stringify(messages.slice(-8), null, 2)
          : "waiting…"}
      </pre>
    </section>
  );
}
