"use client";

import type { JobStatusResponse, TelemetryMessage } from "@viana/contracts";

import {
  crossingsFromTelemetry,
  formatProgressLine,
  progressFromTelemetry,
} from "@/features/telemetry/telemetry-formatters";
import { videoStem } from "@/lib/geometry";
import {
  formatSubmittedAt,
  getJobLocalMeta,
} from "@/lib/job-local-meta";
import { formatJobErrorMessage } from "@/lib/job-errors";
import { statusLabel } from "@/features/queue/job-status";

function metadataBlock(job: JobStatusResponse) {
  const meta = job.confirmed_metadata ?? job.proposed_metadata;
  if (!meta?.user_start_time && !meta?.user_start_date && !meta?.location) {
    return <p className="text-sm text-neutral-500">No metadata yet — complete prescan review.</p>;
  }
  return (
    <dl className="grid gap-2 text-sm">
      <div>
        <dt className="text-neutral-500">Time</dt>
        <dd className="font-mono">{meta.user_start_time ?? "—"}</dd>
      </div>
      <div>
        <dt className="text-neutral-500">Date</dt>
        <dd className="font-mono">{meta.user_start_date ?? "—"}</dd>
      </div>
      <div>
        <dt className="text-neutral-500">Location</dt>
        <dd>{meta.location ?? "—"}</dd>
      </div>
    </dl>
  );
}

function artifactList(job: JobStatusResponse) {
  if (job.status !== "COMPLETED") {
    return null;
  }
  const stem = videoStem(job.source_video_path);
  const dir = job.output_dir;
  return (
    <div className="mt-3">
      <h3 className="text-xs font-semibold tracking-wide text-neutral-500 uppercase">
        Output files
      </h3>
      <ul className="mt-1 space-y-1 font-mono text-[10px] text-neutral-600">
        <li>{dir}/{stem}_events.csv</li>
        <li>{dir}/{stem}_15min.csv</li>
        <li>{dir}/{stem}_processed.mp4</li>
      </ul>
    </div>
  );
}

export function JobDetailsPanel({
  job,
  messages,
}: {
  job: JobStatusResponse | null;
  messages: TelemetryMessage[];
}) {
  if (!job) {
    return (
      <section className="rounded-lg border border-neutral-200 bg-white p-4">
        <h2 className="text-sm font-semibold tracking-wide text-neutral-500 uppercase">
          Job details
        </h2>
        <p className="mt-3 text-sm text-neutral-500">
          Select a job in the queue to view metadata, progress, and output paths.
        </p>
      </section>
    );
  }

  const local = getJobLocalMeta(job.job_id);
  const progress = progressFromTelemetry(messages, job.job_id);
  const crossings = crossingsFromTelemetry(
    messages,
    job.job_id,
    job.progress?.processing_fps ?? progress?.fps,
  );
  const errorText = formatJobErrorMessage(job.error_message);

  return (
    <section className="rounded-lg border border-neutral-200 bg-white p-4">
      <h2 className="text-sm font-semibold tracking-wide text-neutral-500 uppercase">
        Job details
      </h2>
      <p className="mt-1 font-mono text-xs text-neutral-500">{job.job_id}</p>

      <div className="mt-3 space-y-3 text-sm">
        <p>
          <span className="text-neutral-500">Status:</span> {statusLabel(job.status)}
        </p>
        <p>
          <span className="text-neutral-500">Submitted:</span>{" "}
          {formatSubmittedAt(local.submittedAt)}
        </p>
        {errorText ? <p className="text-red-700">{errorText}</p> : null}

        <div>
          <h3 className="text-xs font-semibold tracking-wide text-neutral-500 uppercase">
            Video metadata
          </h3>
          <div className="mt-2">{metadataBlock(job)}</div>
        </div>

        {progress || job.progress ? (
          <p className="font-medium">
            {progress
              ? formatProgressLine(progress)
              : `${job.progress?.current_frame ?? 0} / ${job.progress?.total_frames ?? "?"} frames`}
          </p>
        ) : null}

        {crossings.length > 0 ? (
          <details className="rounded border border-neutral-200">
            <summary className="cursor-pointer px-3 py-2 text-xs font-semibold text-neutral-600">
              Recent crossings ({crossings.length})
            </summary>
            <ul className="max-h-32 overflow-y-auto border-t border-neutral-100 px-3 py-2 text-xs">
              {crossings.slice(-8).reverse().map((row) => (
                <li key={row.id}>
                  {row.time} · {row.arrow} {row.vehicle}
                </li>
              ))}
            </ul>
          </details>
        ) : null}

        {artifactList(job)}
      </div>
    </section>
  );
}
