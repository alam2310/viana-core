"use client";

import type { JobStatusResponse, TelemetryMessage } from "@viana/contracts";

import { Aggregate15MinTable } from "@/features/telemetry/aggregate-15min-table";
import {
  LiveCrossings,
  liveCrossingCount,
} from "@/features/telemetry/live-crossings";
import { progressFromTelemetry } from "@/features/telemetry/telemetry-formatters";
import {
  statusBadgeClass,
  statusHint,
  statusLabel,
} from "@/features/queue/job-status";
import type { MountConfig } from "@/lib/container-paths";
import { toHostPath } from "@/lib/container-paths";
import { videoStem } from "@/lib/geometry";
import { formatSubmittedAt } from "@/lib/job-local-meta";
import { formatJobErrorMessage } from "@/lib/job-errors";
import { openPathInFileManager } from "@/lib/fs-open";
import { cn } from "@/lib/utils";

function completedOutputFiles(
  job: JobStatusResponse,
  mountConfig: MountConfig,
): { label: string; hostPath: string | null }[] {
  const stem = videoStem(job.source_video_path);
  const base = job.output_dir;
  const entries = [
    { label: "Processed Video", file: `${stem}_processed.mp4` },
    { label: "Raw events", file: `${stem}_events.csv` },
    { label: "15-mins report", file: `${stem}_15min.csv` },
  ];
  return entries.map(({ label, file }) => ({
    label,
    hostPath: toHostPath(`${base}/${file}`, mountConfig.mounts),
  }));
}

function OutputFilesList({
  job,
  mountConfig,
}: {
  job: JobStatusResponse;
  mountConfig: MountConfig;
}) {
  const files = completedOutputFiles(job, mountConfig);

  return (
    <div className="space-y-1 text-sm">
      <p className="text-xs font-semibold text-muted">Output files</p>
      <ul className="space-y-1">
        {files.map((file) => (
          <li key={file.label}>
            {file.hostPath ? (
              <button
                type="button"
                className="cursor-pointer text-left text-primary underline hover:opacity-80 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-border focus-visible:ring-offset-1 rounded-sm"
                onClick={() => {
                  void openPathInFileManager(file.hostPath!).catch(() => undefined);
                }}
              >
                {file.label}
              </button>
            ) : (
              <span className="text-muted">{file.label}</span>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

function metadataBlock(job: JobStatusResponse) {
  const meta = job.confirmed_metadata ?? job.proposed_metadata;
  if (!meta?.user_start_time && !meta?.user_start_date && !meta?.location) {
    return (
      <p className="text-sm text-muted">
        No metadata yet — complete prescan review.
      </p>
    );
  }
  return (
    <>
      <p>
        <span className="text-muted">Video Start Date:</span>{" "}
        <span className="font-mono">{meta.user_start_date ?? "—"}</span>
      </p>
      <p>
        <span className="text-muted">Video Start Time:</span>{" "}
        <span className="font-mono">{meta.user_start_time ?? "—"}</span>
      </p>
      <p>
        <span className="text-muted">Location:</span> {meta.location ?? "—"}
      </p>
    </>
  );
}


export function JobDetailsPanel({
  job,
  mountConfig,
  messages,
}: {
  job: JobStatusResponse | null;
  mountConfig: MountConfig | null;
  messages: TelemetryMessage[];
}) {
  if (!job) {
    return (
      <section className="rounded-lg border border-border bg-card p-4">
        <h2 className="text-sm font-semibold tracking-wide text-muted uppercase">
          Job details
        </h2>
        <p className="mt-3 text-sm text-muted">
          Select a job in the queue to view metadata, progress, and output paths.
        </p>
      </section>
    );
  }

  // PAUSED/CANCELLED may keep error_message="interrupted" as an internal marker — not an operator error.
  // CHECKPOINT_EXISTS (Partial): show a fixed overwrite guidance banner.
  const PARTIAL_EXISTING_OUTPUT_MESSAGE =
    "This video file was processed previously, and output files already exist. You can force restart the job to overwrite the existing files and process the video from the beginning.";
  const errorText =
    job.status === "PAUSED" || job.status === "CANCELLED"
      ? null
      : job.status === "CHECKPOINT_EXISTS"
        ? PARTIAL_EXISTING_OUTPUT_MESSAGE
        : formatJobErrorMessage(job.error_message);
  const stem = videoStem(job.source_video_path);
  const csvHostPath =
    job.status === "COMPLETED" && mountConfig
      ? toHostPath(`${job.output_dir}/${stem}_15min.csv`, mountConfig.mounts)
      : null;
  const telemetryProgress = progressFromTelemetry(messages, job.job_id);
  const totalFrames =
    telemetryProgress?.total ?? job.progress?.total_frames;
  const crossingCount = liveCrossingCount(job, messages);
  const totalFramesLabel =
    typeof totalFrames === "number" ? String(totalFrames) : "—";
  const crossingsLabel =
    typeof crossingCount === "number" ? String(crossingCount) : "—";

  return (
    <section className="rounded-lg border border-border bg-card p-4">
      <h2 className="text-sm font-semibold tracking-wide text-muted uppercase">
        Job details
      </h2>
      <p className="mt-1 font-mono text-xs text-muted">{job.job_id}</p>

      <div className="mt-3 space-y-2 text-sm leading-snug">
        <p className="flex items-center gap-2">
          <span className="text-muted">Status:</span>
          <span
            title={statusHint(job.status)}
            className={cn(
              "inline-block rounded px-1.5 py-0.5 text-xs font-medium",
              statusBadgeClass(job.status),
            )}
          >
            {statusLabel(job.status)}
          </span>
        </p>
        <p>
          <span className="text-muted">Submitted:</span>{" "}
          {formatSubmittedAt(job.created_at)}
        </p>
        {errorText ? (
          <div className="rounded border border-red-300 bg-red-50 px-3 py-2 text-sm text-red-800 dark:border-red-800 dark:bg-red-950 dark:text-red-200">
            {errorText}
          </div>
        ) : null}

        {metadataBlock(job)}

        <p>
          <span className="text-muted">Total frames:</span>{" "}
          <span className="font-mono">{totalFramesLabel}</span>
        </p>
        <p>
          <span className="text-muted">Crossings detected:</span>{" "}
          <span className="font-mono">{crossingsLabel}</span>
        </p>

        <LiveCrossings job={job} messages={messages} />

        {job.status === "COMPLETED" && csvHostPath ? (
          <details className="rounded border border-border">
            <summary className="cursor-pointer px-3 py-1.5 text-xs font-semibold text-muted">
              15-minute report
            </summary>
            <div className="border-t border-border px-1 pb-1">
              <Aggregate15MinTable csvHostPath={csvHostPath} />
            </div>
          </details>
        ) : null}

        {job.status === "COMPLETED" && mountConfig ? (
          <OutputFilesList job={job} mountConfig={mountConfig} />
        ) : null}
      </div>
    </section>
  );
}
