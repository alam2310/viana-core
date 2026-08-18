"use client";

import type { JobStatusResponse } from "@viana/contracts";

import { Button } from "@/components/ui/button";
import { videoStem } from "@/lib/geometry";
import { cn } from "@/lib/utils";

function progressPct(job: JobStatusResponse): number | null {
  const total = job.progress?.total_frames;
  const current = job.progress?.current_frame;
  if (!total || total <= 0 || current === undefined) {
    return null;
  }
  return Math.min(100, Math.round((current / total) * 100));
}

export function QueuePanel({
  jobs,
  pendingPaths,
  newPath,
  onNewPath,
  onAddPending,
  onRemovePending,
  onPrescan,
  onSubmitPending,
  onResume,
  onStartFresh,
  busyId,
}: {
  jobs: JobStatusResponse[];
  pendingPaths: string[];
  newPath: string;
  onNewPath: (value: string) => void;
  onAddPending: () => void;
  onRemovePending: (path: string) => void;
  onPrescan: (path: string) => void;
  onSubmitPending: (path: string) => void;
  onResume: (jobId: string) => void;
  onStartFresh: (jobId: string) => void;
  busyId: string | null;
}) {
  return (
    <section className="rounded-lg border border-neutral-200 bg-white p-4">
      <h2 className="text-sm font-semibold tracking-wide text-neutral-500 uppercase">
        Queue
      </h2>
      <p className="mt-1 text-xs text-neutral-500">
        Pending paths stay in localStorage. Submitted jobs sync from GET /jobs.
      </p>

      <form
        className="mt-3 flex gap-2"
        onSubmit={(event) => {
          event.preventDefault();
          onAddPending();
        }}
      >
        <input
          className="min-w-0 flex-1 rounded border border-neutral-300 px-2 py-1 font-mono text-xs"
          placeholder="/data/projects/nh48/videos/clip.mp4"
          value={newPath}
          onChange={(event) => onNewPath(event.target.value)}
        />
        <Button type="submit" size="sm">
          Add path
        </Button>
      </form>

      <ul className="mt-3 divide-y divide-neutral-100">
        {pendingPaths.map((path) => (
          <li key={path} className="flex flex-col gap-2 py-3">
            <p className="truncate font-mono text-xs">{path}</p>
            <div className="flex flex-wrap gap-2">
              <Button type="button" size="sm" variant="outline" onClick={() => onPrescan(path)}>
                Prescan
              </Button>
              <Button type="button" size="sm" onClick={() => onSubmitPending(path)}>
                Submit
              </Button>
              <Button
                type="button"
                size="sm"
                variant="ghost"
                onClick={() => onRemovePending(path)}
              >
                Remove
              </Button>
            </div>
          </li>
        ))}
      </ul>

      <h3 className="mt-4 text-xs font-semibold tracking-wide text-neutral-500 uppercase">
        Backend jobs
      </h3>
      <ul className="mt-2 divide-y divide-neutral-100">
        {jobs.map((job) => {
          const pct = progressPct(job);
          const paused = job.status === "PAUSED";
          const failed = job.status === "FAILED";
          return (
            <li
              key={job.job_id}
              className={cn(
                "py-3",
                paused && "rounded bg-amber-50 px-2 ring-2 ring-amber-400",
              )}
            >
              <div className="flex items-baseline justify-between gap-3">
                <span className="font-mono text-sm">{job.job_id}</span>
                <span
                  className={cn(
                    "text-sm",
                    paused && "font-semibold text-amber-800",
                    job.status === "PROCESSING" && "text-emerald-700",
                  )}
                >
                  {job.status}
                </span>
              </div>
              <p className="mt-1 truncate text-xs text-neutral-500">
                {job.source_video_path}
              </p>
              {pct !== null ? (
                <div className="mt-2 h-1.5 overflow-hidden rounded bg-neutral-200">
                  <div
                    className="h-full bg-neutral-900"
                    style={{ width: `${pct}%` }}
                  />
                </div>
              ) : null}
              {job.progress ? (
                <p className="mt-1 text-xs text-neutral-600">
                  Frame {job.progress.current_frame} / {job.progress.total_frames}
                  {job.progress.processing_fps
                    ? ` · ${job.progress.processing_fps} fps`
                    : ""}
                </p>
              ) : null}
              {job.status === "COMPLETED" ? (
                <p className="mt-1 text-xs text-neutral-600">
                  {job.output_dir}/{videoStem(job.source_video_path)}_events.csv
                </p>
              ) : null}
              {(paused || (failed && job.checkpoint_exists)) && (
                <div className="mt-2 flex gap-2">
                  {paused ? (
                    <Button
                      type="button"
                      size="sm"
                      disabled={busyId === job.job_id}
                      onClick={() => onResume(job.job_id)}
                    >
                      Resume
                    </Button>
                  ) : null}
                  <Button
                    type="button"
                    size="sm"
                    variant="danger"
                    disabled={busyId === job.job_id}
                    onClick={() => onStartFresh(job.job_id)}
                  >
                    Start fresh
                  </Button>
                </div>
              )}
            </li>
          );
        })}
        {jobs.length === 0 ? (
          <li className="py-3 text-sm text-neutral-500">No jobs</li>
        ) : null}
      </ul>
    </section>
  );
}
