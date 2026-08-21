"use client";

import { useEffect, useMemo, useState } from "react";
import type { JobStatusResponse } from "@viana/contracts";

import {
  IconCancel,
  IconFolder,
  IconRestart,
  IconReview,
  RoundIconButton,
} from "@/components/ui/icon-button";
import {
  canOpenOutput,
  canStartFresh,
  isCancellable,
  isReviewable,
  statusBadgeClass,
  statusHint,
  statusLabel,
} from "@/features/queue/job-status";
import { videoStem } from "@/lib/geometry";
import {
  formatSubmittedAt,
  formatVideoLengthHms,
  runTimeSec,
  sortJobsBySubmitted,
} from "@/lib/job-local-meta";
import { progressBarColor } from "@/lib/progress-bar";
import { gpuIdFromDevice } from "@/lib/job-errors";
import { formatEta } from "@/lib/validation";
import { cn } from "@/lib/utils";

/** Visible body rows before the table body scrolls (header stays sticky). */
const VIEWPORT_ROWS = 10;
/** Matches dense ~2rem rows + sticky header; short pages keep this height locked. */
const VIEWPORT_HEIGHT = `calc(${VIEWPORT_ROWS} * 2rem + 1.75rem)`;

const PAGE_SIZE_OPTIONS = [10, 25, 50, "all"] as const;
type PageSizeOption = (typeof PAGE_SIZE_OPTIONS)[number];

function progressPct(job: JobStatusResponse): number | null {
  if (job.status === "COMPLETED") {
    return 100;
  }
  const total = job.progress?.total_frames;
  const current = job.progress?.current_frame;
  if (!total || total <= 0 || current === undefined) {
    return null;
  }
  return Math.min(100, Math.round((current / total) * 100));
}

function timeRemaining(job: JobStatusResponse): string {
  // API `eta_sec` = (total_frames - current_frame) / processing_fps (seconds).
  if (job.status !== "PROCESSING" || job.progress?.eta_sec === undefined) {
    return "—";
  }
  return formatEta(job.progress.eta_sec);
}

function pageSizeLabel(option: PageSizeOption): string {
  return option === "all" ? "All" : String(option);
}

/** Compact page list with ellipses (1 … 4 5 6 … 12). */
function visiblePageNumbers(current: number, total: number): Array<number | "…"> {
  if (total <= 7) {
    return Array.from({ length: total }, (_, i) => i + 1);
  }
  const pages = new Set<number>([1, total, current - 1, current, current + 1]);
  const sorted = [...pages].filter((p) => p >= 1 && p <= total).sort((a, b) => a - b);
  const out: Array<number | "…"> = [];
  for (const page of sorted) {
    const prev = out[out.length - 1];
    if (typeof prev === "number" && page - prev > 1) {
      out.push("…");
    }
    out.push(page);
  }
  return out;
}

export function JobQueueTable({
  jobs,
  busyId,
  selectedJobId,
  onSelectJob,
  onReview,
  onStartFresh,
  onStop,
  onOpenOutput,
}: {
  jobs: JobStatusResponse[];
  busyId: string | null;
  selectedJobId: string | null;
  onSelectJob: (job: JobStatusResponse) => void;
  onReview: (job: JobStatusResponse) => void;
  onStartFresh: (jobId: string) => void;
  onStop: (jobId: string) => void;
  onOpenOutput: (job: JobStatusResponse) => void;
}) {
  // ⚡ Bolt: Memoize the sorted list to prevent expensive O(N log N) sorting on every dashboard re-render (which happens frequently due to polling/telemetry).
  const sorted = useMemo(() => sortJobsBySubmitted(jobs), [jobs]);

  const [pageSize, setPageSize] = useState<PageSizeOption>(10);
  const [page, setPage] = useState(1);

  const pageSizeNum = pageSize === "all" ? Math.max(sorted.length, 1) : pageSize;
  const totalPages = Math.max(1, Math.ceil(sorted.length / pageSizeNum) || 1);
  const safePage = Math.min(page, totalPages);

  useEffect(() => {
    if (page !== safePage) {
      setPage(safePage);
    }
  }, [page, safePage]);

  const rangeStart = sorted.length === 0 ? 0 : (safePage - 1) * pageSizeNum + 1;
  const rangeEnd = Math.min(sorted.length, safePage * pageSizeNum);
  const pageJobs = sorted.slice(
    sorted.length === 0 ? 0 : (safePage - 1) * pageSizeNum,
    rangeEnd,
  );

  const pageNumbers = visiblePageNumbers(safePage, totalPages);

  return (
    <section className="rounded-lg border border-border bg-card p-4">
      <h2 className="text-sm font-semibold tracking-wide text-muted uppercase">
        Job Queue
      </h2>
      <p className="mt-1 text-xs text-muted">
        All jobs for this project · newest first
      </p>

      <div
        className="mt-3 overflow-auto overscroll-contain"
        style={{
          // Same height every page: short pages keep empty space; 10 rows fill without extra slack.
          height: VIEWPORT_HEIGHT,
          scrollbarGutter: "stable",
        }}
      >
        <table className="w-full table-fixed text-left text-sm leading-tight">
          <colgroup>
            <col style={{ width: "16%" }} />
            <col style={{ width: "13%" }} />
            <col style={{ width: "10%" }} />
            <col style={{ width: "5%" }} />
            <col style={{ width: "9%" }} />
            <col style={{ width: "9%" }} />
            <col style={{ width: "14%" }} />
            <col style={{ width: "12%" }} />
            <col style={{ width: "12%" }} />
          </colgroup>
          <thead>
            <tr className="border-b border-border text-xs text-muted">
              {(
                [
                  "Video",
                  "Submitted",
                  "Status",
                  "GPU",
                  "Video Length",
                  "Run Time",
                  "Progress",
                  "Time Remaining",
                  "Actions",
                ] as const
              ).map((label) => (
                <th
                  key={label}
                  className={cn(
                    "sticky top-0 z-10 bg-card px-3 py-1 font-medium shadow-[inset_0_-1px_0_0_var(--ui-border)]",
                    (label === "Video Length" ||
                      label === "Run Time" ||
                      label === "Time Remaining") &&
                      "whitespace-nowrap",
                  )}
                >
                  {label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {pageJobs.map((job) => {
              const pct = progressPct(job);
              const paused = job.status === "PAUSED";
              const busy = busyId === job.job_id;
              const review = isReviewable(job.status);
              const startFresh = canStartFresh(job.status);
              const openOutput = canOpenOutput(job.status);
              const stop = isCancellable(job.status);
              return (
                <tr
                  key={job.job_id}
                  className={cn(
                    "cursor-pointer border-b border-border align-middle hover:bg-card-hover",
                    paused && "bg-row-paused",
                    selectedJobId === job.job_id && "bg-row-selected",
                  )}
                  onClick={() => onSelectJob(job)}
                >
                  <td className="px-3 py-1">
                    <p
                      className="truncate font-mono text-xs leading-tight"
                      title={job.source_video_path}
                    >
                      {videoStem(job.source_video_path)}
                    </p>
                  </td>
                  <td className="px-3 py-1 text-xs text-muted whitespace-nowrap">
                    {formatSubmittedAt(job.created_at)}
                  </td>
                  <td className="px-3 py-1">
                    <span
                      title={statusHint(job.status)}
                      className={cn(
                        "inline-block max-w-full truncate whitespace-nowrap rounded px-1.5 py-0.5 text-[11px] font-medium leading-none",
                        statusBadgeClass(job.status),
                      )}
                    >
                      {statusLabel(job.status)}
                    </span>
                  </td>
                  <td className="px-3 py-1 font-mono text-xs text-muted">
                    {gpuIdFromDevice(job.gpu_device)}
                  </td>
                  <td className="px-3 py-1 font-mono text-xs text-muted whitespace-nowrap">
                    {formatVideoLengthHms(job.video_duration_sec)}
                  </td>
                  <td className="px-3 py-1 font-mono text-xs text-muted whitespace-nowrap">
                    {formatVideoLengthHms(runTimeSec(job))}
                  </td>
                  <td className="px-3 py-1">
                    {pct !== null ? (
                      <div className="flex min-w-0 items-center gap-2">
                        <div className="h-1.5 min-w-0 flex-1 overflow-hidden rounded bg-accent">
                          <div
                            className={cn(
                              "h-full transition-colors",
                              progressBarColor(pct, job.status),
                            )}
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                        <span className="w-8 shrink-0 text-right text-xs tabular-nums">
                          {pct}%
                        </span>
                      </div>
                    ) : job.queue_position ? (
                      <span className="text-xs whitespace-nowrap text-muted">
                        #{job.queue_position}
                      </span>
                    ) : (
                      <span className="text-xs text-muted">—</span>
                    )}
                  </td>
                  <td className="px-3 py-1 text-xs tabular-nums text-muted whitespace-nowrap">
                    {timeRemaining(job)}
                  </td>
                  <td className="px-3 py-1">
                    <div
                      className="flex flex-nowrap items-center justify-start gap-1"
                      onClick={(event) => event.stopPropagation()}
                    >
                      {openOutput ? (
                        <RoundIconButton
                          label="Open Output Directory"
                          variant="info"
                          size="xs"
                          onClick={() => onOpenOutput(job)}
                        >
                          <IconFolder size={14} />
                        </RoundIconButton>
                      ) : (
                        <>
                          <RoundIconButton
                            label="Review Prescan"
                            variant="info"
                            size="xs"
                            disabled={!review}
                            onClick={() => onReview(job)}
                          >
                            <IconReview size={14} />
                          </RoundIconButton>
                          <RoundIconButton
                            label="Restart (Overwrite)"
                            variant="accent"
                            size="xs"
                            disabled={!startFresh || busy}
                            onClick={() => onStartFresh(job.job_id)}
                          >
                            <IconRestart size={14} />
                          </RoundIconButton>
                          <RoundIconButton
                            label="Stop Job"
                            size="xs"
                            variant="danger"
                            disabled={!stop || busy}
                            onClick={() => onStop(job.job_id)}
                          >
                            <IconCancel size={14} />
                          </RoundIconButton>
                        </>
                      )}
                    </div>
                  </td>
                </tr>
              );
            })}
            {sorted.length === 0 ? (
              <tr>
                <td colSpan={9} className="py-6 text-center text-sm text-muted">
                  No jobs yet — add videos above.
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>

      <div className="mt-3 flex flex-wrap items-center justify-between gap-3 border-t border-border pt-3 text-xs text-muted">
        <label className="inline-flex items-center gap-2">
          <span className="whitespace-nowrap">Rows per page</span>
          <select
            className="h-7 rounded border border-input bg-card px-2 text-xs text-foreground"
            value={pageSizeLabel(pageSize)}
            aria-label="Rows per page"
            onChange={(event) => {
              const raw = event.target.value;
              const next: PageSizeOption =
                raw === "All" ? "all" : (Number(raw) as 10 | 25 | 50);
              setPageSize(next);
              setPage(1);
            }}
          >
            {PAGE_SIZE_OPTIONS.map((option) => (
              <option key={pageSizeLabel(option)} value={pageSizeLabel(option)}>
                {pageSizeLabel(option)}
              </option>
            ))}
          </select>
        </label>

        <p className="tabular-nums" aria-live="polite">
          {sorted.length === 0
            ? "0 of 0"
            : `${rangeStart}–${rangeEnd} of ${sorted.length}`}
        </p>

        <nav className="inline-flex items-center gap-1" aria-label="Job queue pages">
          <button
            type="button"
            className="inline-flex h-7 min-w-7 items-center justify-center rounded border border-border px-2 text-foreground disabled:cursor-not-allowed disabled:opacity-40"
            aria-label="Previous page"
            disabled={safePage <= 1 || sorted.length === 0}
            onClick={() => setPage((prev) => Math.max(1, prev - 1))}
          >
            Prev
          </button>
          {pageNumbers.map((item, index) =>
            item === "…" ? (
              <span
                key={`ellipsis-${index}`}
                className="inline-flex h-7 min-w-7 items-center justify-center px-1"
                aria-hidden
              >
                …
              </span>
            ) : (
              <button
                key={item}
                type="button"
                className={cn(
                  "inline-flex h-7 min-w-7 items-center justify-center rounded border px-2 tabular-nums",
                  item === safePage
                    ? "border-foreground/30 bg-accent font-medium text-foreground"
                    : "border-border text-foreground hover:bg-card-hover",
                )}
                aria-label={`Page ${item}`}
                aria-current={item === safePage ? "page" : undefined}
                disabled={sorted.length === 0}
                onClick={() => setPage(item)}
              >
                {item}
              </button>
            ),
          )}
          <button
            type="button"
            className="inline-flex h-7 min-w-7 items-center justify-center rounded border border-border px-2 text-foreground disabled:cursor-not-allowed disabled:opacity-40"
            aria-label="Next page"
            disabled={safePage >= totalPages || sorted.length === 0}
            onClick={() => setPage((prev) => Math.min(totalPages, prev + 1))}
          >
            Next
          </button>
        </nav>
      </div>
    </section>
  );
}
