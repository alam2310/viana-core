"use client";

import { useEffect, useState } from "react";
import type {
  ConfirmedJobMetadata,
  JobStatusResponse,
  LineSegment,
  VideoMeta,
} from "@viana/contracts";

import {
  CalibrationCanvas,
  formatLineCoords,
} from "@/features/calibration/calibration-canvas";
import { initialVideoMeta, estimateVideoMetaFromLines } from "@/features/prescan/prescan-meta";
import { Button } from "@/components/ui/button";
import {
  confirmPrescan,
  prescanPreview,
  previewImageUrl,
  saveProfile,
  sourceVideoUrl,
} from "@/lib/api-client";
import {
  clampLine,
  defaultCalibrationLines,
  scaleLine,
  validateCalibration,
} from "@/lib/geometry";
import { cn } from "@/lib/utils";
import {
  DATE_PATTERN,
  TIME_PATTERN,
  validateMetadataFields,
} from "@/lib/validation";

const DARK_GHOST_BTN =
  "dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-200";
const DARK_OUTLINE_BTN =
  "dark:border-zinc-300 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-200";

function mergeOcrMetadata(
  prev: {
    user_start_time: string;
    user_start_date: string;
    location: string;
  },
  ocr: {
    time?: string | null;
    date?: string | null;
    location?: string | null;
  },
  replace: boolean,
): {
  user_start_time: string;
  user_start_date: string;
  location: string;
} {
  if (replace) {
    return {
      user_start_time: ocr.time ?? prev.user_start_time,
      user_start_date: ocr.date ?? prev.user_start_date,
      location: ocr.location ?? prev.location,
    };
  }
  return {
    user_start_time: ocr.time ?? prev.user_start_time,
    user_start_date: ocr.date ?? prev.user_start_date,
    location: ocr.location ?? prev.location,
  };
}

export function PrescanReviewModal({
  job,
  projectId,
  awaitingReviewJobs,
  onClose,
  onConfirmed,
}: {
  job: JobStatusResponse;
  projectId: string;
  awaitingReviewJobs: JobStatusResponse[];
  onClose: () => void;
  onConfirmed: (jobId: string) => void;
}) {
  const [loading, setLoading] = useState(false);
  const [frameLoading, setFrameLoading] = useState(false);
  const [rescanning, setRescanning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [offset, setOffset] = useState(0);
  const [previewToken, setPreviewToken] = useState(0);
  const [videoMeta, setVideoMeta] = useState<VideoMeta | null>(
    () => initialVideoMeta(job),
  );
  const [previewPath, setPreviewPath] = useState<string | null>(
    job.proposed_preview_url ?? null,
  );
  const [metadata, setMetadata] = useState({
    user_start_time: job.proposed_metadata?.user_start_time ?? "",
    user_start_date: job.proposed_metadata?.user_start_date ?? "",
    location: job.proposed_metadata?.location ?? "",
  });
  const [horizon, setHorizon] = useState<LineSegment | null>(
    job.proposed_lines?.horizon_line ?? null,
  );
  const [counting, setCounting] = useState<LineSegment | null>(
    job.proposed_lines?.counting_line ?? null,
  );
  const [applyToOthers, setApplyToOthers] = useState(false);
  const [saveAsProfile, setSaveAsProfile] = useState(false);
  const [profileId, setProfileId] = useState("session");

  const previewUrl = previewPath
    ? previewImageUrl(previewPath, previewToken)
    : null;

  useEffect(() => {
    setLoading(false);
    setError(null);

    const estimated = initialVideoMeta(job);
    if (estimated) {
      setVideoMeta(estimated);
    }
    if (job.proposed_lines) {
      setHorizon(job.proposed_lines.horizon_line);
      setCounting(job.proposed_lines.counting_line);
    } else if (estimated) {
      const fallback = defaultCalibrationLines(estimated.width, estimated.height);
      setHorizon(fallback.horizon);
      setCounting(fallback.counting);
    }
    if (job.proposed_preview_url) {
      setPreviewPath(job.proposed_preview_url);
      setPreviewToken(Date.now());
    }

    let cancelled = false;
    if (!job.proposed_preview_url) {
      void (async () => {
        try {
          const preview = await prescanPreview(job.job_id, 0);
          if (cancelled) {
            return;
          }
          setVideoMeta(preview.video_meta);
          setPreviewPath(preview.preview_url);
          setPreviewToken(Date.now());
          if (!job.proposed_lines && preview.proposed_lines) {
            setHorizon(preview.proposed_lines.horizon_line);
            setCounting(preview.proposed_lines.counting_line);
          }
          setMetadata((prev) => mergeOcrMetadata(prev, preview.ocr, false));
        } catch (err) {
          if (!cancelled) {
            setError(err instanceof Error ? err.message : String(err));
          }
        }
      })();
    }

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- load once per job
  }, [job.job_id]);

  async function rescanAtOffset(frameOffset: number) {
    setRescanning(true);
    setError(null);
    try {
      const preview = await prescanPreview(job.job_id, frameOffset);
      setVideoMeta(preview.video_meta);
      setPreviewPath(preview.preview_url);
      setPreviewToken(Date.now());
      setMetadata((prev) => mergeOcrMetadata(prev, preview.ocr, true));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setRescanning(false);
    }
  }

  const meta = videoMeta;
  const metaIssues = validateMetadataFields(metadata);
  const geometryIssues =
    meta && horizon && counting
      ? validateCalibration(horizon, counting, meta.width, meta.height)
      : ["Loading preview…"];
  const allIssues = [...metaIssues, ...geometryIssues];

  const time = metadata.user_start_time.trim();
  const date = metadata.user_start_date.trim();
  const location = metadata.location.trim();
  const timeInvalid = !time || !TIME_PATTERN.test(time);
  const dateInvalid = !date || !DATE_PATTERN.test(date);
  const locationInvalid = !location;

  function fieldLabelClass(invalid: boolean): string {
    return cn("text-sm", invalid && "text-red-600 dark:text-red-400");
  }

  function fieldInputClass(invalid: boolean, mono = false): string {
    return cn(
      "mt-1 w-full rounded border bg-card px-2 py-1 text-sm text-foreground",
      mono && "font-mono",
      invalid ? "border-red-500" : "border-input",
    );
  }

  async function submitConfirm() {
    if (!meta || !horizon || !counting || allIssues.length > 0) {
      return;
    }
    setLoading(true);
    setError(null);
    const confirmed: ConfirmedJobMetadata = {
      user_start_time: metadata.user_start_time.trim(),
      user_start_date: metadata.user_start_date.trim(),
      location: metadata.location.trim(),
    };
    const taskParams = {
      horizon_line: clampLine(horizon, meta.width, meta.height),
      counting_line: clampLine(counting, meta.width, meta.height),
      confidence_threshold: 0.75,
      use_heuristic_truck_split: true,
      render_video: true,
      telemetry_detail: true,
    };

    try {
      if (saveAsProfile && /^[a-z0-9][a-z0-9_-]*$/.test(profileId)) {
        await saveProfile(projectId, {
          profile_id: profileId,
          profile_name: profileId,
          reference_resolution: [meta.width, meta.height],
          horizon_line: taskParams.horizon_line,
          counting_line: taskParams.counting_line,
          source: job.proposed_lines ? "user_edited" : "user_drawn",
        });
      }

      await confirmPrescan(job.job_id, {
        metadata: confirmed,
        task_parameters: taskParams,
      });

      if (applyToOthers) {
        const others = awaitingReviewJobs.filter(
          (item) =>
            item.job_id !== job.job_id &&
            (item.status === "AWAITING_REVIEW" || item.status === "READY"),
        );
        for (const other of others) {
          let lines = taskParams;
          const otherEstimated = estimateVideoMetaFromLines(other.proposed_lines);
          if (
            otherEstimated &&
            (otherEstimated.width !== meta.width || otherEstimated.height !== meta.height)
          ) {
            lines = {
              ...taskParams,
              horizon_line: scaleLine(
                taskParams.horizon_line,
                meta.width,
                meta.height,
                otherEstimated.width,
                otherEstimated.height,
              ),
              counting_line: scaleLine(
                taskParams.counting_line,
                meta.width,
                meta.height,
                otherEstimated.width,
                otherEstimated.height,
              ),
            };
          }
          const otherMeta: ConfirmedJobMetadata = {
            user_start_time:
              other.proposed_metadata?.user_start_time?.trim() ??
              other.confirmed_metadata?.user_start_time ??
              confirmed.user_start_time,
            user_start_date:
              other.proposed_metadata?.user_start_date?.trim() ??
              other.confirmed_metadata?.user_start_date ??
              confirmed.user_start_date,
            location: confirmed.location,
          };
          await confirmPrescan(other.job_id, {
            metadata: otherMeta,
            task_parameters: lines,
          });
        }
      }

      onConfirmed(job.job_id);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  const otherCount = awaitingReviewJobs.filter(
    (item) =>
      item.job_id !== job.job_id &&
      (item.status === "AWAITING_REVIEW" || item.status === "READY"),
  ).length;

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-black/50 p-4">
      <div className="my-4 w-full max-w-5xl rounded-lg bg-card p-5 shadow-xl">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="text-lg font-semibold">Prescan review</h2>
            <p className="mt-1 font-mono text-xs text-muted">
              {job.source_video_path}
            </p>
            <p className="mt-1 text-xs text-muted">
              Edit metadata and lines, then confirm when ready.
            </p>
          </div>
          <Button
            type="button"
            size="sm"
            variant="ghost"
            className={DARK_GHOST_BTN}
            onClick={onClose}
          >
            Close
          </Button>
        </div>

        {error ? <p className="mt-3 text-sm text-red-700">{error}</p> : null}

        {meta && horizon && counting ? (
          <div className="mt-4 grid gap-4 lg:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
            <div className="flex flex-col gap-3">
              {frameLoading ? (
                <p className="text-xs text-muted">Loading frame…</p>
              ) : null}
              <CalibrationCanvas
                width={meta.width}
                height={meta.height}
                horizon={horizon}
                counting={counting}
                previewUrl={previewUrl}
                sourceVideoUrl={sourceVideoUrl(job.job_id)}
                frameOffsetSec={offset}
                onVideoMeta={setVideoMeta}
                onFrameLoading={setFrameLoading}
                onChange={(next) => {
                  setHorizon(next.horizon);
                  setCounting(next.counting);
                }}
              />
              <p className="text-xs text-muted">
                Drag endpoints or lines to fine-tune. Coordinates are in video pixels.
              </p>
              <div className="grid gap-2 rounded border border-border bg-accent p-2 font-mono text-xs">
                <p>
                  <span className="text-red-700">Horizon:</span> {formatLineCoords(horizon)}
                </p>
                <p>
                  <span className="text-green-700">Counting:</span>{" "}
                  {formatLineCoords(counting)}
                </p>
              </div>
              <label className="text-sm">
                Frame offset (sec)
                <input
                  type="range"
                  min={0}
                  max={Math.max(1, Math.floor(meta.duration_sec || 300))}
                  step={1}
                  value={offset}
                  className="mt-1 w-full"
                  onChange={(event) => setOffset(Number(event.target.value))}
                />
                <span className="ml-2 font-mono text-xs">{offset}s</span>
              </label>
              <Button
                type="button"
                size="sm"
                variant="outline"
                className={DARK_OUTLINE_BTN}
                disabled={rescanning || loading}
                onClick={() => void rescanAtOffset(offset)}
              >
                {rescanning ? "Scanning frame…" : `Re-scan at ${offset}s`}
              </Button>
            </div>
            <div className="flex flex-col gap-3">
              <label className={fieldLabelClass(dateInvalid)}>
                Video Start Date (DD-MM-YYYY) *
                <input
                  className={fieldInputClass(dateInvalid, true)}
                  value={metadata.user_start_date}
                  onChange={(event) =>
                    setMetadata((prev) => ({
                      ...prev,
                      user_start_date: event.target.value,
                    }))
                  }
                />
              </label>
              <label className={fieldLabelClass(timeInvalid)}>
                Video Start Time (HH:MM:SS) *
                <input
                  className={fieldInputClass(timeInvalid, true)}
                  value={metadata.user_start_time}
                  onChange={(event) =>
                    setMetadata((prev) => ({
                      ...prev,
                      user_start_time: event.target.value,
                    }))
                  }
                />
              </label>
              <label className={fieldLabelClass(locationInvalid)}>
                Location *
                <input
                  className={fieldInputClass(locationInvalid)}
                  value={metadata.location}
                  onChange={(event) =>
                    setMetadata((prev) => ({ ...prev, location: event.target.value }))
                  }
                />
              </label>
              {otherCount > 0 ? (
                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={applyToOthers}
                    onChange={(event) => setApplyToOthers(event.target.checked)}
                  />
                  Apply lines + location to {otherCount} other awaiting-review job
                  {otherCount === 1 ? "" : "s"} (not time/date)
                </label>
              ) : null}
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={saveAsProfile}
                  onChange={(event) => setSaveAsProfile(event.target.checked)}
                />
                Save as project profile
              </label>
              {saveAsProfile ? (
                <label className="text-sm">
                  profile_id
                  <input
                    className="mt-1 w-full rounded border border-input bg-card px-2 py-1 font-mono text-sm"
                    value={profileId}
                    onChange={(event) => setProfileId(event.target.value)}
                  />
                </label>
              ) : null}
              <div className="mt-auto flex justify-end gap-2">
                <Button
                  type="button"
                  variant="outline"
                  className={DARK_OUTLINE_BTN}
                  onClick={onClose}
                >
                  Cancel
                </Button>
                <Button
                  type="button"
                  disabled={allIssues.length > 0 || loading || rescanning}
                  onClick={() => void submitConfirm()}
                >
                  {loading ? "Submitting…" : "Submit"}
                </Button>
              </div>
            </div>
          </div>
        ) : (
          <p className="mt-4 text-sm text-muted">
            {loading ? "Loading prescan preview…" : "Waiting for preview data…"}
          </p>
        )}
      </div>
    </div>
  );
}
