"use client";

import { useEffect, useRef, useState } from "react";
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
import { initialVideoMeta } from "@/features/prescan/prescan-meta";
import { Button } from "@/components/ui/button";
import {
  confirmPrescan,
  prescanPreview,
  previewImageUrl,
  saveProfile,
} from "@/lib/api-client";
import {
  clampLine,
  defaultCalibrationLines,
  scaleLine,
  validateCalibration,
} from "@/lib/geometry";
import { validateMetadataFields } from "@/lib/validation";

type Step = "edit" | "summary";

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
  telemetryDetail,
  awaitingReviewJobs,
  onClose,
  onConfirmed,
}: {
  job: JobStatusResponse;
  projectId: string;
  telemetryDetail: boolean;
  awaitingReviewJobs: JobStatusResponse[];
  onClose: () => void;
  onConfirmed: (jobId: string) => void;
}) {
  const [step, setStep] = useState<Step>("edit");
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
  const frameRequestRef = useRef(0);
  const skipInitialFrameFetch = useRef(true);

  const previewUrl = previewPath
    ? previewImageUrl(previewPath, previewToken)
    : null;

  useEffect(() => {
    setLoading(false);
    setError(null);
    skipInitialFrameFetch.current = true;

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

  useEffect(() => {
    if (skipInitialFrameFetch.current) {
      skipInitialFrameFetch.current = false;
      return;
    }
    if (offset === 0 && job.proposed_preview_url) {
      setPreviewPath(job.proposed_preview_url);
      setPreviewToken(Date.now());
      return;
    }
    const requestId = frameRequestRef.current + 1;
    frameRequestRef.current = requestId;
    const timer = window.setTimeout(() => {
      void loadFrameAtOffset(offset, requestId);
    }, 500);
    return () => window.clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- debounced per offset
  }, [offset, job.job_id]);

  async function loadFrameAtOffset(frameOffset: number, requestId: number) {
    setFrameLoading(true);
    setError(null);
    try {
      const preview = await prescanPreview(job.job_id, frameOffset);
      if (frameRequestRef.current !== requestId) {
        return;
      }
      setPreviewPath(preview.preview_url);
      setPreviewToken(Date.now());
      if (preview.video_meta) {
        setVideoMeta(preview.video_meta);
      }
    } catch (err) {
      if (frameRequestRef.current === requestId) {
        setError(err instanceof Error ? err.message : String(err));
      }
    } finally {
      if (frameRequestRef.current === requestId) {
        setFrameLoading(false);
      }
    }
  }

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
      telemetry_detail: telemetryDetail,
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
          const otherPreview = await prescanPreview(other.job_id, 0);
          if (
            otherPreview.video_meta.width !== meta.width ||
            otherPreview.video_meta.height !== meta.height
          ) {
            lines = {
              ...taskParams,
              horizon_line: scaleLine(
                taskParams.horizon_line,
                meta.width,
                meta.height,
                otherPreview.video_meta.width,
                otherPreview.video_meta.height,
              ),
              counting_line: scaleLine(
                taskParams.counting_line,
                meta.width,
                meta.height,
                otherPreview.video_meta.width,
                otherPreview.video_meta.height,
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
      <div className="my-4 w-full max-w-5xl rounded-lg bg-white p-5 shadow-xl">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="text-lg font-semibold">Prescan review</h2>
            <p className="mt-1 font-mono text-xs text-neutral-500">
              {job.source_video_path}
            </p>
            <p className="mt-1 text-xs text-neutral-500">
              Step {step === "edit" ? "1–2" : "3"}:{" "}
              {step === "edit" ? "Edit" : "Confirm summary"}
            </p>
          </div>
          <Button type="button" size="sm" variant="ghost" onClick={onClose}>
            Close
          </Button>
        </div>

        {error ? <p className="mt-3 text-sm text-red-700">{error}</p> : null}

        {step === "summary" ? (
          <div className="mt-4 space-y-3 text-sm">
            <h3 className="font-medium">Review summary</h3>
            <dl className="grid gap-2 sm:grid-cols-2">
              <div>
                <dt className="text-neutral-500">Time</dt>
                <dd className="font-mono">{metadata.user_start_time}</dd>
              </div>
              <div>
                <dt className="text-neutral-500">Date</dt>
                <dd className="font-mono">{metadata.user_start_date}</dd>
              </div>
              <div className="sm:col-span-2">
                <dt className="text-neutral-500">Location</dt>
                <dd>{metadata.location}</dd>
              </div>
              <div className="sm:col-span-2">
                <dt className="text-neutral-500">Horizon line</dt>
                <dd className="font-mono text-xs">
                  {horizon ? formatLineCoords(horizon) : "—"}
                </dd>
              </div>
              <div className="sm:col-span-2">
                <dt className="text-neutral-500">Counting line</dt>
                <dd className="font-mono text-xs">
                  {counting ? formatLineCoords(counting) : "—"}
                </dd>
              </div>
            </dl>
            <div className="flex justify-end gap-2 pt-2">
              <Button type="button" variant="outline" onClick={() => setStep("edit")}>
                Back
              </Button>
              <Button type="button" disabled={loading} onClick={() => void submitConfirm()}>
                {loading ? "Submitting…" : "Confirm → READY"}
              </Button>
            </div>
          </div>
        ) : meta && horizon && counting ? (
          <div className="mt-4 grid gap-4 lg:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
            <div className="flex flex-col gap-3">
              <p className="text-xs text-neutral-500">
                {meta.width}×{meta.height} · {meta.fps} fps · {meta.duration_sec}s
                {frameLoading ? " · loading frame…" : ""}
              </p>
              <CalibrationCanvas
                width={meta.width}
                height={meta.height}
                horizon={horizon}
                counting={counting}
                previewUrl={previewUrl}
                onChange={(next) => {
                  setHorizon(next.horizon);
                  setCounting(next.counting);
                }}
              />
              <p className="text-xs text-neutral-500">
                Drag endpoints or lines to fine-tune. Coordinates are in video pixels.
              </p>
              <div className="grid gap-2 rounded border border-neutral-200 bg-neutral-50 p-2 font-mono text-xs">
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
                disabled={rescanning || loading}
                onClick={() => void rescanAtOffset(offset)}
              >
                {rescanning ? "Scanning frame…" : `Re-scan OCR at ${offset}s`}
              </Button>
            </div>
            <div className="flex flex-col gap-3">
              <label className="text-sm">
                Time (HH:MM:SS)
                <input
                  className="mt-1 w-full rounded border border-neutral-300 px-2 py-1 font-mono text-sm"
                  value={metadata.user_start_time}
                  onChange={(event) =>
                    setMetadata((prev) => ({
                      ...prev,
                      user_start_time: event.target.value,
                    }))
                  }
                />
              </label>
              <label className="text-sm">
                Date (DD-MM-YYYY)
                <input
                  className="mt-1 w-full rounded border border-neutral-300 px-2 py-1 font-mono text-sm"
                  value={metadata.user_start_date}
                  onChange={(event) =>
                    setMetadata((prev) => ({
                      ...prev,
                      user_start_date: event.target.value,
                    }))
                  }
                />
              </label>
              <label className="text-sm">
                Location
                <input
                  className="mt-1 w-full rounded border border-neutral-300 px-2 py-1 text-sm"
                  value={metadata.location}
                  onChange={(event) =>
                    setMetadata((prev) => ({ ...prev, location: event.target.value }))
                  }
                />
              </label>
              {allIssues.length > 0 ? (
                <ul className="text-sm text-red-700">
                  {allIssues.map((issue) => (
                    <li key={issue}>{issue}</li>
                  ))}
                </ul>
              ) : null}
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
                    className="mt-1 w-full rounded border border-neutral-300 px-2 py-1 font-mono text-sm"
                    value={profileId}
                    onChange={(event) => setProfileId(event.target.value)}
                  />
                </label>
              ) : null}
              <div className="mt-auto flex justify-end gap-2">
                <Button type="button" variant="outline" onClick={onClose}>
                  Cancel
                </Button>
                <Button
                  type="button"
                  disabled={allIssues.length > 0 || loading || rescanning}
                  onClick={() => setStep("summary")}
                >
                  Next: summary
                </Button>
              </div>
            </div>
          </div>
        ) : (
          <p className="mt-4 text-sm text-neutral-500">
            {loading ? "Loading prescan preview…" : "Waiting for preview data…"}
          </p>
        )}
      </div>
    </div>
  );
}
