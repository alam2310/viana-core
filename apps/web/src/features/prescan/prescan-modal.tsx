"use client";

import { useState } from "react";
import type {
  JobMetadata,
  LineSegment,
  PrescanResponse,
  VideoMeta,
} from "@viana/contracts";

import { CalibrationCanvas } from "@/features/calibration/calibration-canvas";
import { Button } from "@/components/ui/button";
import {
  prescan,
  resolveApiAssetUrl,
  saveProfile,
} from "@/lib/api-client";
import {
  defaultCalibrationLines,
  validateCalibration,
} from "@/lib/geometry";

export interface CalibrationDraft {
  source_video_path: string;
  video_meta: VideoMeta;
  metadata: JobMetadata;
  horizon_line: LineSegment;
  counting_line: LineSegment;
  applyToPending: boolean;
}

export function PrescanModal({
  projectId,
  sourceVideoPath,
  onClose,
  onConfirm,
}: {
  projectId: string;
  sourceVideoPath: string;
  onClose: () => void;
  onConfirm: (draft: CalibrationDraft) => void;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [offset, setOffset] = useState(0);
  const [result, setResult] = useState<PrescanResponse | null>(null);
  const [metadata, setMetadata] = useState<JobMetadata>({});
  const [horizon, setHorizon] = useState<LineSegment | null>(null);
  const [counting, setCounting] = useState<LineSegment | null>(null);
  const [applyToPending, setApplyToPending] = useState(false);
  const [saveAsProfile, setSaveAsProfile] = useState(false);
  const [profileId, setProfileId] = useState("session");

  async function runPrescan(frameOffset = offset) {
    setBusy(true);
    setError(null);
    try {
      const response = await prescan({
        source_video_path: sourceVideoPath,
        project_id: projectId,
        frame_offset_sec: frameOffset,
      });
      setResult(response);
      setMetadata({
        user_start_time: response.ocr.time ?? undefined,
        user_start_date: response.ocr.date ?? undefined,
        location: response.ocr.location ?? undefined,
      });
      const fallback = defaultCalibrationLines(
        response.video_meta.width,
        response.video_meta.height,
      );
      if (response.proposed_lines) {
        setHorizon(response.proposed_lines.horizon_line);
        setCounting(response.proposed_lines.counting_line);
      } else {
        setHorizon(fallback.horizon);
        setCounting(fallback.counting);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  const meta = result?.video_meta;
  const issues =
    meta && horizon && counting
      ? validateCalibration(horizon, counting, meta.width, meta.height)
      : ["Run prescan first"];

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-black/50 p-4">
      <div className="my-8 w-full max-w-3xl rounded-lg bg-white p-5 shadow-xl">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="text-lg font-semibold">Prescan & calibration</h2>
            <p className="mt-1 font-mono text-xs text-neutral-500">{sourceVideoPath}</p>
          </div>
          <Button type="button" size="sm" variant="ghost" onClick={onClose}>
            Close
          </Button>
        </div>

        {!result ? (
          <div className="mt-4">
            <Button type="button" onClick={() => void runPrescan(0)} disabled={busy}>
              {busy ? "Scanning…" : "Run prescan"}
            </Button>
          </div>
        ) : null}

        {error ? <p className="mt-3 text-sm text-red-700">{error}</p> : null}

        {result && meta && horizon && counting ? (
          <div className="mt-4 flex flex-col gap-4">
            <p className="text-xs text-neutral-500">
              {meta.width}×{meta.height} · {meta.fps} fps · {meta.duration_sec}s
              {result.proposed_lines
                ? ` · line confidence ${result.proposed_lines.confidence}`
                : ""}
              {result.ocr.confidence != null
                ? ` · OCR ${result.ocr.confidence}`
                : ""}
            </p>
            <CalibrationCanvas
              width={meta.width}
              height={meta.height}
              horizon={horizon}
              counting={counting}
              previewUrl={resolveApiAssetUrl(result.preview_url)}
              onChange={(next) => {
                setHorizon(next.horizon);
                setCounting(next.counting);
              }}
            />
            <label className="text-sm">
              Frame offset (sec)
              <input
                type="range"
                min={0}
                max={Math.max(0, meta.duration_sec)}
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
              onClick={() => void runPrescan(offset)}
              disabled={busy}
            >
              Re-prescan at offset
            </Button>
            <div className="grid gap-3 sm:grid-cols-3">
              <label className="text-sm">
                Time
                <input
                  className="mt-1 w-full rounded border border-neutral-300 px-2 py-1 font-mono text-sm"
                  value={metadata.user_start_time ?? ""}
                  onChange={(event) =>
                    setMetadata((prev) => ({
                      ...prev,
                      user_start_time: event.target.value,
                    }))
                  }
                />
              </label>
              <label className="text-sm">
                Date
                <input
                  className="mt-1 w-full rounded border border-neutral-300 px-2 py-1 font-mono text-sm"
                  value={metadata.user_start_date ?? ""}
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
                  value={metadata.location ?? ""}
                  onChange={(event) =>
                    setMetadata((prev) => ({
                      ...prev,
                      location: event.target.value,
                    }))
                  }
                />
              </label>
            </div>
            {issues.length > 0 ? (
              <ul className="text-sm text-red-700">
                {issues.map((issue) => (
                  <li key={issue}>{issue}</li>
                ))}
              </ul>
            ) : null}
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={applyToPending}
                onChange={(event) => setApplyToPending(event.target.checked)}
              />
              Apply lines to all pending videos
            </label>
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
            <div className="flex justify-end gap-2">
              <Button type="button" variant="outline" onClick={onClose}>
                Cancel
              </Button>
              <Button
                type="button"
                disabled={issues.length > 0 || busy}
                onClick={() =>
                  void (async () => {
                    if (saveAsProfile && /^[a-z0-9][a-z0-9_-]*$/.test(profileId)) {
                      try {
                        await saveProfile(projectId, {
                          profile_id: profileId,
                          profile_name: profileId,
                          reference_resolution: [meta.width, meta.height],
                          horizon_line: horizon,
                          counting_line: counting,
                          source: result.proposed_lines
                            ? "user_edited"
                            : "user_drawn",
                        });
                      } catch (err) {
                        setError(err instanceof Error ? err.message : String(err));
                        return;
                      }
                    }
                    onConfirm({
                      source_video_path: sourceVideoPath,
                      video_meta: meta,
                      metadata,
                      horizon_line: horizon,
                      counting_line: counting,
                      applyToPending,
                    });
                  })()
                }
              >
                Save calibration
              </Button>
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}
