/**
 * PARKED 2026-08-20 — helpers for live-preview ↔ crossing frame sync.
 *
 * Used only by the parked `live-processed-video` path. Job details no longer
 * delays crossings; do not wire these filters into Live Crossings until the
 * video preview is un-parked. See docs/steps/STABILIZATION_BACKLOG.md § S20 / S24.
 */
import type { CrossingRow } from "@/features/telemetry/telemetry-formatters";

/** Initial delay behind engine when first positioning the preview (seconds). */
export const LIVE_BUFFER_SEC = 6;

/** Source-timeline fps for frame↔time conversion (never use processing_fps). */
export function sourceFpsFromJob(options: {
  crossingFps?: number;
  videoDurationSec?: number | null;
  totalFrames?: number | null;
}): number | null {
  if (typeof options.crossingFps === "number" && options.crossingFps > 0) {
    return options.crossingFps;
  }
  const duration = options.videoDurationSec;
  const total = options.totalFrames;
  if (
    typeof duration === "number" &&
    duration > 0 &&
    typeof total === "number" &&
    total > 0
  ) {
    return total / duration;
  }
  return null;
}

/** Engine frame minus buffer — used only to *start* the preview, not to scrub. */
export function delayedDisplayFrame(
  engineFrame: number | null | undefined,
  sourceFps: number | null | undefined,
  bufferSec: number = LIVE_BUFFER_SEC,
): number | null {
  if (
    engineFrame == null ||
    sourceFps == null ||
    !(sourceFps > 0) ||
    !Number.isFinite(engineFrame)
  ) {
    return null;
  }
  const bufferFrames = Math.max(1, Math.round(bufferSec * sourceFps));
  return Math.max(0, Math.floor(engineFrame) - bufferFrames);
}

export function frameToMediaTimeSec(frame: number, sourceFps: number): number {
  return frame / sourceFps;
}

export function mediaTimeSecToFrame(
  timeSec: number,
  sourceFps: number,
): number {
  return Math.max(0, Math.floor(timeSec * sourceFps));
}

/**
 * Gate crossings to the frame currently shown in the ``<video>``.
 * Row appears when ``frame_index <= uiFrame`` — same timeline as the picture.
 */
export function filterCrossingsForDisplayFrame(
  rows: CrossingRow[],
  displayFrame: number | null,
  options: { followLive: boolean; sourceFps?: number | null },
): CrossingRow[] {
  if (!options.followLive) {
    return rows;
  }
  if (displayFrame == null || !Number.isFinite(displayFrame)) {
    return [];
  }
  const fps = options.sourceFps;
  return rows.filter((row) => {
    if (typeof row.frameIndex === "number" && Number.isFinite(row.frameIndex)) {
      return row.frameIndex <= displayFrame;
    }
    if (typeof row.videoPtsMs === "number" && fps != null && fps > 0) {
      return row.videoPtsMs / 1000 <= displayFrame / fps;
    }
    return false;
  });
}
