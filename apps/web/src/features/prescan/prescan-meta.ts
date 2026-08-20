import type { JobStatusResponse, ProposedLines, VideoMeta } from "@viana/contracts";

/**
 * Best-effort size from proposed line endpoints when authoritative video_meta
 * is not on the job yet.
 *
 * Width from max X is usually reliable (lines span the frame). Height from max Y
 * is not — counting lines often sit mid-frame — so pad up to a 16:9 guess from
 * width. Prefer JPEG/video intrinsic size or API video_meta whenever available.
 */
export function estimateVideoMetaFromLines(
  lines: ProposedLines | undefined,
): VideoMeta | null {
  if (!lines) {
    return null;
  }
  const xs = [
    lines.horizon_line.start[0],
    lines.horizon_line.end[0],
    lines.counting_line.start[0],
    lines.counting_line.end[0],
  ];
  const ys = [
    lines.horizon_line.start[1],
    lines.horizon_line.end[1],
    lines.counting_line.start[1],
    lines.counting_line.end[1],
  ];
  const width = Math.max(...xs) + 1;
  const heightFromLines = Math.max(...ys) + 1;
  const heightFromAspect = Math.round((width * 9) / 16);
  const height = Math.max(heightFromLines, heightFromAspect);
  if (width < 320 || height < 240) {
    return null;
  }
  return {
    width,
    height,
    fps: 25,
    duration_sec: 0,
    frame_count: 0,
  };
}

export function initialVideoMeta(job: JobStatusResponse): VideoMeta | null {
  const estimated = estimateVideoMetaFromLines(job.proposed_lines ?? undefined);
  if (!estimated) {
    return null;
  }
  const duration_sec = job.video_duration_sec ?? estimated.duration_sec;
  return {
    ...estimated,
    duration_sec,
    frame_count: Math.round(duration_sec * estimated.fps),
  };
}

/** Merge canvas/API meta without wiping known duration/fps with zeros. */
export function mergeVideoMeta(
  prev: VideoMeta | null,
  incoming: VideoMeta,
  fallbackDurationSec?: number | null,
): VideoMeta {
  const fps = incoming.fps > 0 ? incoming.fps : (prev?.fps ?? 25);
  const duration_sec =
    incoming.duration_sec > 0
      ? incoming.duration_sec
      : (prev?.duration_sec ?? fallbackDurationSec ?? 0);
  const width = incoming.width > 0 ? incoming.width : (prev?.width ?? 0);
  const height = incoming.height > 0 ? incoming.height : (prev?.height ?? 0);
  const frame_count =
    incoming.frame_count > 0
      ? incoming.frame_count
      : Math.round(duration_sec * fps);
  return { width, height, fps, duration_sec, frame_count };
}
