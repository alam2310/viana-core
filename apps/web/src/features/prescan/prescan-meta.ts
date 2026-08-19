import type { JobStatusResponse, ProposedLines, VideoMeta } from "@viana/contracts";

/** Rough video dimensions from proposed line geometry when meta is not on the job record. */
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
  const height = Math.max(...ys) + 1;
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
  return estimateVideoMetaFromLines(
    job.proposed_lines ?? undefined,
  );
}
