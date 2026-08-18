import type { LineSegment, Point } from "@viana/contracts";

export const PROJECT_ID_PATTERN = /^[a-z0-9][a-z0-9_-]*$/;

export function clampCoord(value: number, maxExclusive: number): number {
  if (maxExclusive <= 0) {
    return 0;
  }
  return Math.max(0, Math.min(maxExclusive - 1, Math.round(value)));
}

export function clampPoint(
  point: Point,
  width: number,
  height: number,
): Point {
  return [clampCoord(point[0], width), clampCoord(point[1], height)];
}

export function clampLine(
  line: LineSegment,
  width: number,
  height: number,
): LineSegment {
  return {
    start: clampPoint(line.start, width, height),
    end: clampPoint(line.end, width, height),
  };
}

export function isDegenerate(line: LineSegment): boolean {
  return line.start[0] === line.end[0] && line.start[1] === line.end[1];
}

export function pointInFrame(
  point: Point,
  width: number,
  height: number,
): boolean {
  return (
    point[0] >= 0 &&
    point[0] < width &&
    point[1] >= 0 &&
    point[1] < height
  );
}

export function validateLine(
  line: LineSegment | undefined,
  width: number,
  height: number,
  label: string,
): string | null {
  if (!line) {
    return `Missing ${label} line`;
  }
  if (!pointInFrame(line.start, width, height) || !pointInFrame(line.end, width, height)) {
    return `${label} point out of bounds`;
  }
  if (isDegenerate(line)) {
    return `${label} line is degenerate (start == end)`;
  }
  return null;
}

export function validateCalibration(
  horizon: LineSegment | undefined,
  counting: LineSegment | undefined,
  width: number,
  height: number,
): string[] {
  return [
    validateLine(horizon, width, height, "Horizon"),
    validateLine(counting, width, height, "Counting"),
  ].filter((msg): msg is string => msg !== null);
}

export function scaleLine(
  line: LineSegment,
  fromWidth: number,
  fromHeight: number,
  toWidth: number,
  toHeight: number,
): LineSegment {
  const scaleX = toWidth / fromWidth;
  const scaleY = toHeight / fromHeight;
  const mapPoint = (p: Point): Point =>
    clampPoint(
      [Math.round(p[0] * scaleX), Math.round(p[1] * scaleY)],
      toWidth,
      toHeight,
    );
  return { start: mapPoint(line.start), end: mapPoint(line.end) };
}

export function videoStem(path: string): string {
  const base = path.split("/").pop() ?? path;
  return base.replace(/\.[^.]+$/, "");
}

/** In-frame fallback when prescan does not propose lines (v2 never uses off-screen geometry). */
export function defaultCalibrationLines(
  width: number,
  height: number,
): { horizon: LineSegment; counting: LineSegment } {
  const x0 = 0;
  const x1 = Math.max(1, width - 1);
  const horizonLeftY = clampCoord(height * 0.6, height);
  return {
    horizon: { start: [x0, horizonLeftY], end: [x1, 0] },
    counting: { start: [x0, Math.max(0, height - 1)], end: [x1, 0] },
  };
}
