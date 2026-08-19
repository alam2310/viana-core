"use client";

import { useEffect, useRef, useState } from "react";
import type { LineSegment, Point, VideoMeta } from "@viana/contracts";

import { clampLine, clampPoint } from "@/lib/geometry";
import { cn } from "@/lib/utils";

type Handle = "horizon-start" | "horizon-end" | "counting-start" | "counting-end";

type DragState =
  | { kind: "handle"; handle: Handle }
  | {
      kind: "translate";
      line: "horizon" | "counting";
      origin: Point;
      snapshot: LineSegment;
    };

const HANDLE_RADIUS = 14;

function dist(a: Point, b: Point): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

function toDisplay(point: Point, scale: number): Point {
  return [point[0] * scale, point[1] * scale];
}

function distToSegment(p: Point, a: Point, b: Point): number {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  if (dx === 0 && dy === 0) {
    return dist(p, a);
  }
  const t = Math.max(
    0,
    Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy)),
  );
  return dist(p, [a[0] + t * dx, a[1] + t * dy]);
}

function hitHandle(
  displayPoint: Point,
  horizon: LineSegment,
  counting: LineSegment,
  scale: number,
): Handle | null {
  const candidates: Array<[Handle, Point]> = [
    ["horizon-start", toDisplay(horizon.start, scale)],
    ["horizon-end", toDisplay(horizon.end, scale)],
    ["counting-start", toDisplay(counting.start, scale)],
    ["counting-end", toDisplay(counting.end, scale)],
  ];
  let best: Handle | null = null;
  let bestD = HANDLE_RADIUS;
  for (const [id, pt] of candidates) {
    const d = dist(displayPoint, pt);
    if (d <= bestD) {
      bestD = d;
      best = id;
    }
  }
  return best;
}

function hitLineBody(
  displayPoint: Point,
  line: LineSegment,
  scale: number,
): boolean {
  const start = toDisplay(line.start, scale);
  const end = toDisplay(line.end, scale);
  return distToSegment(displayPoint, start, end) <= HANDLE_RADIUS;
}

function setHandle(
  handle: Handle,
  point: Point,
  horizon: LineSegment,
  counting: LineSegment,
): { horizon: LineSegment; counting: LineSegment } {
  if (handle === "horizon-start") {
    return { horizon: { ...horizon, start: point }, counting };
  }
  if (handle === "horizon-end") {
    return { horizon: { ...horizon, end: point }, counting };
  }
  if (handle === "counting-start") {
    return { horizon, counting: { ...counting, start: point } };
  }
  return { horizon, counting: { ...counting, end: point } };
}

function translateLine(line: LineSegment, delta: Point): LineSegment {
  return {
    start: [line.start[0] + delta[0], line.start[1] + delta[1]],
    end: [line.end[0] + delta[0], line.end[1] + delta[1]],
  };
}

export function formatLineCoords(line: LineSegment): string {
  return `(${line.start[0]}, ${line.start[1]}) → (${line.end[0]}, ${line.end[1]})`;
}

export function CalibrationCanvas({
  width,
  height,
  horizon,
  counting,
  onChange,
  previewUrl,
  sourceVideoUrl,
  frameOffsetSec = 0,
  onVideoMeta,
  onFrameLoading,
  className,
}: {
  width: number;
  height: number;
  horizon: LineSegment;
  counting: LineSegment;
  onChange: (next: { horizon: LineSegment; counting: LineSegment }) => void;
  previewUrl?: string | null;
  sourceVideoUrl?: string | null;
  frameOffsetSec?: number;
  onVideoMeta?: (meta: VideoMeta) => void;
  onFrameLoading?: (loading: boolean) => void;
  className?: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [drag, setDrag] = useState<DragState | null>(null);
  const [preview, setPreview] = useState<HTMLImageElement | null>(null);
  const [videoReady, setVideoReady] = useState(false);
  const [videoFrameTick, setVideoFrameTick] = useState(0);
  const displayWidth = 720;
  const scale = displayWidth / width;
  const displayHeight = Math.round(height * scale);

  useEffect(() => {
    if (!previewUrl) {
      setPreview(null);
      return;
    }
    let cancelled = false;
    let objectUrl: string | undefined;

    void (async () => {
      try {
        const response = await fetch(previewUrl, { cache: "no-store" });
        if (!response.ok) {
          throw new Error(`preview ${response.status}`);
        }
        const blob = await response.blob();
        if (cancelled) {
          return;
        }
        objectUrl = URL.createObjectURL(blob);
        const image = new Image();
        image.onload = () => {
          if (!cancelled) {
            setPreview(image);
          }
        };
        image.onerror = () => {
          if (!cancelled) {
            setPreview(null);
          }
        };
        image.src = objectUrl;
      } catch {
        if (!cancelled) {
          setPreview(null);
        }
      }
    })();

    return () => {
      cancelled = true;
      if (objectUrl) {
        URL.revokeObjectURL(objectUrl);
      }
    };
  }, [previewUrl]);

  useEffect(() => {
    if (!sourceVideoUrl) {
      setVideoReady(false);
      return;
    }
    const video = videoRef.current;
    if (!video) {
      return;
    }

    let cancelled = false;
    setVideoReady(false);
    onFrameLoading?.(true);

    const onLoadedMetadata = () => {
      if (cancelled) {
        return;
      }
      const fps = 25;
      const durationSec = video.duration;
      onVideoMeta?.({
        width: video.videoWidth,
        height: video.videoHeight,
        fps,
        duration_sec: durationSec,
        frame_count: Math.round(durationSec * fps),
      });
      video.currentTime = frameOffsetSec;
    };

    const onSeeked = () => {
      if (cancelled) {
        return;
      }
      setVideoReady(true);
      setVideoFrameTick((tick) => tick + 1);
      onFrameLoading?.(false);
    };

    const onError = () => {
      if (!cancelled) {
        setVideoReady(false);
        onFrameLoading?.(false);
      }
    };

    video.addEventListener("loadedmetadata", onLoadedMetadata);
    video.addEventListener("seeked", onSeeked);
    video.addEventListener("error", onError);
    video.src = sourceVideoUrl;
    video.load();

    return () => {
      cancelled = true;
      video.removeEventListener("loadedmetadata", onLoadedMetadata);
      video.removeEventListener("seeked", onSeeked);
      video.removeEventListener("error", onError);
      video.removeAttribute("src");
      video.load();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- bind once per source URL
  }, [sourceVideoUrl]);

  useEffect(() => {
    if (!sourceVideoUrl || !videoReady) {
      return;
    }
    const video = videoRef.current;
    if (!video) {
      return;
    }
    if (Math.abs(video.currentTime - frameOffsetSec) < 0.05) {
      return;
    }
    onFrameLoading?.(true);
    video.currentTime = frameOffsetSec;
  }, [frameOffsetSec, sourceVideoUrl, videoReady, onFrameLoading]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }
    ctx.fillStyle = "#111827";
    ctx.fillRect(0, 0, displayWidth, displayHeight);
    const video = videoRef.current;
    const useVideo =
      sourceVideoUrl &&
      video &&
      videoReady &&
      video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA;
    if (useVideo) {
      ctx.drawImage(video, 0, 0, displayWidth, displayHeight);
    } else if (preview) {
      ctx.drawImage(preview, 0, 0, displayWidth, displayHeight);
    } else {
      ctx.strokeStyle = "#374151";
      ctx.lineWidth = 1;
      for (let x = 0; x < displayWidth; x += 80) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, displayHeight);
        ctx.stroke();
      }
      for (let y = 0; y < displayHeight; y += 80) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(displayWidth, y);
        ctx.stroke();
      }
    }

    const drawLine = (line: LineSegment, color: string, label: string) => {
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(line.start[0] * scale, line.start[1] * scale);
      ctx.lineTo(line.end[0] * scale, line.end[1] * scale);
      ctx.stroke();
      for (const pt of [line.start, line.end]) {
        ctx.beginPath();
        ctx.arc(pt[0] * scale, pt[1] * scale, 8, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
      }
      ctx.font = "12px sans-serif";
      ctx.fillStyle = color;
      ctx.fillText(label, line.start[0] * scale + 10, line.start[1] * scale - 8);
    };

    drawLine(horizon, "#dc2626", "Horizon");
    drawLine(counting, "#16a34a", "Counting");
  }, [counting, displayHeight, horizon, preview, scale, sourceVideoUrl, videoFrameTick, videoReady]);

  function eventDisplayPoint(event: React.MouseEvent<HTMLCanvasElement>): Point {
    const canvas = canvasRef.current;
    if (!canvas) {
      return [0, 0];
    }
    const rect = canvas.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * displayWidth;
    const y = ((event.clientY - rect.top) / rect.height) * displayHeight;
    return [x, y];
  }

  function eventVideoPoint(event: React.MouseEvent<HTMLCanvasElement>): Point {
    const display = eventDisplayPoint(event);
    return [display[0] / scale, display[1] / scale];
  }

  function applyChange(next: { horizon: LineSegment; counting: LineSegment }) {
    onChange({
      horizon: clampLine(next.horizon, width, height),
      counting: clampLine(next.counting, width, height),
    });
  }

  function onPointerDown(event: React.MouseEvent<HTMLCanvasElement>) {
    const display = eventDisplayPoint(event);
    const handle = hitHandle(display, horizon, counting, scale);
    if (handle) {
      setDrag({ kind: "handle", handle });
      return;
    }
    if (hitLineBody(display, horizon, scale)) {
      setDrag({
        kind: "translate",
        line: "horizon",
        origin: eventVideoPoint(event),
        snapshot: horizon,
      });
      return;
    }
    if (hitLineBody(display, counting, scale)) {
      setDrag({
        kind: "translate",
        line: "counting",
        origin: eventVideoPoint(event),
        snapshot: counting,
      });
    }
  }

  function onPointerMove(event: React.MouseEvent<HTMLCanvasElement>) {
    if (!drag) {
      return;
    }
    const videoPoint = eventVideoPoint(event);
    if (drag.kind === "handle") {
      const next = setHandle(drag.handle, videoPoint, horizon, counting);
      applyChange(next);
      return;
    }
    const delta: Point = [
      videoPoint[0] - drag.origin[0],
      videoPoint[1] - drag.origin[1],
    ];
    const moved = translateLine(drag.snapshot, delta);
    const clamped: LineSegment = {
      start: clampPoint(moved.start, width, height),
      end: clampPoint(moved.end, width, height),
    };
    if (drag.line === "horizon") {
      applyChange({ horizon: clamped, counting });
    } else {
      applyChange({ horizon, counting: clamped });
    }
  }

  return (
    <>
      {sourceVideoUrl ? (
        <video ref={videoRef} className="hidden" preload="auto" muted playsInline>
          <track kind="captions" />
        </video>
      ) : null}
      <canvas
        ref={canvasRef}
        width={displayWidth}
        height={displayHeight}
        className={cn("w-full cursor-crosshair rounded border border-neutral-300", className)}
        onMouseDown={onPointerDown}
        onMouseMove={onPointerMove}
        onMouseUp={() => setDrag(null)}
        onMouseLeave={() => setDrag(null)}
      />
    </>
  );
}
