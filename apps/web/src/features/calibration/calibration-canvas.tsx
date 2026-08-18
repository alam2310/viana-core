"use client";

import { useEffect, useRef, useState } from "react";
import type { LineSegment, Point } from "@viana/contracts";

import { clampLine } from "@/lib/geometry";
import { cn } from "@/lib/utils";

type Handle = "horizon-start" | "horizon-end" | "counting-start" | "counting-end";

const HANDLE_R = 18;

function dist(a: Point, b: Point): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

function hitHandle(
  p: Point,
  horizon: LineSegment,
  counting: LineSegment,
): Handle | null {
  const candidates: Array<[Handle, Point]> = [
    ["horizon-start", horizon.start],
    ["horizon-end", horizon.end],
    ["counting-start", counting.start],
    ["counting-end", counting.end],
  ];
  let best: Handle | null = null;
  let bestD = HANDLE_R;
  for (const [id, pt] of candidates) {
    const d = dist(p, pt);
    if (d <= bestD) {
      bestD = d;
      best = id;
    }
  }
  return best;
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

export function CalibrationCanvas({
  width,
  height,
  horizon,
  counting,
  onChange,
  previewUrl,
  className,
}: {
  width: number;
  height: number;
  horizon: LineSegment;
  counting: LineSegment;
  onChange: (next: { horizon: LineSegment; counting: LineSegment }) => void;
  previewUrl?: string | null;
  className?: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [drag, setDrag] = useState<Handle | null>(null);
  const [preview, setPreview] = useState<HTMLImageElement | null>(null);
  const displayWidth = 720;
  const scale = displayWidth / width;
  const displayHeight = Math.round(height * scale);

  useEffect(() => {
    if (!previewUrl) {
      setPreview(null);
      return;
    }
    const image = new Image();
    image.crossOrigin = "anonymous";
    image.onload = () => setPreview(image);
    image.onerror = () => setPreview(null);
    image.src = previewUrl;
  }, [previewUrl]);

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
    if (preview) {
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
        ctx.arc(pt[0] * scale, pt[1] * scale, 7, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.font = "12px sans-serif";
      ctx.fillText(label, line.start[0] * scale + 10, line.start[1] * scale - 8);
    };

    drawLine(horizon, "#dc2626", "Horizon");
    drawLine(counting, "#16a34a", "Counting");
  }, [counting, displayHeight, horizon, preview, scale]);

  function eventPoint(event: React.MouseEvent<HTMLCanvasElement>): Point {
    const canvas = canvasRef.current;
    if (!canvas) {
      return [0, 0];
    }
    const rect = canvas.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * width;
    const y = ((event.clientY - rect.top) / rect.height) * height;
    return [x, y];
  }

  return (
    <canvas
      ref={canvasRef}
      width={displayWidth}
      height={displayHeight}
      className={cn("w-full cursor-crosshair rounded border border-neutral-300", className)}
      onMouseDown={(event) => {
        const p = eventPoint(event);
        setDrag(hitHandle(p, horizon, counting));
      }}
      onMouseMove={(event) => {
        if (!drag) {
          return;
        }
        const p = eventPoint(event);
        const next = setHandle(drag, p, horizon, counting);
        onChange({
          horizon: clampLine(next.horizon, width, height),
          counting: clampLine(next.counting, width, height),
        });
      }}
      onMouseUp={() => setDrag(null)}
      onMouseLeave={() => setDrag(null)}
    />
  );
}
