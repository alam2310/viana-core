"use client";

/**
 * PARKED 2026-08-20 — do not mount from production UI.
 *
 * Live in-progress `_processed.mp4` preview was unstable in Chromium (seek/reload
 * blackouts, FD exhaustion on Range storms, sync-vs-picture mismatches).
 * Decision: hide from the UI; keep this module for a future revisit.
 * See docs/steps/STABILIZATION_BACKLOG.md § S20 / S24.
 *
 * Not imported by the dashboard or job details. Do not remount.
 */

import { useEffect, useRef, useState } from "react";

import { partialVideoUrl } from "@/lib/api-client";
import {
  LIVE_BUFFER_SEC,
  frameToMediaTimeSec,
  mediaTimeSecToFrame,
} from "@/features/monitor/crossing-media-sync";

const MIN_BYTES_BEFORE_PLAY = 256 * 1024;
const PROBE_WHILE_WAITING_MS = 4000;
const PROBE_WHILE_PLAYING_MS = 30000;
/** Reload growing file only when playback reaches EOF of this snapshot. */
const SRC_REFRESH_MIN_MS = 15000;
const ERROR_RETRY_MIN_MS = 8000;
const UI_FRAME_EMIT_MS = 200;

export { LIVE_BUFFER_SEC };

function contentRangeTotal(header: string | null): number | null {
  if (!header) {
    return null;
  }
  const match = /\/(\d+)\s*$/.exec(header);
  if (!match) {
    return null;
  }
  const total = Number(match[1]);
  return Number.isFinite(total) ? total : null;
}

/**
 * Continuous playback of partial ``_processed.mp4``.
 *
 * Do **not** seek on every engine PROGRESS tick — that blanks Chrome on
 * fragmented MP4 (play a few frames → black → repeat).
 *
 * Sync model:
 * - Position once at engine_frame − buffer, then play forward.
 * - ``uiFrame = floor(currentTime × sourceFps)`` is the picture clock.
 * - Crossings gate on that same ``uiFrame`` (see monitor sidebar).
 * - Refresh src only when this snapshot ends (throttled).
 */
export function LiveProcessedVideo({
  jobId,
  isGrowing,
  startFrame,
  sourceFps,
  onUiFrame,
}: {
  jobId: string;
  isGrowing: boolean;
  /** One-shot start position (engine − buffer). Ignored after first paint. */
  startFrame: number | null;
  sourceFps: number | null;
  /** Frame currently shown — drives Live Crossings. */
  onUiFrame?: (frame: number | null) => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [available, setAvailable] = useState(false);
  const [statusText, setStatusText] = useState("Waiting for processed video…");
  const [mediaError, setMediaError] = useState<string | null>(null);
  const [hasPainted, setHasPainted] = useState(false);
  const [cacheBust, setCacheBust] = useState(0);

  const lastSrcRefreshRef = useRef(0);
  const didStartSeekRef = useRef(false);
  const wasGrowingRef = useRef(isGrowing);
  const errorRetriesRef = useRef(0);
  const availableRef = useRef(false);
  const loadedSrcRef = useRef<string | null>(null);
  const startFrameRef = useRef(startFrame);
  const sourceFpsRef = useRef(sourceFps);
  const onUiFrameRef = useRef(onUiFrame);
  const lastEmitRef = useRef(0);
  const lastUiFrameRef = useRef<number | null>(null);
  startFrameRef.current = startFrame;
  sourceFpsRef.current = sourceFps;
  onUiFrameRef.current = onUiFrame;
  availableRef.current = available;

  const probeUrl = partialVideoUrl(jobId);
  const src = partialVideoUrl(jobId, cacheBust);

  const refreshSrc = (minGapMs: number) => {
    const now = Date.now();
    if (now - lastSrcRefreshRef.current < minGapMs) {
      return false;
    }
    lastSrcRefreshRef.current = now;
    // Re-apply seek on the new snapshot (prefer last UI frame for continuity).
    didStartSeekRef.current = false;
    setCacheBust((v) => v + 1);
    return true;
  };

  const emitUiFrame = (video: HTMLVideoElement, force = false) => {
    const cb = onUiFrameRef.current;
    const fps = sourceFpsRef.current;
    if (!cb || fps == null || !(fps > 0)) {
      return;
    }
    const now = Date.now();
    if (!force && now - lastEmitRef.current < UI_FRAME_EMIT_MS) {
      return;
    }
    lastEmitRef.current = now;
    if (!Number.isFinite(video.currentTime)) {
      cb(null);
      return;
    }
    const frame = mediaTimeSecToFrame(video.currentTime, fps);
    lastUiFrameRef.current = frame;
    cb(frame);
  };

  useEffect(() => {
    return () => {
      onUiFrameRef.current?.(null);
    };
  }, [jobId]);

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;

    async function probe() {
      try {
        const response = await fetch(probeUrl, {
          method: "GET",
          headers: { Range: "bytes=0-1" },
          cache: "no-store",
        });
        if (cancelled) {
          return;
        }
        if (response.ok || response.status === 206) {
          const total =
            contentRangeTotal(response.headers.get("Content-Range")) ??
            Number(response.headers.get("Content-Length") ?? 0);
          if (total < MIN_BYTES_BEFORE_PLAY) {
            setAvailable(false);
            setStatusText(
              `Processed video writing… (${Math.max(0, Math.floor(total / 1024))} KB)`,
            );
          } else {
            setAvailable(true);
            setStatusText("");
          }
        } else if (response.status === 404) {
          setAvailable(false);
          setStatusText("Processed video not ready yet…");
        } else {
          setAvailable(false);
          setStatusText(`Video unavailable (${response.status})`);
        }
      } catch {
        if (!cancelled) {
          setAvailable(false);
          setStatusText("Cannot reach processed video stream");
        }
      } finally {
        if (!cancelled) {
          const delay = availableRef.current
            ? PROBE_WHILE_PLAYING_MS
            : PROBE_WHILE_WAITING_MS;
          timer = window.setTimeout(() => void probe(), delay);
        }
      }
    }

    void probe();
    return () => {
      cancelled = true;
      if (timer !== undefined) {
        window.clearTimeout(timer);
      }
    };
  }, [probeUrl, jobId]);

  useEffect(() => {
    if (wasGrowingRef.current && !isGrowing && available) {
      refreshSrc(0);
    }
    wasGrowingRef.current = isGrowing;
  }, [isGrowing, available]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !available) {
      return;
    }
    let cancelled = false;
    let errorRetryTimer: number | undefined;

    const onLoaded = () => {
      if (cancelled) {
        return;
      }
      setMediaError(null);
      errorRetriesRef.current = 0;
      const fps = sourceFpsRef.current;
      const resumeFrame = lastUiFrameRef.current ?? startFrameRef.current;
      if (
        !didStartSeekRef.current &&
        isGrowing &&
        resumeFrame != null &&
        fps != null &&
        fps > 0 &&
        Number.isFinite(video.duration) &&
        video.duration > 0
      ) {
        const target = Math.min(
          frameToMediaTimeSec(resumeFrame, fps),
          Math.max(0, video.duration - 0.05),
        );
        try {
          video.currentTime = Math.max(0, target);
        } catch {
          /* ignore */
        }
        didStartSeekRef.current = true;
      }
      void video.play().catch(() => undefined);
    };

    const onSeeked = () => {
      setHasPainted(true);
      emitUiFrame(video, true);
      void video.play().catch(() => undefined);
    };

    const onPlaying = () => {
      setHasPainted(true);
    };

    const onTimeUpdate = () => {
      emitUiFrame(video);
    };

    const onEnded = () => {
      if (isGrowing) {
        refreshSrc(SRC_REFRESH_MIN_MS);
      }
    };

    const onError = () => {
      const code = video.error?.code;
      if (isGrowing && errorRetriesRef.current < 4) {
        errorRetriesRef.current += 1;
        setMediaError("Retrying preview…");
        errorRetryTimer = window.setTimeout(() => {
          refreshSrc(ERROR_RETRY_MIN_MS);
        }, ERROR_RETRY_MIN_MS);
        return;
      }
      onUiFrameRef.current?.(null);
      if (code === 4) {
        setMediaError("Browser could not decode this processed MP4 stream.");
      } else if (code != null) {
        setMediaError(`Video decode error (code ${code}).`);
      }
    };

    video.addEventListener("loadedmetadata", onLoaded);
    video.addEventListener("seeked", onSeeked);
    video.addEventListener("playing", onPlaying);
    video.addEventListener("timeupdate", onTimeUpdate);
    video.addEventListener("ended", onEnded);
    video.addEventListener("error", onError);

    if (loadedSrcRef.current !== src) {
      loadedSrcRef.current = src;
      lastSrcRefreshRef.current = Date.now();
      video.src = src;
      video.load();
    }

    return () => {
      cancelled = true;
      if (errorRetryTimer !== undefined) {
        window.clearTimeout(errorRetryTimer);
      }
      video.removeEventListener("loadedmetadata", onLoaded);
      video.removeEventListener("seeked", onSeeked);
      video.removeEventListener("playing", onPlaying);
      video.removeEventListener("timeupdate", onTimeUpdate);
      video.removeEventListener("ended", onEnded);
      video.removeEventListener("error", onError);
    };
  }, [src, available, isGrowing]);

  return (
    <div>
      {available ? (
        <div className="space-y-2">
          <div className="relative aspect-video w-full overflow-hidden rounded border border-border bg-black">
            <video
              ref={videoRef}
              className="h-full w-full"
              controls={!isGrowing}
              playsInline
              preload="auto"
            >
              <track kind="captions" />
            </video>
            {!hasPainted ? (
              <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-black/50 text-xs text-white">
                Loading preview…
              </div>
            ) : null}
          </div>
          {isGrowing ? (
            <p className="text-xs text-muted">
              Continuous preview (~{LIVE_BUFFER_SEC}s behind at start). Crossings
              appear when this picture reaches that frame — not when the engine
              emits the event.
            </p>
          ) : null}
          {mediaError ? (
            <p className="text-xs text-muted">{mediaError}</p>
          ) : null}
        </div>
      ) : (
        <div className="flex aspect-video w-full flex-col items-center justify-center gap-2 rounded border border-dashed border-border bg-accent px-4 text-center text-sm text-muted">
          <p>{statusText}</p>
          <p className="text-xs text-muted">
            Waiting for enough processed video before starting the delayed live
            preview.
          </p>
        </div>
      )}
    </div>
  );
}
