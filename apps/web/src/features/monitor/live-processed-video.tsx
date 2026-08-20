"use client";

import { useEffect, useRef, useState } from "react";

import { partialVideoUrl } from "@/lib/api-client";

/** Stream growing `_processed.mp4` via same-origin proxy; refresh as frames are written. */
export function LiveProcessedVideo({
  jobId,
  refreshKey,
}: {
  jobId: string;
  refreshKey: number;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [available, setAvailable] = useState(false);
  const [statusText, setStatusText] = useState("Waiting for processed video…");
  const [mediaError, setMediaError] = useState<string | null>(null);
  const [cacheBust, setCacheBust] = useState(0);
  const lastTimeRef = useRef(0);

  const src = partialVideoUrl(jobId, cacheBust);

  useEffect(() => {
    const timer = window.setInterval(() => {
      setCacheBust((value) => value + 1);
    }, 4000);
    return () => window.clearInterval(timer);
  }, [jobId]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      setCacheBust(refreshKey);
    }, 2000);
    return () => window.clearTimeout(timer);
  }, [refreshKey]);

  useEffect(() => {
    let cancelled = false;

    async function probe() {
      try {
        const response = await fetch(src, {
          method: "GET",
          headers: { Range: "bytes=0-1" },
        });
        if (cancelled) {
          return;
        }
        if (response.ok || response.status === 206) {
          setAvailable(true);
          setStatusText("");
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
      }
    }

    setAvailable(false);
    setMediaError(null);
    setStatusText("Waiting for processed video…");
    void probe();
    const timer = window.setInterval(() => void probe(), 3000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [src, jobId]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !available) {
      return;
    }
    const savedTime = lastTimeRef.current;
    video.load();
    const onLoaded = () => {
      setMediaError(null);
      if (savedTime > 0 && Number.isFinite(video.duration) && savedTime < video.duration) {
        video.currentTime = savedTime;
      }
      void video.play().catch(() => undefined);
    };
    const onError = () => {
      const code = video.error?.code;
      // MEDIA_ERR_SRC_NOT_SUPPORTED = 4 — typically HEVC/hev1 in Chromium.
      if (code === 4) {
        setMediaError(
          "Browser cannot decode this processed MP4 (needs H.264 / avc1). Native players may still play HEVC.",
        );
      } else {
        setMediaError(`Video decode error (code ${code ?? "?"}).`);
      }
    };
    video.addEventListener("loadedmetadata", onLoaded, { once: true });
    video.addEventListener("error", onError);
    return () => {
      video.removeEventListener("loadedmetadata", onLoaded);
      video.removeEventListener("error", onError);
    };
  }, [src, available]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) {
      return;
    }
    const onTimeUpdate = () => {
      lastTimeRef.current = video.currentTime;
    };
    video.addEventListener("timeupdate", onTimeUpdate);
    return () => video.removeEventListener("timeupdate", onTimeUpdate);
  }, []);

  return (
    <div>
      {available ? (
        <div className="space-y-2">
          <video
            ref={videoRef}
            key={src}
            className="w-full rounded border border-border bg-black"
            controls
            playsInline
            preload="metadata"
            src={src}
          >
            <track kind="captions" />
          </video>
          {mediaError ? (
            <p className="text-xs text-muted">{mediaError}</p>
          ) : null}
        </div>
      ) : (
        <div className="flex aspect-video w-full flex-col items-center justify-center gap-2 rounded border border-dashed border-border bg-accent px-4 text-center text-sm text-muted">
          <p>{statusText}</p>
          <p className="text-xs text-muted">
            Live preview appears once the growing processed MP4 is available over the
            same-origin proxy (Range streaming).
          </p>
        </div>
      )}
    </div>
  );
}
