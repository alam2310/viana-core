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
    return () => window.clearInterval(timer);
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
      if (savedTime > 0 && savedTime < video.duration) {
        video.currentTime = savedTime;
      }
      void video.play().catch(() => undefined);
    };
    video.addEventListener("loadedmetadata", onLoaded, { once: true });
    return () => video.removeEventListener("loadedmetadata", onLoaded);
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
        <video
          ref={videoRef}
          key={src}
          className="w-full rounded border border-neutral-300 bg-black"
          controls
          playsInline
          src={src}
        >
          <track kind="captions" />
        </video>
      ) : (
        <div className="flex aspect-video w-full flex-col items-center justify-center gap-2 rounded border border-dashed border-neutral-300 bg-neutral-50 px-4 text-center text-sm text-neutral-500">
          <p>{statusText}</p>
          <p className="text-xs text-neutral-400">
            Live preview may stay blank until processing finishes — the growing MP4
            file is often not playable until the job completes (backend fix tracked).
          </p>
        </div>
      )}
    </div>
  );
}
