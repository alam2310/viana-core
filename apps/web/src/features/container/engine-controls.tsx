"use client";

import type { ReactNode } from "react";
import { useCallback, useEffect, useState } from "react";

import type { ContainerStatus } from "@/lib/container-types";

function StatusDot({ running }: { running: boolean }) {
  return (
    <span
      className={`inline-block h-2.5 w-2.5 shrink-0 rounded-full ${
        running ? "bg-emerald-500" : "bg-amber-500"
      }`}
      title={running ? "Analytics engine running" : "Analytics engine stopped"}
      aria-hidden
    />
  );
}

function IconButton({
  label,
  onClick,
  disabled,
  children,
}: {
  label: string;
  onClick: () => void;
  disabled?: boolean;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      title={label}
      aria-label={label}
      disabled={disabled}
      className="inline-flex h-8 w-8 items-center justify-center rounded border border-neutral-200 text-neutral-700 hover:bg-neutral-50 disabled:opacity-40"
      onClick={onClick}
    >
      {children}
    </button>
  );
}

export function EngineControls({
  onStatusChange,
}: {
  onStatusChange?: (status: ContainerStatus) => void;
}) {
  const [status, setStatus] = useState<ContainerStatus | null>(null);
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    const response = await fetch("/api/container/status");
    const data = (await response.json()) as ContainerStatus;
    setStatus(data);
    onStatusChange?.(data);
  }, [onStatusChange]);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), 8000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  async function onStartOrRestart() {
    setBusy(true);
    try {
      if (status?.running) {
        await fetch("/api/container/stop", { method: "POST" });
        await new Promise((r) => window.setTimeout(r, 800));
      }
      await fetch("/api/container/start", { method: "POST" });
      await refresh();
    } finally {
      setBusy(false);
    }
  }

  async function onStop() {
    setBusy(true);
    try {
      await fetch("/api/container/stop", { method: "POST" });
      await refresh();
    } finally {
      setBusy(false);
    }
  }

  const running = status?.running === true;

  return (
    <div className="flex items-center gap-2">
      <StatusDot running={running} />
      <IconButton
        label={running ? "Restart analytics engine" : "Start analytics engine"}
        disabled={busy}
        onClick={() => void onStartOrRestart()}
      >
        {running ? (
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden>
            <path d="M17.65 6.35A7.958 7.958 0 0 0 12 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08a5.99 5.99 0 0 1-5.65 4c-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.67 4.22 1.78L13 11h7V4l-2.35 2.35z" />
          </svg>
        ) : (
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden>
            <path d="M8 5v14l11-7z" />
          </svg>
        )}
      </IconButton>
      <IconButton
        label="Stop analytics engine"
        disabled={busy || !running}
        onClick={() => void onStop()}
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden>
          <rect x="6" y="6" width="12" height="12" rx="1" />
        </svg>
      </IconButton>
    </div>
  );
}
