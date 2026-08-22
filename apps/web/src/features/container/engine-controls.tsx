"use client";

import type { ReactNode } from "react";
import { useCallback, useEffect, useState } from "react";

import {
  IconPlay,
  IconRestart,
  IconStop,
  RoundIconButton,
} from "@/components/ui/icon-button";
import type { ContainerStatus } from "@/lib/container-types";

function StatusDot({ running }: { running: boolean }) {
  return (
    <span
      className={`inline-block h-3.5 w-3.5 shrink-0 rounded-full ring-2 ring-white ${
        running ? "bg-emerald-500" : "bg-red-500"
      }`}
      title={running ? "Analytics engine running" : "Analytics engine stopped"}
      aria-hidden
    />
  );
}

export function EngineControls({
  onStatusChange,
  label,
}: {
  onStatusChange?: (status: ContainerStatus) => void;
  label?: ReactNode;
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
    <div className="flex items-center gap-3 rounded-lg border border-border bg-card px-3 py-2">
      {label ? (
        <span className="shrink-0 text-sm font-medium">{label}</span>
      ) : null}
      <StatusDot running={running} />
      <div className="flex items-center gap-1">
        <RoundIconButton
          label={running ? "Restart analytics engine" : "Start analytics engine"}
          variant="neutral"
          disabled={busy}
          onClick={() => void onStartOrRestart()}
        >
          {running ? <IconRestart /> : <IconPlay />}
        </RoundIconButton>
        <RoundIconButton
          label="Stop analytics engine"
          variant="neutral"
          disabled={busy || !running}
          onClick={() => void onStop()}
        >
          <IconStop />
        </RoundIconButton>
      </div>
    </div>
  );
}
