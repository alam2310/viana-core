"use client";

import { useCallback, useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import type { ContainerStatus } from "@/lib/container-types";

const BENIGN_STOP_ERRORS = new Set([
  "container_not_found",
  "docker_not_installed",
  "docker_inspect_failed",
]);

function containerHint(status: ContainerStatus | null): string | null {
  if (!status || status.running) {
    return null;
  }
  if (status.error === "docker_not_installed") {
    return "Docker is not available on this host. Install Docker to run the ViAna orchestrator.";
  }
  return "ViAna container is not running. Click Start below to launch the orchestrator.";
}

export function ContainerPanel({
  compact = false,
  onStatusChange,
}: {
  compact?: boolean;
  onStatusChange?: (status: ContainerStatus) => void;
}) {
  const [status, setStatus] = useState<ContainerStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    const response = await fetch("/api/container/status");
    const data = (await response.json()) as ContainerStatus;
    setStatus(data);
    onStatusChange?.(data);
  }, [onStatusChange]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  async function onStart() {
    setBusy(true);
    setError(null);
    try {
      const response = await fetch("/api/container/start", { method: "POST" });
      const data = (await response.json()) as ContainerStatus;
      setStatus(data);
      if (!data.running) {
        setError(data.error ?? "Container did not start");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  const hint = containerHint(status);
  const actionError =
    error ??
    (status?.error && !BENIGN_STOP_ERRORS.has(status.error) ? status.error : null);

  if (compact) {
    return (
      <div className="text-xs">
        <p className="text-sm font-medium">
          <span
            className={status?.running ? "text-emerald-700" : "text-amber-700"}
          >
            {status?.running ? "Running" : "Stopped"}
          </span>
        </p>
        {hint ? (
          <p className="mt-1 text-amber-800">{hint}</p>
        ) : null}
        <div className="mt-1 flex gap-1">
          <Button type="button" size="sm" variant="outline" onClick={() => void refresh()}>
            Refresh
          </Button>
          <Button type="button" size="sm" onClick={() => void onStart()} disabled={busy}>
            {busy ? "…" : "Start"}
          </Button>
        </div>
        {actionError ? <p className="mt-1 text-red-700">{actionError}</p> : null}
      </div>
    );
  }

  return (
    <section className="rounded-lg border border-border bg-card p-4">
      <h2 className="text-sm font-semibold tracking-wide text-muted uppercase">
        Container
      </h2>
      <p className="mt-2 font-mono text-sm">
        Analytics Engine —{" "}
        <span className={status?.running ? "text-emerald-600 dark:text-emerald-400" : "text-red-600 dark:text-red-400"}>
          {status?.running ? "running" : "stopped"}
        </span>
      </p>
      {hint ? <p className="mt-2 text-sm text-amber-800 dark:text-amber-200">{hint}</p> : null}
      <p className="mt-1 text-xs text-muted">
        Config: {status?.config_found ? "found" : "using example defaults"} (
        {status?.config_path})
      </p>
      {actionError ? (
        <p className="mt-2 text-sm text-red-700">{actionError}</p>
      ) : null}
      <div className="mt-3 flex gap-2">
        <Button type="button" size="sm" variant="outline" onClick={() => void refresh()}>
          Refresh
        </Button>
        <Button type="button" size="sm" onClick={() => void onStart()} disabled={busy}>
          {busy ? "Starting…" : "Start container"}
        </Button>
      </div>
    </section>
  );
}
