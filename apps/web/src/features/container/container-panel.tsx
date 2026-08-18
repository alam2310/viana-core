"use client";

import { useCallback, useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import type { ContainerStatus } from "@/lib/container-types";

export function ContainerPanel() {
  const [status, setStatus] = useState<ContainerStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    const response = await fetch("/api/container/status");
    const data = (await response.json()) as ContainerStatus;
    setStatus(data);
  }, []);

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

  return (
    <section className="rounded-lg border border-neutral-200 bg-white p-4">
      <h2 className="text-sm font-semibold tracking-wide text-neutral-500 uppercase">
        Container
      </h2>
      <p className="mt-2 font-mono text-sm">
        {status?.container_name ?? "…"} —{" "}
        <span className={status?.running ? "text-emerald-700" : "text-amber-700"}>
          {status?.running ? "running" : "down"}
        </span>
      </p>
      <p className="mt-1 text-xs text-neutral-500">
        Config: {status?.config_found ? "found" : "using example defaults"} (
        {status?.config_path})
      </p>
      {error || status?.error ? (
        <p className="mt-2 text-sm text-red-700">{error ?? status?.error}</p>
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
