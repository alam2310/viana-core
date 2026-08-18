"use client";

import { useEffect, useState } from "react";
import type { JobStatusResponse, TelemetryMessage } from "@viana/contracts";

import { ContainerPanel } from "@/features/container/container-panel";
import {
  apiClient,
  getHealth,
  listJobs,
  subscribeJobTelemetry,
  type HealthResponse,
} from "@/lib/api-client";

export function Dashboard() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [jobs, setJobs] = useState<JobStatusResponse[]>([]);
  const [telemetry, setTelemetry] = useState<TelemetryMessage | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const [h, j] = await Promise.all([getHealth(), listJobs()]);
        if (!cancelled) {
          setHealth(h);
          setJobs(j);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    return subscribeJobTelemetry((message) => {
      setTelemetry(message);
    });
  }, []);

  return (
    <div className="mx-auto flex max-w-5xl flex-col gap-6 p-6">
      <header>
        <p className="text-xs font-medium tracking-widest text-neutral-500 uppercase">
          ViAna Moving Count
        </p>
        <h1 className="mt-1 text-2xl font-semibold">Dashboard</h1>
        <p className="mt-2 text-sm text-neutral-600">
          Phase 7 foundation. Job API is{" "}
          {apiClient.useMocks ? (
            <strong>mocked from packages/contracts/fixtures</strong>
          ) : (
            <span>
              live at <code>{apiClient.apiBaseUrl}</code>
            </span>
          )}
          . Prescan, calibration, and queue workflows land in Phase 8.
        </p>
      </header>

      <ContainerPanel />

      <section className="rounded-lg border border-neutral-200 bg-white p-4">
        <h2 className="text-sm font-semibold tracking-wide text-neutral-500 uppercase">
          Orchestrator health
        </h2>
        {error ? <p className="mt-2 text-sm text-red-700">{error}</p> : null}
        <p className="mt-2 font-mono text-sm">
          {health
            ? `${health.status}${health.phase !== undefined ? ` (phase ${health.phase})` : ""}`
            : "…"}
        </p>
      </section>

      <section className="rounded-lg border border-neutral-200 bg-white p-4">
        <h2 className="text-sm font-semibold tracking-wide text-neutral-500 uppercase">
          Job queue
        </h2>
        <p className="mt-1 text-xs text-neutral-500">
          Synced from GET /jobs (fixture while endpoint is unimplemented).
        </p>
        <ul className="mt-3 divide-y divide-neutral-100">
          {jobs.map((job) => (
            <li key={job.job_id} className="py-3">
              <div className="flex items-baseline justify-between gap-4">
                <span className="font-mono text-sm">{job.job_id}</span>
                <span
                  className={
                    job.status === "PAUSED"
                      ? "text-sm font-semibold text-amber-700"
                      : "text-sm text-neutral-700"
                  }
                >
                  {job.status}
                </span>
              </div>
              <p className="mt-1 truncate text-xs text-neutral-500">
                {job.source_video_path}
              </p>
              {job.progress ? (
                <p className="mt-1 text-xs text-neutral-600">
                  Frame {job.progress.current_frame} / {job.progress.total_frames}
                  {job.progress.processing_fps
                    ? ` · ${job.progress.processing_fps} fps`
                    : ""}
                </p>
              ) : null}
            </li>
          ))}
          {jobs.length === 0 ? (
            <li className="py-3 text-sm text-neutral-500">No jobs</li>
          ) : null}
        </ul>
      </section>

      <section className="rounded-lg border border-neutral-200 bg-white p-4">
        <h2 className="text-sm font-semibold tracking-wide text-neutral-500 uppercase">
          Telemetry
        </h2>
        <pre className="mt-3 overflow-x-auto rounded bg-neutral-50 p-3 text-xs">
          {telemetry ? JSON.stringify(telemetry, null, 2) : "waiting…"}
        </pre>
      </section>
    </div>
  );
}
