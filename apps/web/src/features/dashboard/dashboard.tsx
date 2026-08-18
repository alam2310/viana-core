"use client";

import { useEffect, useMemo, useState } from "react";
import type {
  JobMetadata,
  JobStatusResponse,
  LineSegment,
  TelemetryMessage,
  VideoMeta,
} from "@viana/contracts";

import { ContainerPanel } from "@/features/container/container-panel";
import {
  PrescanModal,
  type CalibrationDraft,
} from "@/features/prescan/prescan-modal";
import { QueuePanel } from "@/features/queue/queue-panel";
import { TelemetryPanel } from "@/features/telemetry/telemetry-panel";
import {
  apiClient,
  getHealth,
  listJobs,
  resumeJob,
  startFreshJob,
  submitJob,
  subscribeJobTelemetry,
  type HealthResponse,
} from "@/lib/api-client";
import {
  PROJECT_ID_PATTERN,
  scaleLine,
  validateCalibration,
} from "@/lib/geometry";
import {
  readPendingPaths,
  readProjectId,
  readTelemetryDetail,
  writePendingPaths,
  writeProjectId,
  writeTelemetryDetail,
} from "@/lib/prefs";

interface DraftCalib {
  horizon_line: LineSegment;
  counting_line: LineSegment;
  metadata: JobMetadata;
  video_meta: VideoMeta;
}

export function Dashboard() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [jobs, setJobs] = useState<JobStatusResponse[]>([]);
  const [telemetry, setTelemetry] = useState<TelemetryMessage | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [projectId, setProjectId] = useState("nh48");
  const [pendingPaths, setPendingPaths] = useState<string[]>([]);
  const [newPath, setNewPath] = useState(
    "/data/projects/nh48/videos/2026-03-15_09-00.mp4",
  );
  const [prescanPath, setPrescanPath] = useState<string | null>(null);
  const [drafts, setDrafts] = useState<Record<string, DraftCalib>>({});
  const [template, setTemplate] = useState<DraftCalib | null>(null);
  const [telemetryDetail, setTelemetryDetail] = useState(false);
  const [busyId, setBusyId] = useState<string | null>(null);

  const projectValid = PROJECT_ID_PATTERN.test(projectId);

  useEffect(() => {
    setProjectId(readProjectId());
    setPendingPaths(readPendingPaths());
    setTelemetryDetail(readTelemetryDetail());
  }, []);

  async function refreshJobs(id = projectId) {
    const list = await listJobs(id);
    setJobs(list);
  }

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const h = await getHealth();
        if (!cancelled) {
          setHealth(h);
        }
        await refreshJobs(projectId);
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
        }
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- load once; project filter on demand
  }, []);

  useEffect(() => {
    return subscribeJobTelemetry((message) => {
      setTelemetry(message);
    });
  }, []);

  function persistPending(paths: string[]) {
    setPendingPaths(paths);
    writePendingPaths(paths);
  }

  function onConfirmCalibration(draft: CalibrationDraft) {
    const calib: DraftCalib = {
      horizon_line: draft.horizon_line,
      counting_line: draft.counting_line,
      metadata: draft.metadata,
      video_meta: draft.video_meta,
    };
    setDrafts((prev) => ({ ...prev, [draft.source_video_path]: calib }));
    setTemplate(calib);
    if (draft.applyToPending) {
      setDrafts((prev) => {
        const next = { ...prev };
        for (const path of pendingPaths) {
          const existing = next[path];
          if (existing) {
            next[path] = {
              ...existing,
              horizon_line: scaleLine(
                calib.horizon_line,
                calib.video_meta.width,
                calib.video_meta.height,
                existing.video_meta.width,
                existing.video_meta.height,
              ),
              counting_line: scaleLine(
                calib.counting_line,
                calib.video_meta.width,
                calib.video_meta.height,
                existing.video_meta.width,
                existing.video_meta.height,
              ),
            };
          } else {
            next[path] = calib;
          }
        }
        next[draft.source_video_path] = calib;
        return next;
      });
    }
    setPrescanPath(null);
  }

  async function onSubmitPending(path: string) {
    if (!PROJECT_ID_PATTERN.test(projectId)) {
      setError("Invalid project_id");
      return;
    }
    setError(null);
    const draft = drafts[path] ?? template;
    if (!draft) {
      setError("Calibrate this video (prescan) before submit.");
      setPrescanPath(path);
      return;
    }
    const issues = validateCalibration(
      draft.horizon_line,
      draft.counting_line,
      draft.video_meta.width,
      draft.video_meta.height,
    );
    if (issues.length > 0) {
      setError(issues.join("; "));
      return;
    }
    try {
      await submitJob({
        task_type: "ViAna_Moving",
        source_video_path: path,
        project_id: projectId,
        metadata: draft.metadata,
        task_parameters: {
          horizon_line: draft.horizon_line,
          counting_line: draft.counting_line,
          confidence_threshold: 0.75,
          use_heuristic_truck_split: true,
          render_video: true,
          telemetry_detail: telemetryDetail,
        },
      });
      persistPending(pendingPaths.filter((item) => item !== path));
      await refreshJobs();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function onResume(jobId: string) {
    setBusyId(jobId);
    setError(null);
    try {
      await resumeJob(jobId);
      await refreshJobs();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusyId(null);
    }
  }

  async function onStartFresh(jobId: string) {
    setBusyId(jobId);
    setError(null);
    try {
      await startFreshJob(jobId);
      await refreshJobs();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusyId(null);
    }
  }

  const calibratedCount = useMemo(
    () => pendingPaths.filter((path) => drafts[path] || template).length,
    [drafts, pendingPaths, template],
  );

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-6 p-6">
      <header>
        <p className="text-xs font-medium tracking-widest text-neutral-500 uppercase">
          ViAna Moving Count
        </p>
        <h1 className="mt-1 text-2xl font-semibold">Dashboard</h1>
        <p className="mt-2 text-sm text-neutral-600">
          Phase 8 workflows. API is{" "}
          {apiClient.useMocks ? (
            <strong>mocked from packages/contracts/fixtures</strong>
          ) : (
            <span>
              live at <code>{apiClient.apiBaseUrl}</code>
            </span>
          )}
          . Task type is ViAna_Moving only.
        </p>
      </header>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
        <div className="flex flex-col gap-6">
          <ContainerPanel />
          <section className="rounded-lg border border-neutral-200 bg-white p-4">
            <h2 className="text-sm font-semibold tracking-wide text-neutral-500 uppercase">
              Project
            </h2>
            <label className="mt-2 block text-sm">
              project_id
              <input
                className="mt-1 w-full rounded border border-neutral-300 px-2 py-1 font-mono text-sm"
                value={projectId}
                onChange={(event) => {
                  const value = event.target.value;
                  setProjectId(value);
                  if (PROJECT_ID_PATTERN.test(value)) {
                    writeProjectId(value);
                    void refreshJobs(value);
                  }
                }}
              />
            </label>
            {!projectValid ? (
              <p className="mt-1 text-xs text-red-700">
                Must match ^[a-z0-9][a-z0-9_-]*$
              </p>
            ) : null}
            <p className="mt-2 font-mono text-xs text-neutral-500">
              Health:{" "}
              {health
                ? `${health.status}${health.phase !== undefined ? ` (phase ${health.phase})` : ""}`
                : "…"}
            </p>
          </section>
          <QueuePanel
            jobs={jobs}
            pendingPaths={pendingPaths}
            newPath={newPath}
            onNewPath={setNewPath}
            onAddPending={() => {
              const path = newPath.trim();
              if (!path || pendingPaths.includes(path)) {
                return;
              }
              persistPending([...pendingPaths, path]);
              setNewPath("");
            }}
            onRemovePending={(path) =>
              persistPending(pendingPaths.filter((item) => item !== path))
            }
            onPrescan={setPrescanPath}
            onSubmitPending={(path) => void onSubmitPending(path)}
            onResume={(id) => void onResume(id)}
            onStartFresh={(id) => void onStartFresh(id)}
            busyId={busyId}
          />
        </div>
        <div className="flex flex-col gap-6">
          {error ? (
            <p className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800">
              {error}
            </p>
          ) : null}
          <p className="text-xs text-neutral-500">
            {calibratedCount}/{pendingPaths.length} pending videos have
            calibration ready.
          </p>
          <TelemetryPanel
            message={telemetry}
            telemetryDetail={telemetryDetail}
            onTelemetryDetail={(value) => {
              setTelemetryDetail(value);
              writeTelemetryDetail(value);
            }}
          />
        </div>
      </div>

      {prescanPath && projectValid ? (
        <PrescanModal
          projectId={projectId}
          sourceVideoPath={prescanPath}
          onClose={() => setPrescanPath(null)}
          onConfirm={onConfirmCalibration}
        />
      ) : null}
    </div>
  );
}
