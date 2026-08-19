"use client";

import { useCallback, useEffect, useState } from "react";
import type { JobStatusResponse, TelemetryMessage } from "@viana/contracts";

import { IntakePanel } from "@/features/intake/intake-panel";
import { PathBrowser } from "@/features/intake/path-browser";
import { MonitorSidebar } from "@/features/monitor/monitor-sidebar";
import { PrescanReviewModal } from "@/features/prescan/prescan-review-modal";
import { ProjectBar } from "@/features/project/project-bar";
import { shouldPollJobs } from "@/features/queue/job-status";
import { JobQueueTable } from "@/features/queue/job-queue-table";
import { TelemetryPanel } from "@/features/telemetry/telemetry-panel";
import {
  apiClient,
  cancelJob,
  getHealth,
  intakeJobs,
  listJobs,
  resumeJob,
  retryPrescan,
  startFreshJob,
  subscribeJobTelemetry,
  type HealthResponse,
} from "@/lib/api-client";
import { PROJECT_ID_PATTERN } from "@/lib/geometry";
import {
  type MountConfig,
  toContainerPath,
} from "@/lib/container-paths";
import type { ContainerStatus } from "@/lib/container-types";
import {
  readOutputDir,
  readProjectId,
  readTaskType,
  readTelemetryDetail,
  writeOutputDir,
  writeProjectId,
  writeTaskType,
  writeTelemetryDetail,
  type TaskTypePref,
} from "@/lib/prefs";

export function Dashboard() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [jobs, setJobs] = useState<JobStatusResponse[]>([]);
  const [telemetry, setTelemetry] = useState<TelemetryMessage[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [projectId, setProjectId] = useState("nh48");
  const [outputDir, setOutputDir] = useState("");
  const [taskType, setTaskType] = useState<TaskTypePref>("ViAna_Moving");
  const [telemetryDetail, setTelemetryDetail] = useState(false);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [intakeBusy, setIntakeBusy] = useState(false);
  const [reviewJob, setReviewJob] = useState<JobStatusResponse | null>(null);
  const [monitorJob, setMonitorJob] = useState<JobStatusResponse | null>(null);
  const [browseOutputDir, setBrowseOutputDir] = useState(false);
  const [mountConfig, setMountConfig] = useState<MountConfig | null>(null);
  const [apiReachable, setApiReachable] = useState<boolean | null>(null);
  const [containerStatus, setContainerStatus] = useState<ContainerStatus | null>(
    null,
  );

  const projectValid = PROJECT_ID_PATTERN.test(projectId);
  const orchestratorUp = apiReachable === true && containerStatus?.running === true;

  const refreshJobs = useCallback(async (id = projectId) => {
    const list = await listJobs(id);
    setJobs(list);
    return list;
  }, [projectId]);

  useEffect(() => {
    setProjectId(readProjectId());
    setOutputDir(readOutputDir());
    setTaskType(readTaskType());
    setTelemetryDetail(readTelemetryDetail());
  }, []);

  useEffect(() => {
    void fetch("/api/container/mounts")
      .then((response) => response.json())
      .then((data: MountConfig) => setMountConfig(data))
      .catch(() => {
        /* mounts route is host-only; intake stays disabled until loaded */
      });
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function probeApi() {
      try {
        const h = await getHealth();
        if (!cancelled) {
          setHealth(h);
          setApiReachable(true);
          await refreshJobs(projectId);
        }
      } catch {
        if (!cancelled) {
          setApiReachable(false);
        }
      }
    }

    void probeApi();
    const timer = window.setInterval(() => {
      void probeApi();
    }, 5000);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [projectId, refreshJobs]);

  useEffect(() => {
    if (apiReachable !== true) {
      return;
    }
    const poll = shouldPollJobs(jobs);
    const intervalMs = poll ? 2000 : 8000;
    const timer = window.setInterval(() => {
      void refreshJobs();
    }, intervalMs);
    return () => window.clearInterval(timer);
  }, [apiReachable, jobs, refreshJobs]);

  useEffect(() => {
    return subscribeJobTelemetry((message) => {
      setTelemetry((prev) => [...prev.slice(-499), message]);
      setJobs((prev) =>
        prev.map((job) => {
          if (job.job_id !== message.job_id) {
            return job;
          }
          if (message.telemetry_type !== "PROGRESS") {
            return { ...job, status: message.status ?? job.status };
          }
          const data = message.data;
          const current =
            typeof data.current_frame === "number" ? data.current_frame : undefined;
          const total =
            typeof data.total_frames === "number" ? data.total_frames : undefined;
          return {
            ...job,
            status: message.status ?? job.status,
            progress:
              current !== undefined && total !== undefined
                ? {
                    current_frame: current,
                    total_frames: total,
                    processing_fps:
                      typeof data.processing_fps === "number"
                        ? data.processing_fps
                        : job.progress?.processing_fps,
                    eta_sec:
                      typeof data.eta_sec === "number"
                        ? data.eta_sec
                        : job.progress?.eta_sec,
                    crossing_count:
                      typeof data.crossing_count === "number"
                        ? data.crossing_count
                        : job.progress?.crossing_count,
                  }
                : job.progress,
          };
        }),
      );
    });
  }, []);

  async function onIntake(paths: string[]) {
    if (!projectValid || taskType !== "ViAna_Moving") {
      setError("Set a valid project_id and select ViAna_Moving.");
      return;
    }
    setIntakeBusy(true);
    setError(null);
    let createdJobIds: string[] = [];
    try {
      const response = await intakeJobs({
        task_type: "ViAna_Moving",
        project_id: projectId,
        source_video_paths: paths,
        ...(outputDir.trim() ? { output_dir: outputDir.trim() } : {}),
      });
      createdJobIds = response.jobs.map((item) => item.job_id);
      await refreshJobs();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setIntakeBusy(false);
    }

    if (createdJobIds.length === 1) {
      void waitForPrescanReview(createdJobIds[0]);
    }
  }

  async function waitForPrescanReview(jobId: string) {
    for (let attempt = 0; attempt < 60; attempt += 1) {
      try {
        const list = await refreshJobs();
        const job = list.find((item) => item.job_id === jobId);
        if (job?.status === "AWAITING_REVIEW") {
          setReviewJob(job);
          return;
        }
        if (job?.status === "PRESCAN_FAILED") {
          setError(job.error_message ?? "Prescan failed");
          return;
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
        return;
      }
      await new Promise((resolve) => window.setTimeout(resolve, 2000));
    }
  }

  async function onRetryPrescan(jobId: string) {
    setBusyId(jobId);
    setError(null);
    try {
      await retryPrescan(jobId);
      await refreshJobs();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusyId(null);
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

  async function onCancel(jobId: string) {
    setBusyId(jobId);
    setError(null);
    try {
      await cancelJob(jobId);
      if (monitorJob?.job_id === jobId) {
        setMonitorJob(null);
      }
      await refreshJobs();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusyId(null);
    }
  }

  return (
    <div className="mx-auto flex max-w-7xl flex-col gap-6 p-6">
      <header>
        <p className="text-xs font-medium tracking-widest text-neutral-500 uppercase">
          ViAna Moving Count
        </p>
        <h1 className="mt-1 text-2xl font-semibold">Dashboard</h1>
        <p className="mt-2 text-sm text-neutral-600">
          Backend-owned job queue. API is{" "}
          {apiClient.useMocks ? (
            <strong>mocked from packages/contracts/fixtures</strong>
          ) : (
            <span>
              live at <code>{apiClient.apiBaseUrl}</code>
            </span>
          )}
          {health ? ` · health ${health.status}` : ""}
          {apiReachable === false ? " · orchestrator unreachable" : ""}
        </p>
      </header>

      {apiReachable === false || containerStatus?.running === false ? (
        <p className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900">
          {containerStatus?.running === false ? (
            <>
              The ViAna orchestrator container is not running. Use the{" "}
              <strong>Start</strong> button in the project bar to launch it, then
              click <strong>Refresh</strong> once it is up.
            </>
          ) : (
            <>
              The orchestrator API is not responding yet. Start the ViAna container
              from the project bar, or wait a few seconds if it is still starting.
            </>
          )}
        </p>
      ) : null}

      {error ? (
        <p className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800">
          {error}
        </p>
      ) : null}

      <ProjectBar
        projectId={projectId}
        outputDir={outputDir}
        taskType={taskType}
        projectValid={projectValid}
        onProjectId={(value) => {
          setProjectId(value);
          if (PROJECT_ID_PATTERN.test(value)) {
            writeProjectId(value);
            if (apiReachable) {
              void refreshJobs(value);
            }
          }
        }}
        onOutputDir={(value) => {
          setOutputDir(value);
          writeOutputDir(value);
        }}
        onTaskType={(value) => {
          setTaskType(value);
          writeTaskType(value);
        }}
        onBrowseOutputDir={() => setBrowseOutputDir(true)}
        onContainerStatus={setContainerStatus}
      />

      <IntakePanel
        disabled={!projectValid || taskType !== "ViAna_Moving" || !orchestratorUp}
        busy={intakeBusy}
        mountConfig={mountConfig}
        onIntake={onIntake}
      />

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,22rem)]">
        <JobQueueTable
          jobs={jobs}
          busyId={busyId}
          monitorJobId={monitorJob?.job_id ?? null}
          onReview={setReviewJob}
          onMonitor={(job) => {
            setMonitorJob(job);
            setTelemetry([]);
          }}
          onRetryPrescan={(id) => void onRetryPrescan(id)}
          onResume={(id) => void onResume(id)}
          onStartFresh={(id) => void onStartFresh(id)}
          onCancel={(id) => void onCancel(id)}
        />
        {monitorJob ? (
          <MonitorSidebar
            job={monitorJob}
            messages={telemetry}
            onClose={() => setMonitorJob(null)}
          />
        ) : (
          <TelemetryPanel
            messages={telemetry}
            focusedJobId={null}
            telemetryDetail={telemetryDetail}
            onTelemetryDetail={(value) => {
              setTelemetryDetail(value);
              writeTelemetryDetail(value);
            }}
          />
        )}
      </div>

      {reviewJob ? (
        <PrescanReviewModal
          job={reviewJob}
          projectId={projectId}
          telemetryDetail={telemetryDetail}
          awaitingReviewJobs={jobs}
          onClose={() => setReviewJob(null)}
          onConfirmed={() => void refreshJobs()}
        />
      ) : null}

      {browseOutputDir ? (
        <PathBrowser
          purpose="output_dir"
          open
          mountConfig={mountConfig}
          onClose={() => setBrowseOutputDir(false)}
          onSelect={(paths) => {
            if (paths[0] && mountConfig) {
              const { containerPath } = toContainerPath(
                paths[0],
                mountConfig.mounts,
              );
              setOutputDir(containerPath);
              writeOutputDir(containerPath);
            }
            setBrowseOutputDir(false);
          }}
        />
      ) : null}
    </div>
  );
}
