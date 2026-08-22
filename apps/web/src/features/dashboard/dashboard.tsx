"use client";

import { useCallback, useEffect, useState } from "react";
import type { JobStatusResponse, TelemetryMessage } from "@viana/contracts";

import { EngineControls } from "@/features/container/engine-controls";
import { IntakePanel } from "@/features/intake/intake-panel";
import { PathBrowser } from "@/features/intake/path-browser";
import { PrescanReviewModal } from "@/features/prescan/prescan-review-modal";
import { ProjectBar } from "@/features/project/project-bar";
import { shouldPollJobs } from "@/features/queue/job-status";
import { JobQueueTable } from "@/features/queue/job-queue-table";
import { JobDetailsPanel } from "@/features/telemetry/job-details-panel";
import { IconMoon, IconSun, RoundIconButton } from "@/components/ui/icon-button";
import {
  cancelJob,
  getJob,
  getHealth,
  intakeJobs,
  listJobs,
  startFreshJob,
  subscribeJobTelemetry,
} from "@/lib/api-client";
import { formatJobErrorMessage } from "@/lib/job-errors";
import { PROJECT_ID_PATTERN } from "@/lib/geometry";
import {
  type MountConfig,
  toContainerPath,
  toHostPath,
} from "@/lib/container-paths";
import type { ContainerStatus } from "@/lib/container-types";
import { openPathInFileManager } from "@/lib/fs-open";
import { ensureProjectOutputDir } from "@/lib/output-paths";
import {
  DEFAULT_PROJECT_ID,
  readOutputDir,
  readProjectId,
  readTaskType,
  readThemePreference,
  writeOutputDir,
  writeProjectId,
  writeTaskType,
  writeThemePreference,
  type TaskTypePref,
  type UiTheme,
} from "@/lib/prefs";

function applyTelemetryToJob(
  job: JobStatusResponse,
  message: TelemetryMessage,
): JobStatusResponse {
  if (job.job_id !== message.job_id) {
    return job;
  }
  if (message.telemetry_type !== "PROGRESS") {
    return { ...job, status: message.status ?? job.status };
  }
  const data = message.data;
  const current =
    typeof data.current_frame === "number"
      ? data.current_frame
      : job.progress?.current_frame;
  const total =
    typeof data.total_frames === "number"
      ? data.total_frames
      : job.progress?.total_frames;
  const crossingCount =
    typeof data.crossing_count === "number"
      ? data.crossing_count
      : job.progress?.crossing_count;
  if (current === undefined || total === undefined) {
    return {
      ...job,
      status: message.status ?? job.status,
      progress:
        job.progress && crossingCount !== undefined
          ? { ...job.progress, crossing_count: crossingCount }
          : job.progress,
    };
  }
  return {
    ...job,
    status: message.status ?? job.status,
    progress: {
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
      crossing_count: crossingCount,
    },
  };
}

export function Dashboard() {
  const [jobs, setJobs] = useState<JobStatusResponse[]>([]);
  const [telemetry, setTelemetry] = useState<TelemetryMessage[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [projectId, setProjectId] = useState(DEFAULT_PROJECT_ID);
  const [outputDir, setOutputDir] = useState("");
  const [taskType, setTaskType] = useState<TaskTypePref>("ViAna_Moving");
  const [busyId, setBusyId] = useState<string | null>(null);
  const [intakeBusy, setIntakeBusy] = useState(false);
  const [reviewJob, setReviewJob] = useState<JobStatusResponse | null>(null);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [browseOutputDir, setBrowseOutputDir] = useState(false);
  const [mountConfig, setMountConfig] = useState<MountConfig | null>(null);
  const [apiReachable, setApiReachable] = useState<boolean | null>(null);
  const [jobsRefreshError, setJobsRefreshError] = useState<string | null>(null);
  const [theme, setTheme] = useState<UiTheme>("light");
  const [containerStatus, setContainerStatus] = useState<ContainerStatus | null>(
    null,
  );

  const projectValid = PROJECT_ID_PATTERN.test(projectId);
  const orchestratorUp = apiReachable === true && containerStatus?.running === true;

  /** Never throws — pollers must not surface unhandled ApiClientError (S30 / F025). */
  const refreshJobs = useCallback(async (id = projectId): Promise<boolean> => {
    try {
      const list = await listJobs(id);
      setJobs((prev) =>
        list.map((incoming) => {
          const existing = prev.find((job) => job.job_id === incoming.job_id);
          const prevCount = existing?.progress?.crossing_count;
          const nextCount = incoming.progress?.crossing_count;
          if (
            typeof prevCount === "number" &&
            (typeof nextCount !== "number" || prevCount > nextCount)
          ) {
            return {
              ...incoming,
              progress: incoming.progress
                ? { ...incoming.progress, crossing_count: prevCount }
                : existing?.progress,
            };
          }
          return incoming;
        }),
      );
      setJobsRefreshError(null);
      return true;
    } catch (err) {
      setJobsRefreshError(
        err instanceof Error ? err.message : String(err),
      );
      return false;
    }
  }, [projectId]);

  function applyTheme(nextTheme: UiTheme): void {
    document.documentElement.dataset.theme = nextTheme;
    document.documentElement.style.colorScheme = nextTheme;
    setTheme(nextTheme);
  }

  useEffect(() => {
    setProjectId(readProjectId());
    setOutputDir(readOutputDir());
    setTaskType(readTaskType());
    const preferred = readThemePreference();
    const resolved =
      preferred ??
      (window.matchMedia("(prefers-color-scheme: dark)").matches
        ? "dark"
        : "light");
    applyTheme(resolved);
  }, []);

  useEffect(() => {
    void fetch("/api/container/mounts")
      .then((response) => response.json())
      .then((data: MountConfig) => setMountConfig(data))
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function probeApi() {
      try {
        await getHealth();
        if (!cancelled) {
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
    const timer = window.setInterval(() => void probeApi(), 5000);
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
        prev.map((job) => applyTelemetryToJob(job, message)),
      );
    });
  }, []);

  function toggleTheme() {
    const next: UiTheme = theme === "dark" ? "light" : "dark";
    applyTheme(next);
    writeThemePreference(next);
  }

  async function onIntake(paths: string[]) {
    if (!projectValid || taskType !== "ViAna_Moving") {
      setError("Set a valid project ID and select an analytics type.");
      return;
    }
    setIntakeBusy(true);
    setError(null);
    try {
      let resolvedOutput = "";
      if (outputDir.trim() && mountConfig) {
        resolvedOutput = await ensureProjectOutputDir(
          outputDir.trim(),
          projectId,
          mountConfig.mounts,
        );
      }
      await intakeJobs({
        task_type: "ViAna_Moving",
        project_id: projectId,
        source_video_paths: paths,
        ...(resolvedOutput ? { output_dir: resolvedOutput } : {}),
      });
      const refreshed = await refreshJobs();
      if (!refreshed) {
        setError(
          "Intake succeeded, but the job list could not be refreshed. It should update shortly.",
        );
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setIntakeBusy(false);
    }
  }

  async function onStartFresh(jobId: string) {
    setBusyId(jobId);
    setError(null);
    try {
      await startFreshJob(jobId);
      const refreshed = await refreshJobs();
      if (!refreshed) {
        setError(
          "Restart accepted, but the job list could not be refreshed. It should update shortly.",
        );
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusyId(null);
    }
  }

  async function onStop(jobId: string) {
    setBusyId(jobId);
    setError(null);
    try {
      await cancelJob(jobId);
      for (let attempt = 0; attempt < 20; attempt += 1) {
        await new Promise<void>((resolve) => {
          window.setTimeout(resolve, 250);
        });
        const latest = await getJob(jobId);
        if (latest.status === "CANCELLED") {
          await refreshJobs();
          return;
        }
        if (latest.status === "PAUSED") {
          await cancelJob(jobId);
          await refreshJobs();
          return;
        }
        if (latest.status !== "PROCESSING") {
          await refreshJobs();
          return;
        }
      }
      await refreshJobs();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusyId(null);
    }
  }

  const selectedJob =
    selectedJobId === null
      ? null
      : (jobs.find((job) => job.job_id === selectedJobId) ?? null);

  return (
    <div className="mx-auto flex w-full max-w-[min(100%,96rem)] flex-col gap-6 p-4 sm:p-6">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">
          Vehicle Analytics Dashboard
        </h1>
        <div className="flex items-center gap-2">
          <RoundIconButton
            label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
            onClick={toggleTheme}
          >
            {theme === "dark" ? <IconSun /> : <IconMoon />}
          </RoundIconButton>
          <EngineControls
            label="Analytics Engine Status"
            onStatusChange={setContainerStatus}
          />
        </div>
      </header>

      {apiReachable === false || containerStatus?.running === false ? (
        <p className="rounded-lg border border-amber-300 bg-amber-50 px-3 py-2 text-sm text-amber-950 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-100">
          {containerStatus?.running === false ? (
            <>
              The analytics engine is not running. Use the controls at the top
              right to start it.
            </>
          ) : (
            <>
              The analytics engine is not responding yet. Start it from the top
              right or wait a few seconds if it is still starting.
            </>
          )}
        </p>
      ) : null}

      {apiReachable === true &&
      containerStatus?.running !== false &&
      jobsRefreshError ? (
        <p className="rounded-lg border border-amber-300 bg-amber-50 px-3 py-2 text-sm text-amber-950 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-100">
          Job list refresh failed briefly (
          {formatJobErrorMessage(jobsRefreshError) ?? jobsRefreshError}).
          Showing the last known queue; retrying automatically.
        </p>
      ) : null}

      {error ? (
        <p className="rounded-lg border border-red-300 bg-red-50 px-3 py-2 text-sm text-red-800 dark:border-red-800 dark:bg-red-950 dark:text-red-200">
          {formatJobErrorMessage(error) ?? error}
        </p>
      ) : null}

      <ProjectBar
        projectId={projectId}
        outputDir={outputDir}
        taskType={taskType}
        projectValid={projectValid}
        onProjectId={(value) => {
          setProjectId(value);
          if (value && PROJECT_ID_PATTERN.test(value)) {
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
      />

      <IntakePanel
        disabled={!projectValid || taskType !== "ViAna_Moving" || !orchestratorUp}
        busy={intakeBusy}
        mountConfig={mountConfig}
        onIntake={onIntake}
      />

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_minmax(18rem,22rem)]">
        <JobQueueTable
          jobs={jobs}
          busyId={busyId}
          selectedJobId={selectedJobId}
          onSelectJob={(job) => setSelectedJobId(job.job_id)}
          onReview={setReviewJob}
          onStartFresh={(id) => void onStartFresh(id)}
          onStop={(id) => void onStop(id)}
          onOpenOutput={(job) => {
            if (!mountConfig || !job.output_dir) {
              return;
            }
            const hostPath =
              toHostPath(job.output_dir, mountConfig.mounts) ?? job.output_dir;
            void openPathInFileManager(hostPath).catch((err) => {
              setError(err instanceof Error ? err.message : String(err));
            });
          }}
        />
        <JobDetailsPanel
          job={selectedJob}
          mountConfig={mountConfig}
          messages={telemetry}
        />
      </div>

      {reviewJob ? (
        <PrescanReviewModal
          job={reviewJob}
          projectId={projectId}
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
