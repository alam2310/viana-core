/**
 * In-memory mock job list for UI development. Seeded from contract fixtures;
 * does not invent API fields.
 */

import type {
  ConfirmedJobMetadata,
  JobIntakeRequest,
  JobIntakeResponse,
  JobPrescanConfirmRequest,
  JobStatusResponse,
  JobSubmitRequest,
  JobSubmitResponse,
} from "@viana/contracts";
import jobAwaitingReviewFixture from "@viana/fixtures/job_status_awaiting_review.json";
import jobIntakeFixture from "@viana/fixtures/job_intake_response.json";
import jobStatusPausedFixture from "@viana/fixtures/job_status_paused.json";
import jobSubmitResponseFixture from "@viana/fixtures/job_submit_response.json";

function asContract<T>(value: unknown): T {
  return value as T;
}

const jobs = new Map<string, JobStatusResponse>();
let submitSeq = 1;
let intakeSeq = 10;

function seed(): void {
  if (jobs.size > 0) {
    return;
  }
  const paused = asContract<JobStatusResponse>(jobStatusPausedFixture);
  jobs.set(paused.job_id, paused);
  const awaiting = asContract<JobStatusResponse>(jobAwaitingReviewFixture);
  jobs.set(awaiting.job_id, awaiting);
}

export function mockListJobs(projectId?: string): JobStatusResponse[] {
  seed();
  const all = Array.from(jobs.values());
  if (!projectId) {
    return all;
  }
  return all.filter((job) => job.project_id === projectId);
}

export function mockGetJob(jobId: string): JobStatusResponse | null {
  seed();
  return jobs.get(jobId) ?? null;
}

export function mockIntakeJobs(request: JobIntakeRequest): JobIntakeResponse {
  seed();
  const base = asContract<JobIntakeResponse>(jobIntakeFixture);
  const created = request.source_video_paths.map((path, index) => {
    intakeSeq += 1;
    const jobId = `job_intake_${String(intakeSeq).padStart(3, "0")}`;
    const outputDir =
      request.output_dir ?? `/data/viana-outputs/${request.project_id}`;
    const item = {
      job_id: jobId,
      status: "PRESCAN_PENDING" as const,
      source_video_path: path,
      output_dir: outputDir,
      queue_position: jobs.size + index + 1,
    };
    jobs.set(jobId, {
      job_id: jobId,
      status: "PRESCAN_PENDING",
      task_type: request.task_type ?? "ViAna_Moving",
      source_video_path: path,
      project_id: request.project_id,
      output_dir: outputDir,
      checkpoint_exists: false,
      queue_position: item.queue_position,
    });
    return item;
  });
  return { jobs: created.length > 0 ? created : base.jobs };
}

export function mockConfirmPrescan(
  jobId: string,
  body: JobPrescanConfirmRequest,
): JobStatusResponse | null {
  seed();
  const existing = jobs.get(jobId);
  if (!existing) {
    return null;
  }
  const updated: JobStatusResponse = {
    ...existing,
    status: "READY",
    confirmed_metadata: body.metadata as ConfirmedJobMetadata,
    confirmed_task_parameters: body.task_parameters,
  };
  jobs.set(jobId, updated);
  return updated;
}

export function mockRetryPrescan(jobId: string): JobStatusResponse | null {
  seed();
  const existing = jobs.get(jobId);
  if (!existing) {
    return null;
  }
  const updated: JobStatusResponse = {
    ...existing,
    status: "PRESCAN_PENDING",
    error_message: null,
  };
  jobs.set(jobId, updated);
  return updated;
}

export function mockRecordSubmit(
  request: JobSubmitRequest,
): JobSubmitResponse {
  seed();
  const base = asContract<JobSubmitResponse>(jobSubmitResponseFixture);
  submitSeq += 1;
  const response: JobSubmitResponse = {
    ...base,
    job_id: `job_mock_${String(submitSeq).padStart(3, "0")}`,
    status: "READY",
    queue_position: jobs.size,
    output_dir: `/data/viana-outputs/${request.project_id}`,
  };
  const status: JobStatusResponse = {
    job_id: response.job_id,
    status: "READY",
    task_type: request.task_type,
    source_video_path: request.source_video_path,
    project_id: request.project_id,
    output_dir: response.output_dir,
    checkpoint_exists: false,
    gpu_device: response.gpu_device,
    queue_position: response.queue_position,
    confirmed_metadata: request.metadata as ConfirmedJobMetadata | undefined,
    confirmed_task_parameters: request.task_parameters,
  };
  jobs.set(status.job_id, status);
  return response;
}

export function mockResumeJob(jobId: string): JobStatusResponse | null {
  seed();
  const existing = jobs.get(jobId);
  if (!existing) {
    return null;
  }
  const updated: JobStatusResponse = { ...existing, status: "PROCESSING" };
  jobs.set(jobId, updated);
  return updated;
}

export function mockStartFreshJob(jobId: string): JobStatusResponse | null {
  seed();
  const existing = jobs.get(jobId);
  if (!existing) {
    return null;
  }
  const updated: JobStatusResponse = {
    ...existing,
    status: "READY",
    checkpoint_exists: false,
    progress: undefined,
  };
  jobs.set(jobId, updated);
  return updated;
}
