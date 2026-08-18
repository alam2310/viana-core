/**
 * In-memory mock job list for Phase 8. Seeded from contract fixtures;
 * does not invent API fields.
 */

import type {
  JobStatusResponse,
  JobSubmitRequest,
  JobSubmitResponse,
} from "@viana/contracts";
import jobStatusPausedFixture from "@viana/fixtures/job_status_paused.json";
import jobSubmitResponseFixture from "@viana/fixtures/job_submit_response.json";

function asContract<T>(value: unknown): T {
  return value as T;
}

const jobs = new Map<string, JobStatusResponse>();
let submitSeq = 1;

function seed(): void {
  if (jobs.size > 0) {
    return;
  }
  const paused = asContract<JobStatusResponse>(jobStatusPausedFixture);
  jobs.set(paused.job_id, paused);
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

export function mockRecordSubmit(
  request: JobSubmitRequest,
): JobSubmitResponse {
  seed();
  const base = asContract<JobSubmitResponse>(jobSubmitResponseFixture);
  submitSeq += 1;
  const response: JobSubmitResponse = {
    ...base,
    job_id: `job_mock_${String(submitSeq).padStart(3, "0")}`,
    status: "PENDING",
    queue_position: jobs.size,
    output_dir: `/data/viana-outputs/${request.project_id}`,
  };
  const status: JobStatusResponse = {
    job_id: response.job_id,
    status: response.status,
    task_type: request.task_type,
    source_video_path: request.source_video_path,
    project_id: request.project_id,
    output_dir: response.output_dir,
    checkpoint_exists: false,
    gpu_device: response.gpu_device,
    queue_position: response.queue_position,
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
    status: "PENDING",
    checkpoint_exists: false,
    progress: undefined,
  };
  jobs.set(jobId, updated);
  return updated;
}
