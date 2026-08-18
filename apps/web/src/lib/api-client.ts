/**
 * Job API client. Mock path uses packages/contracts/fixtures until
 * docs/PROJECT_STATUS.md marks each endpoint ✅.
 *
 * Never send job_id or gpu_device on POST /jobs.
 */

import type {
  JobStatusResponse,
  JobSubmitRequest,
  JobSubmitResponse,
  PrescanResponse,
  TelemetryMessage,
} from "@viana/contracts";
import jobStatusPausedFixture from "@viana/fixtures/job_status_paused.json";
import jobSubmitResponseFixture from "@viana/fixtures/job_submit_response.json";
import prescanFixture from "@viana/fixtures/prescan_response.json";
import telemetryProgressFixture from "@viana/fixtures/telemetry_progress.json";

import { API_BASE_URL, USE_MOCKS } from "./env";

/** Request body documented in docs/api_contracts.md (no TS type in contracts yet). */
export interface PrescanRequestBody {
  source_video_path: string;
  project_id: string;
  frame_offset_sec?: number;
}

function asContract<T>(value: unknown): T {
  return value as T;
}

export class ApiClientError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly body?: unknown,
  ) {
    super(message);
    this.name = "ApiClientError";
  }
}

/** Whitelist JobSubmitRequest fields so job_id / gpu_device cannot leak. */
export function toJobSubmitPayload(body: JobSubmitRequest): JobSubmitRequest {
  return {
    task_type: body.task_type,
    source_video_path: body.source_video_path,
    project_id: body.project_id,
    ...(body.metadata ? { metadata: body.metadata } : {}),
    task_parameters: body.task_parameters,
    ...(body.calibration_profile_id
      ? { calibration_profile_id: body.calibration_profile_id }
      : {}),
    ...(body.resume !== undefined ? { resume: body.resume } : {}),
    ...(body.start_fresh !== undefined ? { start_fresh: body.start_fresh } : {}),
  };
}

async function parseJson<T>(response: Response): Promise<T> {
  const text = await response.text();
  const data = text ? (JSON.parse(text) as unknown) : null;
  if (!response.ok) {
    throw new ApiClientError(
      `API ${response.status} ${response.statusText}`,
      response.status,
      data,
    );
  }
  return data as T;
}

async function requestJson<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...init?.headers,
    },
  });
  return parseJson<T>(response);
}

export interface HealthResponse {
  status: string;
  phase?: number;
}

export async function getHealth(): Promise<HealthResponse> {
  if (USE_MOCKS) {
    return { status: "ok", phase: 0 };
  }
  return requestJson<HealthResponse>("/health");
}

export async function prescan(
  body: PrescanRequestBody,
): Promise<PrescanResponse> {
  if (USE_MOCKS) {
    return asContract<PrescanResponse>(prescanFixture);
  }
  return requestJson<PrescanResponse>("/utils/prescan", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function submitJob(
  body: JobSubmitRequest,
): Promise<JobSubmitResponse> {
  const payload = toJobSubmitPayload(body);
  if (USE_MOCKS) {
    return asContract<JobSubmitResponse>(jobSubmitResponseFixture);
  }
  return requestJson<JobSubmitResponse>("/jobs", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function listJobs(
  projectId?: string,
): Promise<JobStatusResponse[]> {
  if (USE_MOCKS) {
    const jobs = [asContract<JobStatusResponse>(jobStatusPausedFixture)];
    if (!projectId) {
      return jobs;
    }
    return jobs.filter((job) => job.project_id === projectId);
  }
  const query = projectId
    ? `?project_id=${encodeURIComponent(projectId)}`
    : "";
  return requestJson<JobStatusResponse[]>(`/jobs${query}`);
}

export async function getJob(jobId: string): Promise<JobStatusResponse> {
  if (USE_MOCKS) {
    const fixture = asContract<JobStatusResponse>(jobStatusPausedFixture);
    if (jobId !== fixture.job_id) {
      throw new ApiClientError("Job not found", 404);
    }
    return fixture;
  }
  return requestJson<JobStatusResponse>(`/jobs/${encodeURIComponent(jobId)}`);
}

export async function resumeJob(jobId: string): Promise<JobStatusResponse> {
  if (USE_MOCKS) {
    const fixture = asContract<JobStatusResponse>(jobStatusPausedFixture);
    return { ...fixture, job_id: jobId, status: "PROCESSING" };
  }
  return requestJson<JobStatusResponse>(
    `/jobs/${encodeURIComponent(jobId)}/resume`,
    { method: "POST" },
  );
}

export async function startFreshJob(jobId: string): Promise<JobStatusResponse> {
  if (USE_MOCKS) {
    const fixture = asContract<JobStatusResponse>(jobStatusPausedFixture);
    return {
      ...fixture,
      job_id: jobId,
      status: "PENDING",
      checkpoint_exists: false,
    };
  }
  return requestJson<JobStatusResponse>(
    `/jobs/${encodeURIComponent(jobId)}/start-fresh`,
    { method: "POST" },
  );
}

export function subscribeJobTelemetry(
  onMessage: (message: TelemetryMessage) => void,
): () => void {
  if (USE_MOCKS) {
    onMessage(asContract<TelemetryMessage>(telemetryProgressFixture));
    return () => undefined;
  }

  if (typeof WebSocket === "undefined") {
    return () => undefined;
  }

  const wsUrl = `${API_BASE_URL.replace(/^http/, "ws")}/ws/jobs`;
  const socket = new WebSocket(wsUrl);
  socket.addEventListener("message", (event) => {
    const parsed = JSON.parse(String(event.data)) as TelemetryMessage;
    onMessage(parsed);
  });
  return () => socket.close();
}

export const apiClient = {
  getHealth,
  prescan,
  submitJob,
  listJobs,
  getJob,
  resumeJob,
  startFreshJob,
  subscribeJobTelemetry,
  useMocks: USE_MOCKS,
  apiBaseUrl: API_BASE_URL,
};
