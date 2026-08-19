/**
 * Job API client. Mock path uses packages/contracts/fixtures when
 * NEXT_PUBLIC_USE_MOCKS is not false.
 *
 * Never send job_id or gpu_device on POST /jobs.
 */

import type {
  CalibrationProfile,
  JobIntakeRequest,
  JobIntakeResponse,
  JobPrescanConfirmRequest,
  JobStatusResponse,
  JobSubmitRequest,
  JobSubmitResponse,
  LineSegment,
  Point,
  PrescanResponse,
  TelemetryMessage,
} from "@viana/contracts";
import prescanFixture from "@viana/fixtures/prescan_response.json";
import telemetryProgressFixture from "@viana/fixtures/telemetry_progress.json";

import { API_BASE_URL, USE_MOCKS } from "./env";
import { browserApiBase, browserWsJobsUrl } from "./orchestrator-url";
import {
  mockConfirmPrescan,
  mockGetJob,
  mockIntakeJobs,
  mockListJobs,
  mockRecordSubmit,
  mockResumeJob,
  mockRetryPrescan,
  mockStartFreshJob,
} from "./mock-jobs";

/** Request body documented in docs/api_contracts.md (no TS type in contracts yet). */
export interface PrescanRequestBody {
  source_video_path: string;
  project_id: string;
  frame_offset_sec?: number;
}

export interface AggregateResponse {
  command?: string;
  aggregate_15min?: string;
  [key: string]: unknown;
}

function asContract<T>(value: unknown): T {
  return value as T;
}

function intPoint(point: Point): Point {
  return [Math.round(point[0]), Math.round(point[1])];
}

function intLine(line: LineSegment): LineSegment {
  return { start: intPoint(line.start), end: intPoint(line.end) };
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

export function isCheckpointConflict(error: unknown): boolean {
  return error instanceof ApiClientError && error.status === 409;
}

function formatApiError(status: number, statusText: string, data: unknown): string {
  if (data && typeof data === "object" && "detail" in data) {
    const detail = (data as { detail: unknown }).detail;
    if (typeof detail === "string") {
      return `API ${status}: ${detail}`;
    }
    return `API ${status}: ${JSON.stringify(detail)}`;
  }
  return `API ${status} ${statusText}`;
}

/** Whitelist JobSubmitRequest fields so JobConfig.job_id / gpu_device cannot leak. */
export function toJobSubmitPayload(body: JobSubmitRequest): JobSubmitRequest {
  const params = body.task_parameters;
  return {
    task_type: body.task_type,
    source_video_path: body.source_video_path,
    project_id: body.project_id,
    ...(body.metadata ? { metadata: body.metadata } : {}),
    task_parameters: {
      ...params,
      horizon_line: intLine(params.horizon_line),
      counting_line: intLine(params.counting_line),
    },
    ...(body.calibration_profile_id
      ? { calibration_profile_id: body.calibration_profile_id }
      : {}),
    ...(body.resume !== undefined ? { resume: body.resume } : {}),
    ...(body.start_fresh !== undefined ? { start_fresh: body.start_fresh } : {}),
  };
}

/** Client-owned POST /jobs body. Rejects engine JobConfig fields at the type level. */
export type JobSubmitClientBody = JobSubmitRequest & {
  job_id?: never;
  gpu_device?: never;
  output_dir?: never;
};

/** Prefix orchestrator-relative paths (preview_url) with NEXT_PUBLIC_API_URL. */
export function resolveApiAssetUrl(path: string, cacheBust?: string | number): string {
  if (!path) {
    return path;
  }
  let url: string;
  if (/^https?:\/\//i.test(path)) {
    url = path;
  } else {
    const base = API_BASE_URL.replace(/\/$/, "");
    url = path.startsWith("/") ? `${base}${path}` : `${base}/${path}`;
  }
  if (cacheBust !== undefined) {
    const sep = url.includes("?") ? "&" : "?";
    return `${url}${sep}v=${encodeURIComponent(String(cacheBust))}`;
  }
  return url;
}

/** Same-origin URL for prescan preview JPEGs (avoids cross-origin canvas/img issues). */
export function previewImageUrl(apiPath: string, cacheBust?: string | number): string {
  if (!apiPath) {
    return apiPath;
  }
  let path = apiPath;
  if (/^https?:\/\//i.test(apiPath)) {
    try {
      path = new URL(apiPath).pathname;
    } catch {
      return apiPath;
    }
  }
  const params = new URLSearchParams({ path });
  if (cacheBust !== undefined) {
    params.set("v", String(cacheBust));
  }
  return `/api/proxy/preview?${params}`;
}

async function parseJson<T>(response: Response): Promise<T> {
  const text = await response.text();
  let data: unknown = null;
  if (text) {
    try {
      data = JSON.parse(text) as unknown;
    } catch {
      data = { detail: text };
    }
  }
  if (!response.ok) {
    throw new ApiClientError(
      formatApiError(response.status, response.statusText, data),
      response.status,
      data,
    );
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return data as T;
}

async function requestJson<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const base = USE_MOCKS ? API_BASE_URL : browserApiBase();
  const response = await fetch(`${base}${path}`, {
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
    return { status: "ok", phase: 6 };
  }
  return requestJson<HealthResponse>("/health");
}

export async function intakeJobs(
  body: JobIntakeRequest,
): Promise<JobIntakeResponse> {
  if (USE_MOCKS) {
    return mockIntakeJobs(body);
  }
  return requestJson<JobIntakeResponse>("/jobs/intake", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function confirmPrescan(
  jobId: string,
  body: JobPrescanConfirmRequest,
): Promise<JobStatusResponse> {
  const payload: JobPrescanConfirmRequest = {
    metadata: body.metadata,
    task_parameters: {
      ...body.task_parameters,
      horizon_line: intLine(body.task_parameters.horizon_line),
      counting_line: intLine(body.task_parameters.counting_line),
    },
    ...(body.calibration_profile_id
      ? { calibration_profile_id: body.calibration_profile_id }
      : {}),
  };
  if (USE_MOCKS) {
    const job = mockConfirmPrescan(jobId, payload);
    if (!job) {
      throw new ApiClientError("Job not found", 404);
    }
    return job;
  }
  return requestJson<JobStatusResponse>(
    `/jobs/${encodeURIComponent(jobId)}/prescan`,
    { method: "PATCH", body: JSON.stringify(payload) },
  );
}

export async function retryPrescan(jobId: string): Promise<JobStatusResponse> {
  if (USE_MOCKS) {
    const job = mockRetryPrescan(jobId);
    if (!job) {
      throw new ApiClientError("Job not found", 404);
    }
    return job;
  }
  return requestJson<JobStatusResponse>(
    `/jobs/${encodeURIComponent(jobId)}/prescan/retry`,
    { method: "POST" },
  );
}

export async function prescanPreview(
  jobId: string,
  frameOffsetSec = 0,
): Promise<PrescanResponse> {
  if (USE_MOCKS) {
    return asContract<PrescanResponse>(prescanFixture);
  }
  const query = `?frame_offset_sec=${encodeURIComponent(String(frameOffsetSec))}`;
  return requestJson<PrescanResponse>(
    `/jobs/${encodeURIComponent(jobId)}/prescan/preview${query}`,
  );
}

export function partialVideoUrl(jobId: string, cacheBust?: string | number): string {
  const path = `/artifacts/${encodeURIComponent(jobId)}/partial.mp4`;
  const params = new URLSearchParams({ path });
  if (cacheBust !== undefined) {
    params.set("v", String(cacheBust));
  }
  return `/api/proxy/partial?${params}`;
}

/** Same-origin URL for intake source MP4 (browser seek + canvas drawImage). */
export function sourceVideoUrl(jobId: string): string {
  const path = `/artifacts/${encodeURIComponent(jobId)}/source.mp4`;
  const params = new URLSearchParams({ path });
  return `/api/proxy/source?${params}`;
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
  body: JobSubmitClientBody,
): Promise<JobSubmitResponse> {
  const payload = toJobSubmitPayload(body);
  if (USE_MOCKS) {
    return mockRecordSubmit(payload);
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
    return mockListJobs(projectId);
  }
  const query = projectId
    ? `?project_id=${encodeURIComponent(projectId)}`
    : "";
  return requestJson<JobStatusResponse[]>(`/jobs${query}`);
}

export async function getJob(jobId: string): Promise<JobStatusResponse> {
  if (USE_MOCKS) {
    const job = mockGetJob(jobId);
    if (!job) {
      throw new ApiClientError("Job not found", 404);
    }
    return job;
  }
  return requestJson<JobStatusResponse>(`/jobs/${encodeURIComponent(jobId)}`);
}

export async function resumeJob(jobId: string): Promise<JobSubmitResponse> {
  if (USE_MOCKS) {
    const job = mockResumeJob(jobId);
    if (!job) {
      throw new ApiClientError("Job not found", 404);
    }
    return {
      job_id: job.job_id,
      status: job.status,
      gpu_device: job.gpu_device ?? "cuda:0",
      queue_position: job.queue_position ?? 0,
      output_dir: job.output_dir,
    };
  }
  return requestJson<JobSubmitResponse>(
    `/jobs/${encodeURIComponent(jobId)}/resume`,
    { method: "POST" },
  );
}

export async function startFreshJob(jobId: string): Promise<JobSubmitResponse> {
  if (USE_MOCKS) {
    const job = mockStartFreshJob(jobId);
    if (!job) {
      throw new ApiClientError("Job not found", 404);
    }
    return {
      job_id: job.job_id,
      status: job.status,
      gpu_device: job.gpu_device ?? "cuda:0",
      queue_position: job.queue_position ?? 0,
      output_dir: job.output_dir,
    };
  }
  return requestJson<JobSubmitResponse>(
    `/jobs/${encodeURIComponent(jobId)}/start-fresh`,
    { method: "POST" },
  );
}

export async function cancelJob(jobId: string): Promise<void> {
  if (USE_MOCKS) {
    return;
  }
  await requestJson<void>(`/jobs/${encodeURIComponent(jobId)}`, {
    method: "DELETE",
  });
}

export async function aggregateJob(jobId: string): Promise<AggregateResponse> {
  if (USE_MOCKS) {
    return { command: "aggregate", aggregate_15min: "mock_15min.csv" };
  }
  return requestJson<AggregateResponse>(
    `/jobs/${encodeURIComponent(jobId)}/aggregate`,
    { method: "POST" },
  );
}

export async function listProfiles(
  projectId: string,
): Promise<CalibrationProfile[]> {
  if (USE_MOCKS) {
    return asContract<PrescanResponse>(prescanFixture).profiles ?? [];
  }
  return requestJson<CalibrationProfile[]>(
    `/projects/${encodeURIComponent(projectId)}/profiles`,
  );
}

export async function saveProfile(
  projectId: string,
  profile: CalibrationProfile,
): Promise<CalibrationProfile> {
  if (USE_MOCKS) {
    return profile;
  }
  return requestJson<CalibrationProfile>(
    `/projects/${encodeURIComponent(projectId)}/profiles`,
    { method: "POST", body: JSON.stringify(profile) },
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

  const wsUrl = browserWsJobsUrl();
  let closed = false;
  let socket: WebSocket | null = null;
  let reconnectTimer: ReturnType<typeof setTimeout> | undefined;

  function connect() {
    if (closed) {
      return;
    }
    socket = new WebSocket(wsUrl);
    socket.addEventListener("message", (event) => {
      const parsed = JSON.parse(String(event.data)) as TelemetryMessage;
      onMessage(parsed);
    });
    socket.addEventListener("close", () => {
      if (!closed) {
        reconnectTimer = setTimeout(connect, 2000);
      }
    });
  }

  connect();
  return () => {
    closed = true;
    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
    }
    socket?.close();
  };
}

export const apiClient = {
  get apiBaseUrl() {
    return USE_MOCKS ? API_BASE_URL : browserApiBase();
  },
  useMocks: USE_MOCKS,
  getHealth,
  intakeJobs,
  confirmPrescan,
  retryPrescan,
  prescanPreview,
  previewImageUrl,
  partialVideoUrl,
  sourceVideoUrl,
  prescan,
  submitJob,
  listJobs,
  getJob,
  resumeJob,
  startFreshJob,
  cancelJob,
  aggregateJob,
  listProfiles,
  saveProfile,
  subscribeJobTelemetry,
  resolveApiAssetUrl,
};
