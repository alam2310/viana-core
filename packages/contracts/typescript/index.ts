/**
 * Hand-maintained TypeScript types aligned with packages/contracts/schemas.
 * Regenerate or extend when schemas change (Phase 1+).
 */

export type JobStatus =
  | "PENDING"
  | "PROCESSING"
  | "PAUSED"
  | "COMPLETED"
  | "FAILED"
  | "CANCELLED";

export type Point = [number, number];

export interface LineSegment {
  start: Point;
  end: Point;
}

export interface JobMetadata {
  user_start_time?: string;
  user_start_date?: string;
  location?: string;
}

export interface ViAnaTaskParameters {
  horizon_line: LineSegment;
  counting_line: LineSegment;
  confidence_threshold?: number;
  use_heuristic_truck_split?: boolean;
  render_video?: boolean;
  telemetry_detail?: boolean;
}

/** UI → orchestrator. Do NOT send job_id or gpu_device. */
export interface JobSubmitRequest {
  task_type: "ViAna_Moving";
  source_video_path: string;
  project_id: string;
  metadata?: JobMetadata;
  task_parameters: ViAnaTaskParameters;
  calibration_profile_id?: string;
  resume?: boolean;
  start_fresh?: boolean;
}

/** Orchestrator → UI after POST /jobs */
export interface JobSubmitResponse {
  job_id: string;
  status: JobStatus;
  gpu_device: string;
  queue_position: number;
  output_dir: string;
}

export interface JobProgress {
  current_frame: number;
  total_frames: number;
  processing_fps?: number;
}

/** GET /jobs/{id} response */
export interface JobStatusResponse {
  job_id: string;
  status: JobStatus;
  task_type: "ViAna_Moving";
  source_video_path: string;
  project_id: string;
  output_dir: string;
  checkpoint_exists: boolean;
  gpu_device?: string;
  queue_position?: number;
  progress?: JobProgress;
  error_message?: string | null;
}

export interface Checkpoint {
  schema_version: 1;
  job_id: string;
  project_id: string;
  source_video_path: string;
  video_stem: string;
  current_frame: number;
  total_frames: number;
  counted_track_ids?: number[];
  events_rows_written?: number;
  manifest_path?: string;
  saved_at: string;
}

export interface RunResultArtifacts {
  events?: string;
  aggregate_15min?: string;
  processed_video?: string;
  manifest?: string;
  time_map?: string;
}

export interface RunResult {
  schema_version: 1;
  job_id: string;
  status: "COMPLETED" | "FAILED" | "CANCELLED";
  source_video_path: string;
  video_stem: string;
  artifacts: RunResultArtifacts;
  error_message?: string | null;
  completed_at: string;
}

export interface VideoMeta {
  width: number;
  height: number;
  fps: number;
  duration_sec: number;
  frame_count: number;
}

export interface PrescanOCR {
  time?: string | null;
  date?: string | null;
  location?: string | null;
  confidence?: number | null;
}

export interface ProposedLines {
  horizon_line: LineSegment;
  counting_line: LineSegment;
  confidence: number;
}

export interface PrescanResponse {
  prescan_id: string;
  video_meta: VideoMeta;
  ocr: PrescanOCR;
  proposed_lines?: ProposedLines | null;
  preview_url: string;
  profiles?: unknown[];
}

export type TelemetryType = "PROGRESS" | "MOVING_EVENT" | "LOG";

export interface TelemetryMessage {
  job_id: string;
  status?: JobStatus;
  telemetry_type: TelemetryType;
  data: Record<string, unknown>;
}
