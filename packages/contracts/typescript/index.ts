/**
 * Hand-maintained TypeScript types aligned with packages/contracts/schemas.
 * Regenerate or extend when schemas change (Phase 1+).
 */

export type JobStatus =
  | "PRESCAN_PENDING"
  | "PRESCAN_RUNNING"
  | "PRESCAN_FAILED"
  | "AWAITING_REVIEW"
  | "READY"
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

export interface ConfirmedJobMetadata {
  user_start_time: string;
  user_start_date: string;
  location: string;
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
  output_dir?: string;
  metadata?: JobMetadata;
  task_parameters: ViAnaTaskParameters;
  calibration_profile_id?: string;
  resume?: boolean;
  start_fresh?: boolean;
}

/** POST /jobs/intake — register videos for prescan (no calibration yet). */
export interface JobIntakeRequest {
  task_type?: "ViAna_Moving";
  project_id: string;
  source_video_paths: string[];
  output_dir?: string;
}

export interface JobIntakeItem {
  job_id: string;
  status: "PRESCAN_PENDING";
  source_video_path: string;
  output_dir: string;
  queue_position: number;
}

export interface JobIntakeResponse {
  jobs: JobIntakeItem[];
}

/** PATCH /jobs/{id}/prescan — confirm reviewed calibration → READY. */
export interface JobPrescanConfirmRequest {
  metadata: ConfirmedJobMetadata;
  task_parameters: ViAnaTaskParameters;
  calibration_profile_id?: string;
}

/**
 * Engine CLI JSON (`viana run` / `viana resume`).
 * Backend assigns job_id, gpu_device, output_dir — UI must not POST this object.
 */
export interface JobConfig extends JobSubmitRequest {
  job_id: string;
  gpu_device: string;
  output_dir: string;
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
  eta_sec?: number;
  crossing_count?: number;
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
  proposed_metadata?: JobMetadata;
  proposed_lines?: ProposedLines;
  proposed_preview_url?: string | null;
  confirmed_metadata?: JobMetadata;
  confirmed_task_parameters?: ViAnaTaskParameters;
  /** ISO-8601 UTC; set by the API at intake/submit. */
  created_at: string;
  /** Source video length from prescan `video_meta.duration_sec`; null until prescan succeeds. */
  video_duration_sec?: number | null;
  /** GPU wall-clock seconds from first PROCESSING; live while running, frozen when the run ends. */
  processing_duration_sec?: number | null;
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

export type CalibrationProfileSource = "user_drawn" | "auto_proposed" | "user_edited";

/** `{parent_dir}/{project_id}/profiles/{profile_id}.json` */
export interface CalibrationProfile {
  profile_id: string;
  profile_name: string;
  reference_resolution: [number, number];
  horizon_line: LineSegment;
  counting_line: LineSegment;
  created_at?: string;
  source?: CalibrationProfileSource;
}

export interface PrescanRequest {
  source_video_path: string;
  project_id: string;
  task_type?: "ViAna_Moving";
  frame_offset_sec?: number;
}

export interface PrescanResponse {
  prescan_id: string;
  video_meta: VideoMeta;
  ocr: PrescanOCR;
  proposed_lines?: ProposedLines | null;
  preview_url: string;
  profiles?: CalibrationProfile[];
}

export type TelemetryType = "PROGRESS" | "MOVING_EVENT" | "LOG";

export interface TelemetryMessage {
  job_id: string;
  status?: JobStatus;
  telemetry_type: TelemetryType;
  data: Record<string, unknown>;
}

/** YOLO class id → analytics hierarchy (`configs/classes.yaml`). */
export interface VehicleClass {
  id: number;
  name: string;
  category: string;
  class_type: string;
  sub_class: string;
  aggregate: boolean;
}

export interface ClassTaxonomy {
  classes: VehicleClass[];
}

export interface ModelPaths {
  vehicle: string;
  pedestrian: string;
}

export interface DetectionDefaults {
  confidence_threshold: number;
  imgsz: number;
  nms_threshold: number;
  suppression_ioa: number;
}

export interface ClassificationDefaults {
  use_heuristic_truck_split: boolean;
  lock_frames: number;
  perspective_scale: number;
  trailer_ratio: number;
  lcv_max_area: number;
  mcv_max_area: number;
}

export interface OcrDefaults {
  min_confidence: number;
  recalibration_interval_sec: number;
  drift_threshold_sec: number;
}

export interface PipelineDefaults {
  checkpoint_interval_frames: number;
  telemetry_progress_frames: number;
  telemetry_detail_progress_frames: number;
}

export interface OutputDefaults {
  parent_dir: string;
}

/** Engine defaults (`configs/engine_defaults.yaml`). Overridable per job. */
export interface EngineDefaults {
  models: ModelPaths;
  detection: DetectionDefaults;
  classification: ClassificationDefaults;
  ocr: OcrDefaults;
  pipeline: PipelineDefaults;
  output: OutputDefaults;
}

export type WallTimeSource =
  | "ocr_recalibrated"
  | "ocr_anchor"
  | "user_fallback"
  | "unavailable";

export interface TimeAnchor {
  video_pts_ms: number;
  wall_time: string;
  source: WallTimeSource;
  ocr_confidence?: number | null;
  date?: string | null;
  location?: string | null;
}

/** `{stem}.time_map.json` — maps video PTS to wall clock. */
export interface TimeMap {
  schema_version: 1;
  job_id: string;
  video_stem: string;
  anchors: TimeAnchor[];
  user_start_date?: string | null;
  user_start_time?: string | null;
}
