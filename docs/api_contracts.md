# API Contracts & Data Structures

Human-readable contract reference. Machine-readable schemas: `packages/contracts/schemas/`. OpenAPI: [`openapi.yaml`](../openapi.yaml).

## 1. Container Management (Host — Next.js API Routes)

The UI on the host manages the Docker container. See `docker/orchestrator_config.yaml.example`.

The **container** runs FastAPI on port `8000` (see `docker-compose.yml`).

**Same-origin proxies** (Next.js App Router, whitelist paths only):

| Route | Upstream | Consumer |
|-------|----------|----------|
| `GET /api/proxy/preview?path=` | Orchestrator prescan JPEG | Calibration canvas initial frame |
| `GET /api/proxy/source?path=` | `GET /artifacts/{id}/source.mp4` (forwards `Range`) | Prescan review frame scrub |
| `GET /api/proxy/partial?path=` | `GET /artifacts/{id}/partial.mp4` (forwards `Range`; `Accept-Ranges` / `Content-Disposition: inline`) | **PARKED in UI** — endpoint remains; player is not mounted (S24) |

## 2. Job Ownership

| Field | Set by |
|-------|--------|
| `job_id` | **Backend** on `POST /jobs` or `POST /jobs/intake` |
| `gpu_device` | **Backend** worker pool (`cuda:0` or `cuda:1`) |
| `output_dir` | **Backend** default `{output.parent_dir}/{project_id}`; optional **override** on intake/submit (G20) |
| Calibration lines | **UI** confirms via `PATCH /jobs/{id}/prescan` (or direct `POST /jobs` legacy path) |

The UI must **not** send `job_id` or `gpu_device` in job submit requests.

## 3. Output Paths

```
{output.parent_dir}/{project_id}/
  {video_stem}_events.csv
  {video_stem}_15min.csv
  {video_stem}_processed.mp4
  {video_stem}.manifest.json
  ...
```

Default `output.parent_dir`: `/data/viana-outputs` (`configs/engine_defaults.yaml`).

## 4. POST /utils/prescan

**Request:**
```json
{
  "source_video_path": "/data/projects/nh48/videos/2026-03-15_09-00.mp4",
  "project_id": "nh48",
  "frame_offset_sec": 0.0
}
```

**Response:** See `packages/contracts/schemas/prescan_response.schema.json` and fixture `packages/contracts/fixtures/prescan_response.json`.

UI displays `preview_url` on canvas with `proposed_lines` overlaid; user may edit before submit.

Engine CLI: `python -m viana prescan --source … --project-id … [--frame-offset] [--output-dir]`. Stdout is `PrescanResponse` JSON. `preview_url` is the disk path `{output_dir}/prescan/{prescan_id}_preview.jpg` (orchestrator rewrites this to an HTTP URL). Profiles are listed from `{output_dir}/profiles/*.json`.

## 5. POST /jobs — JobSubmitRequest

```json
{
  "task_type": "ViAna_Moving",
  "source_video_path": "/data/projects/nh48/videos/2026-03-15_09-00.mp4",
  "project_id": "nh48",
  "metadata": {
    "user_start_time": "09:00:00",
    "user_start_date": "15-03-2026",
    "location": "NH48 Km42"
  },
  "task_parameters": {
    "horizon_line": { "start": [120, 400], "end": [1800, 520] },
    "counting_line": { "start": [80, 900], "end": [1840, 780] },
    "confidence_threshold": 0.75,
    "use_heuristic_truck_split": true,
    "render_video": true,
    "telemetry_detail": false
  },
  "calibration_profile_id": "morning_northbound",
  "resume": false,
  "start_fresh": false
}
```

**Response:** `JobSubmitResponse` — see `job_submit_response.schema.json` and fixture `packages/contracts/fixtures/job_submit_response.json`.

## 5b. POST /jobs/intake — JobIntakeRequest

Register one or more video paths for backend prescan. Creates jobs at `PRESCAN_PENDING` (Step 3 worker runs prescan → `AWAITING_REVIEW`).

```json
{
  "task_type": "ViAna_Moving",
  "project_id": "nh48",
  "source_video_paths": [
    "/data/projects/nh48/videos/2026-03-15_09-00.mp4",
    "/data/projects/nh48/videos/2026-03-15_10-00.mp4"
  ],
  "output_dir": "/data/custom-outputs/nh48"
}
```

`output_dir` is optional (G20). Response: `job_intake_response.schema.json`, fixture `job_intake_response.json`.

**Path validation (Step 6.7 / S09 / F006):** each `source_video_paths` entry must be readable inside the processing container. Host paths that match compose bind-mounts are rewritten (`./data` → `/data`, repo → `/app/ViAna`) and stored as the container path. Unmapped paths return **400** (the whole batch fails) instead of creating jobs that later `PRESCAN_FAILED` with `Video not found`. Allowed roots: `VIANA_INTAKE_ROOTS` (default `/data:/app/ViAna`). Extra drives: bind-mount the volume and set `VIANA_EXTRA_INTAKE_ROOT` + `VIANA_PATH_MAPS=host->container`. The UI already translates via `apps/web/src/lib/container-paths.ts`; this check defends curl and other API clients. Same rewrite/reject applies to `source_video_path` on `POST /jobs`.

## 5c. PATCH /jobs/{id}/prescan — JobPrescanConfirmRequest

Operator confirms reviewed OCR + lines → job transitions to `READY` and enters the GPU FIFO queue.

```json
{
  "metadata": {
    "user_start_time": "09:00:00",
    "user_start_date": "15-03-2026",
    "location": "NH48 Km42"
  },
  "task_parameters": {
    "horizon_line": { "start": [120, 400], "end": [1800, 520] },
    "counting_line": { "start": [80, 900], "end": [1840, 780] },
    "confidence_threshold": 0.75,
    "use_heuristic_truck_split": true,
    "render_video": true,
    "telemetry_detail": false
  },
  "calibration_profile_id": "morning_northbound"
}
```

**Validation (G4):** all three metadata fields mandatory; `user_start_time` must match `HH:MM:SS`; `user_start_date` must match `DD-MM-YYYY`.

Allowed job statuses: `AWAITING_REVIEW`, `READY` (re-review before `PROCESSING`).

Response: full `JobStatus` with `confirmed_metadata` and `confirmed_task_parameters`. Schema: `job_prescan_confirm.schema.json`.

## 6. Job lifecycle endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/jobs/intake` | Register path(s) → `PRESCAN_PENDING` (400 if path not in a container bind-mount) |
| PATCH | `/jobs/{id}/prescan` | Confirm review → `READY` |
| POST | `/jobs/{id}/prescan/retry` | `PRESCAN_FAILED` → `PRESCAN_PENDING` |
| GET | `/jobs/{id}/prescan/preview` | Re-run prescan OCR at `frame_offset_sec` (**Re-scan OCR** button only; frame scrub uses `source.mp4`) |
| GET | `/artifacts/{id}/source.mp4` | Intake source MP4 with HTTP Range (prescan review scrub; `PRESCAN_*`, `AWAITING_REVIEW`, `READY`) |
| GET | `/artifacts/{id}/partial.mp4` | Partial `_processed.mp4` with HTTP Range (live monitor) |
| GET | `/jobs` | List jobs (`?project_id=`) |
| GET | `/jobs/{id}` | Status — see `job_status.schema.json`, fixtures `job_status_paused.json`, `job_status_awaiting_review.json` |

**JobStatus timing fields** (queue columns; do not use UI localStorage for these):

| Field | When set | Notes |
|-------|----------|--------|
| `created_at` | Intake or `POST /jobs` | Required ISO-8601 UTC (`…Z`). Sort key for submitted time. |
| `video_duration_sec` | Prescan success | Copied from prescan `video_meta.duration_sec` (**seconds of source playback**, not frames). Engine corrects MPEG-PS/DVR headers via ffprobe when implied bitrate is implausible (`src/viana/io/media.py`). `null` until prescan succeeds. |
| `processing_duration_sec` | First `PROCESSING` | Live elapsed GPU wall-clock while running; frozen on `COMPLETED` / `FAILED` / `CANCELLED` / `PAUSED`. `null` before the GPU run starts. |
| `progress.eta_sec` | `PROCESSING` telemetry | Wall-clock remaining: `(total_frames - current_frame) / processing_fps`. `processing_fps` is decoded frames per wall-clock second, **not** `video_meta.fps`. |

| Method | Path | Description |
|--------|------|-------------|
| POST | `/jobs/{id}/pause` | Cooperative pause → checkpoint → `PAUSED` |
| POST | `/jobs/{id}/resume` | Explicit resume from checkpoint (`PAUSED` only) |
| POST | `/jobs/{id}/start-fresh` | Delete checkpoint, restart |
| DELETE | `/jobs/{id}` | Cancel job |
| POST | `/jobs/{id}/aggregate` | Re-build `_15min.csv` |
| WS | `/ws/jobs` | Telemetry stream |

## 7. Calibration profiles

Stored at: `{output.parent_dir}/{project_id}/profiles/{profile_id}.json` (schema `calibration_profile.schema.json`, fixture `calibration_profile.json`). Engine load/save is `viana.io.profiles`; HTTP routes remain orchestrator.

| Method | Path |
|--------|------|
| GET | `/projects/{project_id}/profiles` |
| POST | `/projects/{project_id}/profiles` |

## 8. WebSocket telemetry

```json
{
  "job_id": "job_abc",
  "status": "PROCESSING",
  "telemetry_type": "PROGRESS | MOVING_EVENT | LOG",
  "data": {}
}
```

**JobStatus values:** `PRESCAN_PENDING`, `PRESCAN_RUNNING`, `PRESCAN_FAILED`, `AWAITING_REVIEW`, `READY`, `PROCESSING`, `PAUSED`, `COMPLETED`, `FAILED`, `CANCELLED`. Legacy `PENDING` removed.

Fixtures: `packages/contracts/fixtures/telemetry_progress.json`

## 9. Engine disk artifacts (not HTTP)

Written under `{output_dir}/` per video stem. Schemas:

| File | Schema |
|------|--------|
| `{stem}_events.csv` | `events_raw.schema.json` |
| `{stem}_15min.csv` | `events_15min.schema.json` |
| `{stem}.checkpoint.json` | `checkpoint.schema.json` |
| `{stem}.run_result.json` | `run_result.schema.json` |

**CSV headers (S32):** properties order in those schemas is the file header.

- `_events.csv`: `event_id,job_id,video_file,track_id,frame_index,video_pts_ms,wall_time,wall_time_source,date,location,class_id,class_name,direction,confidence`
- `_15min.csv`: `window_start,window_end,date,location,class_name,direction,count,partial` (`window_*` are `HH:MM`; `date` required — S15). Pedestrian is a `class_name` when `aggregate: true` (S33).

Fixture: `packages/contracts/fixtures/checkpoint_resume.json`. Time map fixture: `time_map.json`.

## 10. Engine CLI JobConfig (not HTTP)

`python -m viana run --config job.json` and `viana resume` read `job_config.schema.json`.

The orchestrator writes this file after assigning `job_id`, `gpu_device`, and `output_dir`. The UI must **not** send `JobConfig` on `POST /jobs` (use `job_submit.schema.json` only).

`python -m viana run --config job.json` processes the video (events CSV, checkpoint, time map, optional FFmpeg `{stem}_processed.mp4`). `viana resume` continues from `{stem}.checkpoint.json` only when `resume=true`. Telemetry JSON lines go to **stderr**; the final `RunResult` is stdout.

Do **not** compute 15-minute bins in this loop — use `viana aggregate` (ADR 001).

Fixture: `packages/contracts/fixtures/job_config.json`.

## 11. Future task types

`ViAnaNP_Parked` and `ViAna_Junction` are documented for platform context; **not implemented** in engine v0.1.
