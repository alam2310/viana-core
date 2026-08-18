# API Contracts & Data Structures

Human-readable contract reference. Machine-readable schemas: `packages/contracts/schemas/`.

## 1. Container Management (Host — Next.js API Routes)

The UI on the host manages the Docker container. See `docker/orchestrator_config.yaml.example`.

The **container** runs FastAPI on port `8000` (see `docker-compose.yml`).

## 2. Job Ownership

| Field | Set by |
|-------|--------|
| `job_id` | **Backend** on `POST /jobs` |
| `gpu_device` | **Backend** worker pool (`cuda:0` or `cuda:1`) |
| `output_dir` | **Backend** from `output.parent_dir` + `project_id` |
| Calibration lines | **UI** (user-drawn or edited from prescan proposal) |

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

**Response:** `JobSubmitResponse` — see `job_submit_response.schema.json`.

## 6. Job lifecycle endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/jobs` | List jobs (`?project_id=`) |
| GET | `/jobs/{id}` | Status, `checkpoint_exists`, `output_dir` |
| POST | `/jobs/{id}/resume` | Explicit resume from checkpoint |
| POST | `/jobs/{id}/start-fresh` | Delete checkpoint, restart |
| DELETE | `/jobs/{id}` | Cancel worker |
| POST | `/jobs/{id}/aggregate` | Re-build `_15min.csv` |
| WS | `/ws/jobs` | Telemetry stream |

## 7. Calibration profiles

Stored at: `{output.parent_dir}/{project_id}/profiles/{profile_id}.json`

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

Fixtures: `packages/contracts/fixtures/telemetry_progress.json`

## 9. Future task types

`ViAnaNP_Parked` and `ViAna_Junction` are documented for platform context; **not implemented** in engine v0.1.
