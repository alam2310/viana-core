# User Flows (ViAna Moving Count)

## Flow 1 — First launch

1. User opens UI → settings page
2. UI reads `orchestrator_config.yaml` (host)
3. `GET /api/container/status` → if down, offer **Start container**
4. `GET http://localhost:8000/health` → confirm API ready

## Flow 2 — New project batch

1. User enters `project_id` (slug, e.g. `nh48`)
2. User adds video paths to queue (local paths, no upload)
3. For each video (or first as template):
   - **Prescan** → review OCR → scrub frame if needed
   - **Canvas** → view/edit proposed lines
   - Optional: save as profile under project
   - Optional: apply lines to all pending videos
4. Submit → `POST /jobs` per video
5. Backend returns `job_id`, assigns GPU

## Flow 3 — Monitor processing

1. Dashboard shows queue synced from `GET /jobs`
2. WebSocket `/ws/jobs` for progress + crossing events
3. Detail telemetry: user toggles `telemetry_detail` before submit (or per-job)

## Flow 4 — Paused / failed job

1. `GET /jobs/{id}` → `status: PAUSED`, `checkpoint_exists: true`
2. UI highlights video card
3. User chooses:
   - **Resume** → `POST /jobs/{id}/resume`
   - **Start fresh** → `POST /jobs/{id}/start-fresh`

## Flow 5 — Completed job

1. `status: COMPLETED`
2. UI links to `{output_dir}/{stem}_events.csv`, `_15min.csv`, `_processed.mp4`
3. User may trigger `POST /jobs/{id}/aggregate` to rebuild 15-min report
